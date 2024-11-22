import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal,Categorical

import numpy as np
from Buffer import Buffer # 与DQN.py中的Buffer一样

from copy import deepcopy
import pettingzoo
import gymnasium as gym
import importlib
import argparse
from torch.utils.tensorboard import SummaryWriter
import time

import random
from Attention import  Attention_Critic

''' 此discrete 改自官方代码 暂时先实现原论文的discrete,连续域的之后改
与普通的自改的MASAC (离散)有以下区别
1. 训练的action 从 标量action_dim -> one_hot类型 例：环境action_dim=4 , [1] -> [0,1,0,0]
2. critic 的输入从只输入obs_dim -> obs_dim+action_dim
3. entropy的计算方式 从离散的计算方式-torch.sum(probs * log_probs, dim=1, keepdim=True) -> -log_prob.gather(1, action.reshape(-1,1)) 
4. 更新actor时,actor_loss 多乘一个log_pi 
''' 

from torch.autograd import Variable
## 第一部分：定义Agent类
def categorical_sample(probs, use_cuda=False):
    int_acs = torch.multinomial(probs, 1)
    if use_cuda:
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor
    acs = Variable(tensor_type(*probs.shape).fill_(0)).scatter_(1, int_acs, 1)
    
    return int_acs, acs

class Actor_discrete(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128):
        super(Actor_discrete, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim)

    def forward(self, x ):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = self.l3(x)
        probs = F.softmax(x, dim=1) # 1xaction_dim
        '''
        原论文的实现：等价于 下面 ## ... ## 
        on_gpu = next(self.parameters()).is_cuda
        action, one_hot_action = categorical_sample(probs, use_cuda=on_gpu)
        '''
        ##
        action_dist = Categorical(probs)
        action = action_dist.sample() # action_dim
        one_hot_action = F.one_hot(action, num_classes=probs.size(-1)).float()
        ##
        log_prob = F.log_softmax(x, dim=1) #torch.log(probs + 1e-8)
        log_pi = log_prob.gather(1, action.reshape(-1,1)) # 这里是使用了one-hot 所以这里有这一行代码
        '''
        1.log_probs1 = F.log_softmax(x, dim=1) 1 等价于 2 ,1更稳定
        2.probs = torch.softmax(x, dim=1)
          log_probs2 = torch.log(probs + 1e-8)
        '''
        #entropy = -torch.sum(probs * log_pi, dim=1, keepdim=True)  #
        
        return action ,log_pi ,probs ,one_hot_action
'''
集中式训练Critic
'''    
class Critic(nn.Module):
    def __init__(self, dim_info:dict, hidden_1=128 , hidden_2=128):
        super(Critic, self).__init__()
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())  
        # Q1
        self.l1 = nn.Linear(global_obs_act_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)
        # Q2
        self.l1_2 = nn.Linear(global_obs_act_dim, hidden_1)
        self.l2_2 = nn.Linear(hidden_1, hidden_2)
        self.l3_2 = nn.Linear(hidden_2, 1)


    def forward(self, s, a): # 传入全局观测和动作
        sa = torch.cat(list(s)+list(a), dim = 1)    #batch_size x (obs_dim *agent_num + act_dim * agent_num)
        # #sa = torch.cat([s,a], dim = 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l1_2(sa))
        q2 = F.relu(self.l2_2(q2))
        q2 = self.l3_2(q2)
        return q1, q2
    
class Agent:
    def __init__(self, obs_dim, action_dim, dim_info, actor_lr, critic_lr, device,attention = False):
        
        self.actor = Actor_discrete(obs_dim, action_dim,)
        if attention:
            self.critic = Attention_Critic(dim_info,is_continue=False) #目前attention 只适用于离散动作
        else:
            self.critic = Critic( dim_info ) # 未改完

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

## 第二部分：定义DQN算法类
class Alpha:
    def __init__(self, action_dim, alpha_lr= 0.0001, alpha = 0.01,requires_grad = False,is_continue = True):

        self.log_alpha = torch.tensor(np.log(alpha),dtype = torch.float32, requires_grad=requires_grad) # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        if is_continue:
            self.target_entropy = -action_dim # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper(SAC) 参考原sac论文
        else:
            self.target_entropy =  0.6 * (-torch.log(torch.tensor(1.0 / action_dim))) # 参考:https://zhuanlan.zhihu.com/p/566722896
        self.alpha = self.log_alpha.exp() # 更新actor时无detach会报错,是因为这里只有一个计算图 

    def update_alpha(self, loss):
        self.log_alpha_optimizer.zero_grad()
        loss.backward()
        self.log_alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

''' 选择了MASAC entropy_way_c _a 为1的方式 action_way为1的方式 '''
class MAAC: #先无attention 再加入对比
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, buffer_size, device, trick = None):

        self.one_hot = True
        self.attention = True ###!!!
        self.agents  = {}
        self.buffers = {}
        for agent_id, (obs_dim, action_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, action_dim, dim_info ,actor_lr, critic_lr, device=device,attention = self.attention)
            if self.one_hot:
                self.buffers[agent_id] = Buffer(buffer_size, obs_dim, act_dim = action_dim , device = 'cpu')
            else:
                self.buffers[agent_id] = Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = 'cpu')
            
        self.adaptive_alpha = True
        self.alphas = {} 

        for agent_id, (obs_dim, action_dim) in dim_info.items():
            if self.adaptive_alpha:
                self.alphas[agent_id] = Alpha(action_dim,alpha = 0.01, requires_grad=True, is_continue= is_continue) # Alpha(action_dim).alpha 才是值
            else:
                self.alphas[agent_id] = Alpha(action_dim,alpha = 0.1,requires_grad=False, is_continue= is_continue) 
        
        self.device = device
        self.is_continue = is_continue
        self.agent_x = list(self.agents.keys())[0] #sample 用
    
    def select_action(self, obs):
        actions = {}
        one_hot_actions = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue: # dqn 无此项
                action , _ = self.agents[agent_id].actor(obs)
                actions[agent_id] = action.detach().cpu().numpy().squeeze() # 1xaction_dim -> action_dim
            else:
                action , _ , _ ,one_hot_action = self.agents[agent_id].actor(obs)
                actions[agent_id] = action.detach().cpu().numpy().squeeze() # 1xaction_dim -> action_dim
                one_hot_actions[agent_id] = one_hot_action.detach().cpu().numpy().squeeze()
        return actions ,one_hot_actions
    
    def add(self, obs, action, reward, next_obs, done):
        for agent_id, buffer in self.buffers.items():
            buffer.add(obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id])

    def sample(self, batch_size):
        total_size = len(self.buffers[self.agent_x])
        indices = np.random.choice(total_size, batch_size, replace=False)

        obs, action, reward, next_obs, done = {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id] = buffer.sample(indices)

        return obs, action, reward, next_obs, done #包含所有智能体的数据

    ## SAC算法相关
    def learn(self, batch_size ,gamma , tau):
        # 多智能体特有-- 集中式训练critic:计算next_q值时,要用到所有智能体next状态和动作
        for agent_id, agent in self.agents.items():
            ## 更新前准备
            obs, action, reward, next_obs, done = self.sample(batch_size) # 必须放for里，否则报二次传播错，原因是原来的数据在计算图中已经被释放了
            next_action = {}
            next_log_pi = {}   
            next_probs = {}
            for agent_id_, agent_ in self.agents.items(): # agent_ 和上述agent要区分
                _, next_log_pi_i ,next_probs_i , next_one_hot_action_i = agent_.actor_target(next_obs[agent_id_])
                next_action[agent_id_] = next_one_hot_action_i
                next_log_pi[agent_id_] = next_log_pi_i
                next_probs[agent_id_] = next_probs_i

            if self.attention:
                q  , _ = agent.critic_target(next_obs.values(), next_action.values(), self.agents) # batch_size x 1
                q_next_target = q 
            else: 
                q1_next_target,q2_next_target = agent.critic_target(next_obs.values(), next_action.values()) # batch_size x 1
                q_next_target = torch.min(q1_next_target, q2_next_target)

            entropy_next = - next_log_pi[agent_id]#-torch.sum(next_probs[agent_id] * next_log_pi[agent_id] ,dim=1,keepdim=True)

            # 先更新critic
            ''' 公式: LQ_w = E_{s,a,r,s',d}[(Q_w(s,a) - (r + gamma * (1 - d) * (Q_w'(s',a') - alpha * log_pi_a(s',a')))^2] '''
            q_target = reward[agent_id] + gamma * (1 - done[agent_id]) * (q_next_target + self.alphas[agent_id].alpha.detach() * entropy_next)  
            if self.attention:
                q  , _ = agent.critic(obs.values(), action.values(), self.agents)
                critic_loss = F.mse_loss(q, q_target.detach())
            else: 
                q1, q2 = agent.critic(obs.values(), action.values())
                critic_loss = F.mse_loss(q1, q_target.detach()) + F.mse_loss(q2, q_target.detach())
            agent.update_critic(critic_loss)

            ## 再更新actor
            '''公式: Lpi_θ = E_{s,a ~ D}[-Q_w(s,a) + alpha * log_pi_a(s,a)]  
            理解为 最大化函数V,V = Q + alpha * H
            ***注*** MAAC原代码（离散域）及原论文公式的中 相比于SAC,多乘了log_pi,否则不收敛。
            '''
            new_action = {agent_id: agent.actor(obs[agent_id])[3] for agent_id, agent in self.agents.items()}
            _ , log_pi ,probs , _= agent.actor(obs[agent_id])
            entropy = -log_pi #-torch.sum(probs * log_pi, dim=1, keepdim=True) #- agent.actor(obs[agent_id])[1] * agent.actor(obs[agent_id])[2]
            if self.attention:
                q_pi , all_q = agent.critic(obs.values(), new_action.values(),self.agents)
                v = (all_q * probs).sum(dim=1, keepdim=True)
                pol_target = q_pi - v
                actor_loss = (log_pi*(- pol_target - self.alphas[agent_id].alpha.detach() * entropy)).mean()  #这里alpha一定要加detach(),因为在更新critic时,计算图被丢掉了
            else:
                q1_pi, q2_pi = agent.critic(obs.values(), new_action.values())
                q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (log_pi * (- q_pi - self.alphas[agent_id].alpha.detach() * entropy)).mean()  #这里alpha一定要加detach(),因为在更新critic时,计算图被丢掉了

            
            agent.update_actor(actor_loss)

            ## 更新alpha
            '''公式: Lα = E_{s,a ~ D} [-α * log_pi_a(s,a) - α * H] = E_{s,a ~ D} [α * (-log_pi_a(s,a) - H)]'''
            if self.adaptive_alpha:
                alpha_loss = (self.alphas[agent_id].alpha * (entropy - self.alphas[agent_id].target_entropy).detach()).mean()
                self.alphas[agent_id].update_alpha(alpha_loss)


        ## 更新所有target网络
        self.update_target(tau)

    def update_target(self, tau):
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        for agent in self.agents.values():
            soft_update(agent.actor_target, agent.actor, tau)
            soft_update(agent.critic_target, agent.critic, tau)

    def save(self, model_path):
        ploicy_name = 'MAAC' if self.attention else 'MAAC'
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            os.path.join(model_path, f'{ploicy_name}.pth')
        )

    ## 加载模型
    @staticmethod 
    def load(dim_info, model_dir):
        policy = MAAC(dim_info, is_continue = False, actor_lr = 0, critic_lr = 0, buffer_size = 0, device = 'cpu')
        ploicy_name = 'MAAC' if policy.attention else 'MAAC'
        torch.load(
            os.path.join(model_dir, f'{ploicy_name}.pth')
        )
        return policy
    

## 第三部分 main函数
## 环境配置
def get_env(env_name,env_agent_n = None,continuous_actions=False):
    # 动态导入环境
    module = importlib.import_module(f'pettingzoo.mpe.{env_name}')
    print('env_agent_n or num_good:',env_agent_n) 
    if env_agent_n is None: #默认环境
        env = module.parallel_env(max_cycles=25, continuous_actions=continuous_actions)
    elif env_name == 'simple_spread_v3' or 'simple_adversary_v3': 
        env = module.parallel_env(max_cycles=25, continuous_actions=continuous_actions, N = env_agent_n)
    elif env_name == 'simple_tag_v3': 
        env = module.parallel_env(max_cycles=25, continuous_actions=continuous_actions, num_good= env_agent_n, num_adversaries=3)
    elif env_name == 'simple_world_comm_v3':
        env = module.parallel_env(max_cycles=25, continuous_actions=continuous_actions, num_good= env_agent_n, num_adversaries=4)
    env.reset()
    dim_info = {}
    for agent_id in env.agents:
        dim_info[agent_id] = []
        if isinstance(env.observation_space(agent_id), gym.spaces.Box):
            dim_info[agent_id].append(env.observation_space(agent_id).shape[0])
        else:
            dim_info[agent_id].append(1)
        if isinstance(env.action_space(agent_id), gym.spaces.Box):
            dim_info[agent_id].append(env.action_space(agent_id).shape[0])
        else:
            dim_info[agent_id].append(env.action_space(agent_id).n)
    if continuous_actions:
        return env,dim_info, 1, True # pettingzoo.mpe 环境中，max_action均为1 , 选取连续环境is_continue = True
    else:
        return env,dim_info, None, False

## make_dir 与DQN.py 里一样
def make_dir(env_name,policy_name = 'DQN',trick = None):
    script_dir = os.path.dirname(os.path.abspath(__file__)) # 当前脚本文件夹
    env_dir = os.path.join(script_dir,'./results', env_name)
    os.makedirs(env_dir) if not os.path.exists(env_dir) else None
    print('trick:',trick)
    # 确定前缀
    if trick is None or not any(trick.values()):
        prefix = policy_name + '_' 
    else:
        prefix = policy_name + '_'
        for key in trick.keys():
            if trick[key]:
                prefix += key + '_'
    # 查找现有的文件夹并确定下一个编号
    existing_dirs = [d for d in os.listdir(env_dir) if d.startswith(prefix)]
    max_number = 0 if not existing_dirs else max([int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()])
    model_dir = os.path.join(env_dir, prefix + str(max_number + 1))
    os.makedirs(model_dir)
    return model_dir

''' 
环境见:simple_adversary_v3,simple_crypto_v3,simple_push_v3,simple_reference_v3,simple_speaker_listener_v3,simple_spread_v3,simple_tag_v3
具体见:https://pettingzoo.farama.org/environments/mpe
注意：环境中N个智能体的设置
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="simple_spread_v3")  # alpha = 0.1 for N=5
    parser.add_argument("--N", type=int, default=5) # 环境中智能体数量 默认None 这里用来对比设置
    # 共有参数
    parser.add_argument("--seed", type=int, default=0) # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(600))
    parser.add_argument("--start_steps", type=int, default=500) # 满足此开始更新
    parser.add_argument("--random_steps", type=int, default=0)  #dqn 无此参数 满足此开始自己探索
    parser.add_argument("--learn_steps_interval", type=int, default=1)
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    ## AC参数 
    parser.add_argument("--actor_lr", type=float, default=1e-3) # 1e-3 #
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    ## buffer参数   
    parser.add_argument("--buffer_size", type=int, default=1e6) #1e6默认是float,在bufffer中有int强制转换
    parser.add_argument("--batch_size", type=int, default=256)  #保证比start_steps小
    # trick参数
    parser.add_argument("--policy_name", type=str, default='MAAC_discrete')
    parser.add_argument("--trick", type=dict, default=None) 

    # device参数   
    parser.add_argument("--device", type=str, default='cpu') # cpu/cuda 

    args = parser.parse_args()

    print(args)
    print('-' * 50)
    print('Algorithm:',args.policy_name)

    ## 环境配置
    continuous_actions = False
    env,dim_info,max_action,is_continue = get_env(args.env_name,env_agent_n = args.N,continuous_actions=continuous_actions)
    print(f'Env:{args.env_name}  dim_info:{dim_info}  max_action:{max_action}  max_episodes:{args.max_episodes}')
    
    ## 随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ## cuda
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random Seed:',args.seed)
    # random.seed(args.seed)
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    # torch.use_deterministic_algorithms(True)

    ## 保存model文件夹
    model_dir = make_dir(args.env_name,policy_name = args.policy_name,trick=args.trick)
    writer = SummaryWriter(model_dir)
    print('model_dir:',model_dir)

    ## device参数
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    ## 算法配置
    policy = MAAC(dim_info, is_continue, args.actor_lr, args.critic_lr, args.buffer_size, device, args.trick)

    time_ = time.time()
    ## 训练
    episode_num = 0
    step = 0
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}
    obs,info = env.reset(seed=args.seed)
    {agent: env.action_space(agent).seed(seed=args.seed) for agent in env_agents}  # 针对action复现:env.action_space.sample()
    while episode_num < args.max_episodes:
        step +=1
        # 获取动作
        if step < args.random_steps:
            action = {agent_id: env.action_space(agent_id).sample() for agent_id in env_agents}
            one_hot_actions = {agent_id: np.eye(dim_info[agent_id][1])[action] for agent_id, action in action.items()}
        else:
            if max_action is not None:
                action = policy.select_action(obs)   #[-1,1] -> [0,1] 
                action = {agent_id: (action[agent_id] + 1) / 2 * max_action for agent_id in env_agents}
            else:
                action ,one_hot_actions = policy.select_action(obs)
        action_ = action
        action = one_hot_actions
        
        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
        done_bool = {agent_id: terminated[agent_id]  for agent_id in env_agents} ### truncated 为超过最大步数
        policy.add(obs, action, reward, next_obs, done_bool)
        episode_reward = {agent_id: episode_reward[agent_id] + reward[agent_id] for agent_id in env_agents}
        obs = next_obs
        
        # episode 结束 ### 在pettingzoo中,env.agents 为空时  一个episode结束
        if any(done.values()):
            ## 显示
            if  (episode_num + 1) % 100 == 0:
                print("episode: {}, reward: {}".format(episode_num + 1, episode_reward))
            for agent_id in env_agents:
                writer.add_scalar(f'reward_{agent_id}', episode_reward[agent_id], episode_num + 1)
                train_return[agent_id].append(episode_reward[agent_id])

            episode_num += 1
            obs,info = env.reset(seed=args.seed)
            episode_reward = {agent_id: 0 for agent_id in env_agents}
        
        # 满足step,更新网络
        if step > args.start_steps and step % args.learn_steps_interval == 0:
            policy.learn(args.batch_size, args.gamma, args.tau)

    print('total_time:',time.time()-time_)
    policy.save(model_dir)
    ## 保存数据
    train_return_ = np.array([train_return[agent_id] for agent_id in env.agents])
    if args.N is None:
        np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),train_return_)
    else:
        np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}_N_{len(env_agents)}.npy"),train_return_)
        



