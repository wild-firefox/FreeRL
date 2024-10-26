import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
from Buffer import Buffer # 与DQN.py中的Buffer一样

from copy import deepcopy
import pettingzoo #动态导入
import gymnasium as gym
import importlib
import argparse
from torch.utils.tensorboard import SummaryWriter
import time

## 第一部分：定义Agent类
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.mean_layer = nn.Linear(hidden_2, action_dim)
        self.log_std_layer = nn.Linear(hidden_2, action_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # 我们输出log_std以确保std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # 生成一个高斯分布
        if deterministic:  # 评估时用
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # 方法参考Open AI Spinning up，更稳定。见https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py#L53C12-L53C24
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True) # batch_size x 1
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True) #这里是计算tanh的对数概率，
        else: #常见的其他写法
            '''
            log_pi =  dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= torch.log(1 - torch.tanh(a).pow(2) + 1e-6).sum(dim=1, keepdim=True) # 1e-6是为了数值稳定性 
            '''
            log_pi = None
        
        a =  torch.tanh(a)  # 使用tanh将无界的高斯分布压缩到有界的动作区间内。

        return a, log_pi
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
        sa = torch.cat(list(s)+list(a), dim = 1)
        #sa = torch.cat([s,a], dim = 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l1_2(sa))
        q2 = F.relu(self.l2_2(q2))
        q2 = self.l3_2(q2)
        return q1, q2
    
class Agent:
    def __init__(self, obs_dim, action_dim, dim_info,actor_lr, critic_lr, device):
        
        self.actor = Actor(obs_dim, action_dim, )
        self.critic = Critic( dim_info )

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
    def __init__(self, action_dim, alpha_lr= 0.0001, alpha = 0.2,requires_grad = False):

        self.log_alpha = torch.tensor(np.log(alpha),dtype = torch.float32, requires_grad=requires_grad) # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -action_dim # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper #
        self.alpha = self.log_alpha.exp() # 更新actor时无detach会报错,是因为这里只有一个计算图 

    def update_alpha(self, loss):
        self.log_alpha_optimizer.zero_grad()
        loss.backward()
        self.log_alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

class MASAC: #先无attention 再加入
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, buffer_size, device, trick = None):

        self.attention = False
        self.agents  = {}
        self.buffers = {}
        for agent_id, (obs_dim, action_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, action_dim, dim_info, actor_lr, critic_lr, device=device)
            self.buffers[agent_id] = Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = 'cpu')

        self.adaptive_alpha = True
        self.alphas = {} 

        for agent_id, (obs_dim, action_dim) in dim_info.items():
            if self.adaptive_alpha:
                self.alphas[agent_id] = Alpha(action_dim,alpha = 0.01, requires_grad=True) # Alpha(action_dim).alpha 才是值
            else:
                self.alphas[agent_id] = Alpha(action_dim,alpha = 0.2) # 即 0.2
        '''
        更新critic时 熵的采用方式
        '0' (参考github)中的方式, https://github.com/ffelten/MASAC/blob/main/masac/masac.py#L285  
        '1' MAAC论文中的方式, https://github.com/shariqiqbal2810/MAAC/blob/master/algorithms/attention_sac.py#L99
        '''
        self.entropy_way_c = '1'  # 按w_c,w_a,a_w 顺序'111' > '110' = '001' > '101'  #'111' 则为MAAC更新模式的MASAC 'x10'为MADDPG更新模式的MASAC '001' 则为参考github的MASAC更新模式
        self.entropy_way_a = '1' 
        '''
        更新actor时 动作的采用方式
        '0' MADDPG论文中的方式,  参考 https://github.com/Git-123-Hub/maddpg-pettingzoo-pytorch/blob/master/MADDPG.py#L105 将此方式推广的话, self.entropy_way_a = '1'
        '1' MAAC论文中的方式/(参考github)中的方式 两者 在动作采取一致,但在计算log_pi时有区别
        https://github.com/shariqiqbal2810/MAAC/blob/master/algorithms/attention_sac.py#L139/https://github.com/ffelten/MASAC/blob/main/masac/masac.py#L319
        '''
        self.action_way = '1' 
        self.device = device
        self.is_continue = is_continue
        self.agent_x = list(self.agents.keys())[0] #sample 用
    
    def select_action(self, obs):
        actions = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue: # dqn 无此项
                action , _ = self.agents[agent_id].actor(obs)
                actions[agent_id] = action.detach().cpu().numpy().squeeze(0) # 1xaction_dim -> action_dim
            else:
                action = self.agents[agent_id].argmax(dim = 1).detach().cpu().numpy()[0] # []标量
                actions[agent_id] = action
        return actions
    
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
            ''' 这一部分原理和MADDPG 一样'''
            obs, action, reward, next_obs, done = self.sample(batch_size) # 必须放for里，否则报二次传播错，原因是原来的数据在计算图中已经被释放了
            next_action = {}
            next_log_pi = {}
            for agent_id_, agent_ in self.agents.items():
                next_action_i, next_log_pi_i = agent_.actor_target(next_obs[agent_id_])
                next_action[agent_id_] = next_action_i
                next_log_pi[agent_id_] = next_log_pi_i 

            q1_next_target,q2_next_target = agent.critic_target(next_obs.values(), next_action.values()) # batch_size x 1
            q_next_target = torch.min(q1_next_target, q2_next_target)

            ''' SAC 特有 '0' 参考github 将next_log_pi 求和 来更新critic , '1' MAAC论文 将当前的next_log_pi 用于更新critic '''
            if self.entropy_way_c == '0':
                next_log_pi = torch.stack([next_log_pi[agent_id] for agent_id in self.agents.keys()], dim = 1).sum(dim = 1) # batch_size x 3 x 1 -> batch_size x 1
                entropy_next = - next_log_pi
            elif self.entropy_way_c == '1':
                entropy_next = - next_log_pi[agent_id]

            # 先更新critic
            ''' 公式: LQ_w = E_{s,a,r,s',d}[(Q_w(s,a) - (r + gamma * (1 - d) * (Q_w'(s',a') - alpha * log_pi_a(s',a')))^2] '''
            q_target = reward[agent_id] + gamma * (1 - done[agent_id]) * (q_next_target + self.alphas[agent_id].alpha.detach() * entropy_next)  

            q1, q2 = agent.critic(obs.values(), action.values())
            critic_loss = F.mse_loss(q1, q_target.detach()) + F.mse_loss(q2, q_target.detach())
            agent.update_critic(critic_loss)

            ## 再更新actor
            '''公式: Lpi_θ = E_{s,a ~ D}[-Q_w(s,a) + alpha * log_pi_a(s,a)]  
            理解为 最大化函数V,V = Q + alpha * H
            '''
            if self.action_way == '0':
                new_action, log_pi = agent.actor(obs[agent_id])
                #entropy = - log_pi  # 相当于 self.entropy_way_a == '1'
                action[agent_id] = new_action
                q1_pi, q2_pi = agent.critic(obs.values(), action.values())        
            elif self.action_way == '1':
                new_action = {agent_id: agent.actor(obs[agent_id])[0] for agent_id, agent in self.agents.items()}
                q1_pi, q2_pi = agent.critic(obs.values(), new_action.values())
            
            if self.entropy_way_a == '0':
                new_log_pi = torch.stack([agent.actor(obs[agent_id])[1] for agent_id, agent in self.agents.items()], dim = 1).sum(dim = 1) # batch_size x 3 x 1 -> batch_size x 1
                entropy = - new_log_pi
            elif self.entropy_way_a == '1':
                entropy = - agent.actor(obs[agent_id])[1]


            q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (- q_pi - self.alphas[agent_id].alpha.detach() * entropy).mean()  #这里alpha一定要加detach(),因为在更新critic时,计算图被丢掉了
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
        ploicy_name = 'MAAC' if self.attention else 'MASAC'
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            os.path.join(model_path, f'{ploicy_name}.pth')
        )

    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue, model_dir):
        policy = MASAC(dim_info, is_continue = is_continue, actor_lr = 0, critic_lr = 0, buffer_size = 0, device = 'cpu')
        ploicy_name = 'MAAC' if policy.attention else 'MASAC'
        torch.load(
            os.path.join(model_dir, f'{ploicy_name}.pth')
        )
        return policy
    

## 第三部分 main函数
## 环境配置
def get_env(env_name,env_agent_n = None):
    # 动态导入环境
    module = importlib.import_module(f'pettingzoo.mpe.{env_name}')
    print('env_agent_n or num_good:',env_agent_n) 
    if env_agent_n is None: #默认环境
        env = module.parallel_env(max_cycles=25, continuous_actions=True)
    elif env_name == 'simple_spread_v3' or 'simple_adversary_v3': 
        env = module.parallel_env(max_cycles=25, continuous_actions=True, N = env_agent_n)
    elif env_name == 'simple_tag_v3': 
        env = module.parallel_env(max_cycles=25, continuous_actions=True, num_good= env_agent_n, num_adversaries=3)
    elif env_name == 'simple_world_comm_v3':
        env = module.parallel_env(max_cycles=25, continuous_actions=True, num_good= env_agent_n, num_adversaries=4)
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

    return env,dim_info, 1, True # pettingzoo.mpe 环境中，max_action均为1 , 选取连续环境is_continue = True

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
    existing_dirs = [d for d in os.listdir(env_dir) if d.startswith(prefix) and d[len(prefix):].isdigit()]
    max_number = 0 if not existing_dirs else max([int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()])
    model_dir = os.path.join(env_dir, prefix + str(max_number + 1))
    os.makedirs(model_dir)
    return model_dir

''' 环境见
simple_adversary_v3,simple_crypto_v3,simple_push_v3,simple_reference_v3,simple_speaker_listener_v3,simple_spread_v3,simple_tag_v3
https://pettingzoo.farama.org/environments/mpe
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="simple_spread_v3") 
    parser.add_argument("--N", type=int, default=5) # 环境中智能体数量 默认None 这里用来对比设置
    parser.add_argument("--max_action", type=float, default=None)
    # 共有参数
    parser.add_argument("--seed", type=int, default=0) # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(600))
    parser.add_argument("--start_steps", type=int, default=500) # 满足此开始更新
    parser.add_argument("--random_steps", type=int, default=500)  #dqn 无此参数 满足此开始自己探索
    parser.add_argument("--learn_steps_interval", type=int, default=1)
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    ## AC参数
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    ## buffer参数   
    parser.add_argument("--buffer_size", type=int, default=1e6) #1e6默认是float,在bufffer中有int强制转换
    parser.add_argument("--batch_size", type=int, default=256)  #保证比start_steps小
    # trick参数
    parser.add_argument("--policy_name", type=str, default='MASAC')
    parser.add_argument("--trick", type=dict, default=None)  

    args = parser.parse_args()

    ## 环境配置
    env,dim_info,max_action,is_continue = get_env(args.env_name, env_agent_n = args.N)
    max_action = max_action if max_action is not None else args.max_action

    ## 随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## 保存model文件夹
    model_dir = make_dir(args.env_name,policy_name = args.policy_name,trick=args.trick)
    writer = SummaryWriter(model_dir)

    ##
    device = torch.device('cpu')

    ## 算法配置
    policy = MASAC(dim_info, is_continue, args.actor_lr, args.critic_lr, args.buffer_size, device, args.trick)

    time_ = time.time()
    ## 训练
    episode_num = 0
    step = 0
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}
    obs,info = env.reset(seed=args.seed)
    while episode_num < args.max_episodes:
        step +=1

        # 获取动作
        if step < args.random_steps: # 区分环境里的action_ 和训练的action 
            action_ = {agent: env.action_space(agent).sample() for agent in env_agents}  # [0,1]
            action = {agent_id: (action_[agent_id] * 2 - 1)* max_action for agent_id in env_agents} # [0,1] -> [-1,1]
        else:
            action = policy.select_action(obs)   # [-1,1]
            action_ = {agent_id: (action[agent_id] + 1) / 2 * max_action for agent_id in env_agents} #[-1,1] -> [0,1] 

        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
        done_bool = {agent_id: done[agent_id] if not truncated[agent_id] else False  for agent_id in env_agents} ### truncated 为超过最大步数
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
        



