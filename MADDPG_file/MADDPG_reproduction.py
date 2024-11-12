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

'''maddpg 
论文：Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments 链接：https://arxiv.org/abs/1706.02275
创新点(特点--与COMA不同点)1.为每个agent学习一个集中式critic 允许agent具有不同奖励 2.考虑了具有明确agent之间通信的环境 3.只使用前馈网络 不使用循环网络 4.学习连续策略
缺点：Q 的输入空间随着agent_N的数量呈线性增长,展望：通过一个模块化的Q来修复，该函数只考虑该agent的某个领域的几个代理
'''

## 第一部分：定义Agent类
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128 ,actor_learn_way = None):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim)

        self.actor_learn_way = actor_learn_way
        if self.actor_learn_way == '1':
            self.l4 = nn.Linear(hidden_2, action_dim*2)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        if self.actor_learn_way == '0':
            x = F.tanh(self.l3(x))
        elif self.actor_learn_way == '1':
            x = self.l4(x)
        return x

# 参考：https://github.com/openai/maddpg/blob/master/maddpg/common/distributions.py#L262
class DiagGaussianPd:
    def __init__(self, flat):
        self.flat = flat
        self.mean, self.logstd = torch.split(flat, flat.size(-1) // 2, dim=-1)
        self.std = torch.exp(self.logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def logp(self, x):
        return -0.5 * ((x - self.mean) / self.std).pow(2).sum(dim=-1) \
               - 0.5 * np.log(2.0 * np.pi) * x.size(-1) \
               - self.logstd.sum(dim=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return (other.logstd - self.logstd + (self.std.pow(2) + (self.mean - other.mean).pow(2)) / (2.0 * other.std.pow(2)) - 0.5).sum(dim=-1)

    def entropy(self):
        return (self.logstd + 0.5 * np.log(2.0 * np.pi * np.e)).sum(dim=-1)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)
        
class Critic(nn.Module):
    def __init__(self, dim_info:dict, hidden_1=128 , hidden_2=128):
        super(Critic, self).__init__()
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())  
        
        self.l1 = nn.Linear(global_obs_act_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)

    def forward(self, s, a): # 传入全局观测和动作
        sa = torch.cat(list(s)+list(a), dim = 1)
        #sa = torch.cat([s,a], dim = 1)
        
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q

class Agent:
    def __init__(self, obs_dim, action_dim, dim_info,actor_lr, critic_lr, device, actor_learn_way):   
        self.actor = Actor(obs_dim, action_dim, actor_learn_way = actor_learn_way)
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
class MADDPG: 
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, buffer_size, device, trick = None):

        self.actor_learn_way = '0' # 论文提出2种方式 # 0 ensemble 集成策略 1 Approximate 近似策略
        
        if self.actor_learn_way == '0':
            self.regular = False  ##False效果好 offical代码中为True  -> https://github.com/openai/maddpg/blob/master/maddpg/trainer/maddpg.py#L56 # 为原ddpg的weight_decay
        elif self.actor_learn_way == '1':
            self.lmbda = 1e-3
        
        self.agents  = {}
        self.buffers = {}
        for agent_id, (obs_dim, action_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, action_dim, dim_info, actor_lr, critic_lr, device = device, actor_learn_way = self.actor_learn_way)
            self.buffers[agent_id] = Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = 'cpu')

        self.device = device
        self.is_continue = is_continue
        self.agent_x = list(self.agents.keys())[0] #sample 用


    def select_action(self, obs):
        actions = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue: # dqn 无此项
                if self.actor_learn_way == '0':
                    action = self.agents[agent_id].actor(obs)
                    actions[agent_id] = action.detach().cpu().numpy().squeeze(0) # 1xaction_dim -> action_dim
                elif self.actor_learn_way == '1':
                    logits = self.agents[agent_id].actor(obs)
                    dist = DiagGaussianPd(logits)
                    action = dist.sample()
                    action =  torch.tanh(action)
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
    
    ## DDPG算法相关
    '''论文中提出两种方法更新actor 最终论文取了方式0作为伪代码 论文中比较使用方式0,1 发现方式0的学习曲线效果与方式1比稍差略微，但KL散度差于方式1 
    0. actor_loss = -critic(x, actor(obs),other_act).mean()    知道其他agent的策略来更新             此时next_target_Q = agent.critic_target(next_obs.values(), next_action.values())
    1. actor_loss = -(log(actor(obs)) + lmbda * H(actor_dist)) 知道其他智能体的obs但不知道策略来更新  此时next_target_Q 与上述一样
    '''
    def learn(self, batch_size ,gamma , tau):
        # 多智能体特有-- 集中式训练critic:计算next_q值时,要用到所有智能体next状态和动作
        for agent_id, agent in self.agents.items():
            ## 更新前准备
            obs, action, reward, next_obs, done = self.sample(batch_size) # 必须放for里，否则报二次传播错，原因是原来的数据在计算图中已经被释放了
            next_action = {}
            if self.actor_learn_way == '0':
                for agent_id_, agent_ in self.agents.items():
                    next_action_i = agent_.actor_target(next_obs[agent_id_])
                    next_action[agent_id_] = next_action_i
            elif self.actor_learn_way == '1':
                for agent_id_, agent_ in self.agents.items():
                    logits = agent_.actor_target(next_obs[agent_id_])
                    dist = DiagGaussianPd(logits)
                    next_action[agent_id_] = torch.tanh(dist.sample())
            next_target_Q = agent.critic_target(next_obs.values(), next_action.values())
            
            # 先更新critic
            target_Q = reward[agent_id] + gamma * next_target_Q * (1 - done[agent_id])
            current_Q = agent.critic(obs.values(), action.values())
            critic_loss = F.mse_loss(current_Q, target_Q.detach())
            agent.update_critic(critic_loss)

            # 再更新actor
            if self.actor_learn_way == '0': # github上开源的代码基本复现为方式0
                new_action = agent.actor(obs[agent_id])
                action[agent_id] = new_action
                actor_loss = -agent.critic(obs.values(), action.values()).mean()
                if self.regular : # 猜测：对于目标action在0附近会有优化
                    actor_loss += (new_action**2).mean() * 1e-3
                agent.update_actor(actor_loss)
            elif self.actor_learn_way == '1':
                logits = agent.actor(obs[agent_id])
                dist = DiagGaussianPd(logits)
                action[agent_id] = torch.tanh(dist.sample())
                actor_loss = - (dist.logp(action[agent_id]).mean() + self.lmbda * dist.entropy().mean())   
                agent.update_actor(actor_loss)        
        
        self.update_target(tau)

    def update_target(self, tau):
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        for agent in self.agents.values():
            soft_update(agent.actor_target, agent.actor, tau)
            soft_update(agent.critic_target, agent.critic, tau)


    def save(self, model_path):
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            os.path.join(model_path, 'MADDPG.pth')
        )
        
    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue, model_dir):
        policy = MADDPG(dim_info, is_continue = is_continue, actor_lr = 0, critic_lr = 0, buffer_size = 0, device = 'cpu')
        torch.load(
            os.path.join(model_dir, 'MADDPG.pth')
        )
        return policy
    
## 第三部分 mian 函数
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
    dim_info = {} # dict{agent_id:[obs_dim,action_dim]}
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

''' 
环境见:simple_adversary_v3,simple_crypto_v3,simple_push_v3,simple_reference_v3,simple_speaker_listener_v3,simple_spread_v3,simple_tag_v3
具体见:https://pettingzoo.farama.org/environments/mpe
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="simple_spread_v3") 
    parser.add_argument("--N", type=int, default=3) # 环境中智能体数量 默认None 这里用来对比设置
    parser.add_argument("--max_action", type=float, default=None)
    # 共有参数
    parser.add_argument("--seed", type=int, default=10) # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(600))
    parser.add_argument("--start_steps", type=int, default=500) # 满足此开始更新
    parser.add_argument("--random_steps", type=int, default=0)  #dqn 无此参数 满足此开始自己探索
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
    # DDPG 独有参数 noise
    parser.add_argument("--gauss_sigma", type=float, default=0.1)
    # trick参数
    parser.add_argument("--policy_name", type=str, default='MADDPG_reproduction')
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
    policy = MADDPG(dim_info, is_continue, args.actor_lr, args.critic_lr, args.buffer_size, device, args.trick)

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
        if step < args.random_steps: # # 区分环境里的action_ 和训练的action
            action_ = {agent: env.action_space(agent).sample() for agent in env_agents}  # [0,1]
            action = {agent_id: (action_[agent_id] * 2 - 1)* max_action for agent_id in env_agents} # [0,1] -> [-1,1]
        else:
            action = policy.select_action(obs) 
            # 加噪音 
            if policy.actor_learn_way == '0':
                action_ = {agent_id: np.clip(action[agent_id] * max_action + np.random.normal(scale = args.gauss_sigma * max_action, size = dim_info[agent_id][1]), -max_action, max_action, dtype = np.float32) for agent_id in env_agents} 
                action_ = {agent_id: (action_[agent_id] + 1) / 2 for agent_id in env_agents} # [-1,1] -> [0,1] 
            elif policy.actor_learn_way == '1':
                action_ = {agent_id: (action[agent_id] + 1) / 2 for agent_id in env_agents} # [-1,1] -> [0,1] 
        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        #print(next_obs, reward,terminated, truncated, infos)
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