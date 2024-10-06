import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
import numpy as np
from Buffer import Buffer # 与DQN.py中的Buffer一样

import gymnasium as gym
import argparse

## 其他
import time
from torch.utils.tensorboard import SummaryWriter

'''与simple区别 加入了原论文的补充
参考1:https://github.com/songrotek/DDPG
参考2:https://github.com/shariqiqbal2810/maddpg-pytorch/
'''
'''ddpg ：Deep DPG
论文：CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING  链接：https://arxiv.org/pdf/1509.02971
创新点（特点，与DQN的区别）：
1.提出一种a model-free, off-policy actor-critic 算法 ，并证实了即使使用原始像素的obs,也能使用continue action spaces 稳健解决各种问题。
2.不需要更改环境的情况下，可以进行稳定的学习。
3.比DQN需要更少的经验步就可以收敛。
可借鉴参数：
hidden：400 -300
actor_lr = 1e-4
critic_lr = 1e-3
buffer_size = 1e6
gamma = 0.99
tau = 0.001
std = 0.2 #高斯标准差
细节补充：
1.weight_decay:对Q使用了正则项l2进行1e-2来权重衰减
2.OUNoise:使用时间相关噪声来进行探索used an Ornstein-Uhlenbeck process theta=0.15 std=0.2
3.ObsNorm:使用批量归一化状态值
4.net_init:对于低维环境 对actor和critic的全连接层的最后一层使用uniform distribution[-3e-3,3e-3],其余层为[-1/sqrt(f),1/sqrt(f)],f为输入的维度
对于pixel case  最后一层[-3e-4,3e-4](这里论文笔误写成[3e-4,3e-4])，其余层[-1/sqrt(f),1/sqrt(f)]
'''

## 第一部分：定义Agent类
'''
补充 net_init
参考:https://github.com/floodsung/DDPG/blob/master/actor_network.py#L96
'''
def other_net_init(layer):
    if isinstance(layer, nn.Linear):
        fan_in = layer.weight.data.size(0)
        limit = 1.0 / (fan_in ** 0.5)
        nn.init.uniform_(layer.weight, -limit, limit)
        nn.init.uniform_(layer.bias, -limit, limit)

def final_net_init(layer,low,high):
    if isinstance(layer, nn.Linear):
        nn.init.uniform_(layer.weight, low, high)
        nn.init.uniform_(layer.bias, low, high)
    
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128,supplement=None,pixel_case=False):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim)

        if supplement['net_init']:
            other_net_init(self.l1)
            other_net_init(self.l2)
            if pixel_case:
                final_net_init(self.l3, low=-3e-4, high=3e-4)
            else:
                final_net_init(self.l3, low=-3e-3, high=3e-3)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.tanh(self.l3(x))

        return x
    
class Critic(nn.Module):
    def __init__(self, dim_info:list, hidden_1=128 , hidden_2=128,supplement=None,pixel_case=False):
        super(Critic, self).__init__()
        obs_act_dim = sum(dim_info)  
        
        self.l1 = nn.Linear(obs_act_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)

        if supplement['net_init']:
            other_net_init(self.l1)
            other_net_init(self.l2)
            if pixel_case:
                final_net_init(self.l3, low=-3e-4, high=3e-4)
            else:
                final_net_init(self.l3, low=-3e-3, high=3e-3)

    def forward(self, o, a): # 传入观测和动作
        oa = torch.cat([o,a], dim = 1)
        
        q = F.relu(self.l1(oa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q
    
class Agent:
    def __init__(self, obs_dim, action_dim, dim_info,actor_lr, critic_lr, device, trick, supplement):   
        self.actor = Actor(obs_dim, action_dim,supplement = supplement).to(device)
        self.critic = Critic( dim_info,supplement = supplement).to(device)


        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        '''
        补充 weight_decay 实现上均不相同 这里选择参考1
        参考1:https://github.com/sfujim/TD3/blob/master/DDPG.py#L55 # adam内部实现
        参考2:https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/algorithms/maddpg.py#L161 # 手动实现 weight_decay = 1e-3
        参考3:https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg_learner.py#L187 # tf内部实现 weight_decay = 0
        '''
        if supplement['weight_decay']:                                                                
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr,weight_decay=1e-3) #原论文值: 1e-2
        else:
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
class DDPG: 
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, buffer_size, device, trick = None,supplement = None):

        obs_dim, action_dim = dim_info
        self.agent = Agent(obs_dim, action_dim, dim_info, actor_lr, critic_lr, device, trick, supplement)
        self.buffer = Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = device) #Buffer中说明了act_dim和action_dim的区别
        self.device = device
        self.is_continue = is_continue

        self.trick = trick
        self.supplement = supplement
        if self.supplement['Batch_ObsNorm']:
            self.batch_size_obs_norm = Normalization_batch_size(shape = obs_dim, device = device)

    def select_action(self, obs):
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device) # 1xobs_dim
        if self.supplement['Batch_ObsNorm']:
            obs = self.batch_size_obs_norm(obs,update=False)
        # 先实现连续域下的ddpg
        if self.is_continue: # dqn 无此项
            action = self.agent.actor(obs).detach().cpu().numpy().squeeze(0) # 1xaction_dim -> action_dim
        else:
            action = self.agent.argmax(dim = 1).detach().cpu().numpy()[0] # []标量
        return action
    
    def evaluate_action(self, obs):
        '''确定性策略ddpg,在main中去掉noise'''
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device) # 1xobs_dim
        # 先实现连续域下的ddpg
        if self.is_continue: # dqn 无此项
            action = self.agent.actor(obs).detach().cpu().numpy().squeeze(0) # 1xaction_dim -> action_dim
        else:
            action = self.agent.argmax(dim = 1).detach().cpu().numpy()[0] # []标量
        return action

    ## buffer相关
    def add(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)
    
    def sample(self, batch_size):
        total_size = len(self.buffer)
        batch_size = min(total_size, batch_size) # 防止batch_size比start_steps大, 一般可去掉
        indices = np.random.choice(total_size, batch_size, replace=False)  #默认True 重复采样 
        obs, actions, rewards, next_obs, dones = self.buffer.sample(indices)
        if self.supplement['Batch_ObsNorm']:
            obs = self.batch_size_obs_norm(obs)
            next_obs = self.batch_size_obs_norm(next_obs,update=False) #只对输入obs进行更新

        return obs, actions, rewards, next_obs, dones
    
    ## 算法相关
    def learn(self,batch_size, gamma, tau):

        obs, actions, rewards, next_obs, dones = self.sample(batch_size) 
        
        '''类似于使用了Double的技巧 + target网络技巧'''
        next_action = self.agent.actor_target(next_obs)
        next_Q_target = self.agent.critic_target(next_obs, next_action) # batch_size x 1
        
        ## 先更新critic
        target_Q = rewards + gamma * next_Q_target * (1 - dones) # batch_size x 1
        current_Q = self.agent.critic(obs ,actions)# batch_size x 1
        critic_loss = F.mse_loss(current_Q, target_Q.detach()) # 标量值
        self.agent.update_critic(critic_loss)

        ## 再更新actor
        new_action = self.agent.actor(obs)
        actor_loss = -self.agent.critic(obs, new_action).mean()
        self.agent.update_actor(actor_loss)

        self.update_target(tau)
    
    def update_target(self, tau):
        '''
        更新目标网络参数: θ_target = τ*θ_local + (1 - τ)*θ_target
        切断自举,缓解高估Q值 source -> target
        '''
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        soft_update(self.agent.critic_target, self.agent.critic, tau)
        soft_update(self.agent.actor_target, self.agent.actor, tau)

    
    ## 保存模型
    def save(self, model_dir):
        torch.save(self.agent.actor.state_dict(), os.path.join(model_dir,"DDPG.pt"))
    
    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue ,model_dir,trick = None,supplement = None):
        policy = DDPG(dim_info,is_continue,0,0,0,device = torch.device("cpu"),trick = trick,supplement = supplement)
        policy.agent.actor.load_state_dict(torch.load(os.path.join(model_dir,"DDPG.pt")))
        return policy

## 第三部分：main函数   
''' 这里不用离散转连续域技巧'''
def get_env(env_name,is_dis_to_con = False):
    env = gym.make(env_name)
    if isinstance(env.observation_space, gym.spaces.Box):
        obs_dim = env.observation_space.shape[0]
    else:
        obs_dim = 1
    if isinstance(env.action_space, gym.spaces.Box): # 是否动作连续环境
        action_dim = env.action_space.shape[0]
        dim_info = [obs_dim,action_dim]
        max_action = env.action_space.high[0]
        is_continuous = True # 指定buffer和算法是否用于连续动作
        if is_dis_to_con :
            if action_dim == 1:
                dim_info = [obs_dim,16]  # 离散动作空间
                max_action = None
                is_continuous = False
            else: # 多重连续动作空间->多重离散动作空间
                dim_info = [obs_dim,2**action_dim]  # 离散动作空间
                max_action = None
                is_continuous = False
    else:
        action_dim = env.action_space.n
        dim_info = [obs_dim,action_dim]
        max_action = None
        is_continuous = False
    
    return env,dim_info, max_action, is_continuous #dqn中均转为离散域.max_action没用到

## make_dir
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
补充 OUNoise 两者仅区别在采样时间的有无 OUNoise 适用于时间离散粒度小的环境/惯性环境/需要动量的环境 优势:就像物价和利率的波动一样，这有利于在一个方向上探索。
这里根据参考1，修改加入参考2、3
参考1： https://github.com/songrotek/DDPG/blob/master/ou_noise.py
参考2： https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py#L49
根据博客:https://zhuanlan.zhihu.com/p/96720878 选择参考2 加入采样时间系数
参考3:  https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/noise.py  # maddpg中加入了一个scale参数
'''
class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.1, dt=1e-2, scale= None):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt # 参考1 相当于默认这里是1
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
        self.scale = scale # 参考3

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) +  np.sqrt(self.dt) * self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        if self.scale is None:
            return self.state 
        else:
            return self.state * self.scale
    
'''
补充：ObsNorm 根据原论文的描述:此技术将小批量中样本的每个维度标准化为具有单位均值和方差,此外,它还维护平均值和方差的运行平均值。这个trick更像是RunningMeanStd 
这里选用参考3
参考1：https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/networks.py#L19 # 直接使用batchnorm 不符合原论文
参考2：https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg_learner.py#L103 # √
参考3：https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/5.PPO-continuous/normalization.py#L4
参考4：https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py#L62 与参考3类似
'''
class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False #是否更新均值和方差，在评估时，update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

'''modify 
根据上述RuningMeanStd的方法和ddpg原论文的描述，将RuningMeanStd的方法（一个state一个state的更新）改进成ddpg原论文描述的（一个batch_size的state一个batch_size的state的更新）
'''
class RunningMeanStd_batch_size:
    # Dynamically calculate mean and std
    def __init__(self, shape,device):  # shape:the dimension of input data
        self.n = 0
        self.mean = torch.zeros(shape).to(device)
        self.S = torch.zeros(shape).to(device)
        self.std = torch.sqrt(self.S).to(device)

    def update(self, x):
        x = x.mean(dim=0,keepdim=True)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean 
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n)


class Normalization_batch_size:
    def __init__(self, shape, device):
        self.running_ms = RunningMeanStd_batch_size(shape,device)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False #是否更新均值和方差，在评估时，update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x
''' 
环境见：CartPole-v1,Pendulum-v1,MountainCar-v0,MountainCarContinuous-v0;LunarLander-v2,BipedalWalker-v3;FrozenLake-v1
具体见：https://github.com/openai/gym/blob/master/gym/envs/__init__.py 
此算法 只写了连续域 BipedalWalker-v3  Pendulum-v1
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="MountainCarContinuous-v0") 
    parser.add_argument("--max_action", type=float, default=None)
    # 共有参数
    parser.add_argument("--seed", type=int, default=0) # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(500))
    parser.add_argument("--save_freq", type=int, default=int(500//4)) # 与max_episodes有关
    parser.add_argument("--start_steps", type=int, default=500) #ppo无此参数
    parser.add_argument("--random_steps", type=int, default=0)  ##可选择是否使用 ddpg原论文中没使用
    parser.add_argument("--learn_steps_interval", type=int, default=1)  
    parser.add_argument("--is_dis_to_con", type=bool, default=False) # dqn 默认为True
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    ## A-C参数
    parser.add_argument("--actor_lr", type=float, default=1e-3)   
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    ## buffer参数   
    parser.add_argument("--buffer_size", type=int, default=int(1e6)) #1e6默认是float,在bufffer中有int强制转换
    parser.add_argument("--batch_size", type=int, default=64)  #保证比start_steps小 #256 (64 for MountainCarContinuous-v0)
    # DDPG 独有参数 noise
    ## gauss noise
    parser.add_argument("--gauss_sigma", type=float, default=1) # 高斯标准差 #0.1 (1 for MountainCarContinuous-v0 )
    parser.add_argument("--gauss_scale", type=float, default=1)
    parser.add_argument("--gauss_init_scale", type=float, default=None) # 若不设置衰减，则设置成None
    parser.add_argument("--gauss_final_scale", type=float, default=0.0)
    ## OU noise
    parser.add_argument("--ou_sigma", type=float, default=1) # 
    parser.add_argument("--ou_dt", type=float, default=1)
    parser.add_argument("--init_scale", type=float, default=1) # 若不设置衰减，则设置成None # maddpg值:ou_sigma 0.2 init_scale:0.3
    parser.add_argument("--final_scale", type=float, default=0.0) 
    # trick参数
    parser.add_argument("--policy_name", type=str, default='DDPG') #
    parser.add_argument("--supplement", type=dict, default={'weight_decay':True,'OUNoise':True,'ObsNorm':False,'net_init':True,'Batch_ObsNorm':True}) # ObsNorm效果差于Batch_ObsNorm,选择Batch_ObsNorm
    parser.add_argument("--trick", type=dict, default=None) 
    # device参数
    parser.add_argument("--device", type=str, default='cpu') # cpu/cuda
    
    args = parser.parse_args()
    # 如果 policy_name 是 'DDPG_simple'，或者 supplement 符合simple条件，就设置两者
    if args.policy_name == 'DDPG_simple' or args.supplement == {'weight_decay':False,'OUNoise':False,'ObsNorm':False,'net_init':False,'Batch_ObsNorm':False}:
        args.policy_name = 'DDPG_simple'
        args.supplement = {'weight_decay':False,'OUNoise':False,'ObsNorm':False,'net_init':False,'Batch_ObsNorm':False}
    print(args)
    print('Algorithm:',args.policy_name)
    print("Supplement:",args.supplement) if args.supplement else None
    
    ## 环境配置
    env,dim_info,max_action,is_continue = get_env(args.env_name,args.is_dis_to_con)
    max_action = max_action if max_action is not None else args.max_action
    obs_dim ,action_dim = dim_info
    print(f'Env:{args.env_name}  obs_dim:{dim_info[0]}  action_dim:{dim_info[1]}  max_action:{max_action}  max_episodes:{args.max_episodes}')

    ## 随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ### cuda
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random Seed:',args.seed)

    ## 保存文件夹
    model_dir = make_dir(args.env_name,policy_name = args.policy_name ,trick=args.trick)
    print(f'model_dir: {model_dir}')
    writer = SummaryWriter(model_dir)

    ## device参数
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    
    ## 算法配置
    policy = DDPG(dim_info, is_continue, args.actor_lr, args.critic_lr, args.buffer_size, device, args.trick,args.supplement)

    env_spec = gym.spec(args.env_name)
    print('reward_threshold:',env_spec.reward_threshold if env_spec.reward_threshold else 'No Threshold = Higher is better')
    time_ = time.time()
    ## 训练
    episode_num = 0
    step = 0
    episode_reward = 0
    train_return = []
    obs,info = env.reset(seed=args.seed)
    env.action_space.seed(seed=args.seed) if args.random_steps > 0 else None # 针对action复现:env.action_space.sample()
    if args.gauss_init_scale is not None:
        args.gauss_scale = args.gauss_init_scale

    if args.supplement['ObsNorm']:
        obs_norm = Normalization(shape = obs_dim)
        obs = obs_norm(obs)

    if args.supplement['OUNoise']:
        ou_noise = OUNoise(action_dim, sigma = args.ou_sigma, dt=args.ou_dt, scale = args.init_scale) #

    while episode_num < args.max_episodes:
        step +=1

        # 获取动作 区分动作action_为环境中的动作 action为要训练的动作
        if step < args.random_steps:
            action_ = env.action_space.sample()  # [-max_action , max_action]
            action = action_ / max_action # -> [-1,1]
        else:
            action = policy.select_action(obs)  # (-1,1)
            if args.supplement['OUNoise']:
                action_ = np.clip(action * max_action + ou_noise.noise()* max_action, -max_action, max_action)
            else: # 高斯噪声
                action_ = np.clip(action * max_action + args.gauss_scale * np.random.normal(scale = args.gauss_sigma * max_action, size = action_dim), -max_action, max_action)

        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        if args.supplement['ObsNorm']:
            next_obs = obs_norm(next_obs)
        done = terminated or truncated
        done_bool = terminated     ### truncated 为超过最大步数
        policy.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        obs = next_obs
        
        # episode 结束
        if done:
            if args.supplement['OUNoise']:
                ou_noise.reset()
                ## OUNoise scale若有 scale衰减 参考:https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/main.py#L71
                if args.init_scale is not None:
                    explr_pct_remaining = max(0, args.max_episodes - (episode_num + 1)) / args.max_episodes # 剩余探索百分比
                    ou_noise.scale = args.final_scale + (args.init_scale - args.final_scale) * explr_pct_remaining
            ## gauss_noise 衰减
            if args.gauss_init_scale is not None:
                explr_pct_remaining = max(0, args.max_episodes - (episode_num + 1)) / args.max_episodes
                args.gauss_scale = args.gauss_final_scale + (args.gauss_init_scale - args.gauss_final_scale) * explr_pct_remaining
            ## 显示
            if  (episode_num + 1) % 100 == 0:
                print("episode: {}, reward: {}".format(episode_num + 1, episode_reward))
            ## 保存
            if (episode_num + 1) % args.save_freq == 0:
                policy.save(model_dir)
            writer.add_scalar('reward', episode_reward, episode_num + 1)
            train_return.append(episode_reward)

            episode_num += 1
            obs,info = env.reset(seed=args.seed)
            if args.supplement['ObsNorm']:
                obs = obs_norm(obs)
            episode_reward = 0
        
        # 满足step,更新网络
        if step > args.start_steps and step % args.learn_steps_interval == 0:
            policy.learn(args.batch_size, args.gamma, args.tau)
        
        # 保存模型
        if episode_num % args.save_freq == 0:
            policy.save(model_dir)

    
    print('total_time:',time.time()-time_)
    policy.save(model_dir)
    ## 保存数据
    np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),np.array(train_return))
    if args.supplement['ObsNorm']:
        np.save(os.path.join(model_dir,f"{args.policy_name}_running_mean_std.npy"),np.array([obs_norm.running_ms.mean,obs_norm.running_ms.std]))
    if args.supplement['Batch_ObsNorm']:
        np.save(os.path.join(model_dir,f"{args.policy_name}_running_mean_std_batch_size.npy"),np.array([policy.batch_size_obs_norm.running_ms.mean.detach().cpu().numpy(),policy.batch_size_obs_norm.running_ms.std.detach().cpu().numpy()]))