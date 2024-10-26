import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal,Categorical

from copy import deepcopy
import numpy as np
from Buffer import Buffer

import gymnasium as gym
import argparse

## 其他
import re
import time
from torch.utils.tensorboard import SummaryWriter

'''
一个深度强化学习算法分三个部分实现：
1.Agent类:包括actor、critic、target_actor、target_critic、actor_optimizer、critic_optimizer、
2.DQN算法类:包括select_action,learn、test、save、load等方法,为具体的算法细节实现
3.main函数:实例化DQN类,主要参数的设置,训练、测试、保存模型等
'''
'''SAC 论文：https://arxiv.org/pdf/1812.05905 代码：https://github.com/rail-berkeley/softlearning/blob/master/softlearning/algorithms/sac.py
提出的sac有三个关键要素：1.独立的AC框架 2.off-policy 3.最大化熵以鼓励稳定性和探索
###此代码仅为continue的实现###
注：论文及代码做了continue环境下的实现,并未实现discrete环境下的实现。
创新点(或建议)：
1.基于梯度的自动温度调整方法 
2.使用类似于TD3一样的双截断Q网络来加速收敛
可参考参数：
hidden_dim 256-256
actor_lr = 3e-4
critic_lr = 3e-4
buffer_size = 1e6
batch_size = 256
gamma = 0.99
tau = 5e-3 
## sac独有
alpha_lr = 3e-4
entropy target = - dim(A)
'''


## 第一部分：定义Agent类
'''actor部分 ppo与sac的相同和区别
相同：
1.测试时均是使用 action = tanh(mean)
2.高斯分布时 均是输出mean和std
不同：
1.sac需要对策略高斯分布采样(即使用重参数化技巧),而ppo不需要,因为ppo是更新替代策略(surrogate policy)
2.采样时sac action = tanh(rsample(mean,std))  ppo action = sample(mean,std) 
3.计算log_pi时 1.. sac的log_pi 是对tanh的对数概率，而ppo的log_pi是对动作的对数概率
        (重要) 2.. sac的log_pi 是直接对s_t得出的a_t计算的，而ppo的log_pi是对buffer中存储的s和a计算的(具体而言s->dist dist(a).log->log_pi)        
'''
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.mean_layer = nn.Linear(hidden_2, action_dim)        # 此方法称为对角高斯函数 的主流方法1.对于每个action维度都有独立的方差 第二种方法 2.self.log_std_layer = nn.Linear(hidden_2, action_dim) log_std 是环境状态的函数
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # 方法参考 1.https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py#L134C29-L134C41
                                                                      # 2.    https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/5.PPO-continuous/ppo_continuous.py#L56
                                                                         # 3.    https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py#L85
        #self.log_std_layer = nn.Linear(hidden_2, action_dim) # 法2
    def forward(self, obs, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std.expand_as(mean)  # 使得log_std与mean维度相同 输出log_std以确保std=exp(log_std)>0
        #log_std = self.log_std_layer(x)  # 我们输出log_std以确保std=exp(log_std)>0 # 法2
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
critic部分 ppo与sac区别
区别:sac中critic输出Q1,Q2,而ppo中只输出V
'''    
class Critic(nn.Module):
    def __init__(self, dim_info, hidden_1=128, hidden_2=128):
        super(Critic, self).__init__()
        obs_act_dim = sum(dim_info)
        # Q1
        self.l1 = nn.Linear(obs_act_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)
        # Q2
        self.l4 = nn.Linear(obs_act_dim,hidden_1)
        self.l5 = nn.Linear(hidden_1,hidden_2)
        self.l6 = nn.Linear(hidden_2,1)

    def forward(self, o , a):
        oa = torch.cat([o,a], dim = 1)
        
        q1 = F.relu(self.l1(oa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(oa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2
    
class Agent:
    def __init__(self, obs_dim, action_dim, dim_info ,actor_lr, critic_lr, is_continue, device):
        
        self.actor = Actor(obs_dim, action_dim, ).to(device)
        self.critic = Critic( dim_info ).to(device)

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
    def __init__(self, action_dim, alpha_lr= 0.0001, alpha = 0.2,requires_grad = False,is_continue=True):

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

class SAC:
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, buffer_size, device, trick = None):

        obs_dim, action_dim = dim_info
        self.agent = Agent(obs_dim, action_dim,dim_info,actor_lr, critic_lr, is_continue, device)
        self.buffer = Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = device,) #Buffer中说明了act_dim和action_dim的区别
        self.device = device
        self.is_continue = is_continue

        self.trick = trick
        if self.trick['Batch_ObsNorm']:
            self.batch_size_obs_norm = Normalization_batch_size(shape = obs_dim, device = device)

        self.adaptive_alpha = True
        print('adaptive_alpha:',self.adaptive_alpha)
        self.trick = trick
        if self.adaptive_alpha:
            self.alphas = Alpha(action_dim,alpha = 0.01, requires_grad=True, is_continue= is_continue) # Alpha(action_dim).alpha 才是值
        else:
            self.alphas = Alpha(action_dim,alpha = 0.01,requires_grad=False, is_continue= is_continue) 

    def select_action(self, obs):
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device) # 1xobs_dim
        if self.trick['Batch_ObsNorm']:
            obs = self.batch_size_obs_norm(obs,update=False)
        action , _ = self.agent.actor(obs)
        action = action.detach().cpu().numpy().squeeze(0) # 1xaction_dim -> action_dim
        return action 
    
    def evaluate_action(self, obs):
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
        mean, _ = self.agent.actor(obs,deterministic = True, with_logprob = False)
        action = mean.detach().cpu().numpy().squeeze(0)
        return action
    
    ## buffer相关
    def add(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)
    
    def sample(self, batch_size):
        total_size = len(self.buffer)
        batch_size = min(total_size, batch_size) # 防止batch_size比start_steps大, 一般可去掉
        indices = np.random.choice(total_size, batch_size, replace=False)  #默认True 重复采样 
        obs, actions, rewards, next_obs, dones = self.buffer.sample(indices)
        if self.trick['Batch_ObsNorm']:
            obs = self.batch_size_obs_norm(obs)
            next_obs = self.batch_size_obs_norm(next_obs,update=False) #只对输入obs进行更新

        return obs, actions, rewards, next_obs, dones

    ## SAC算法相关
    def learn(self, batch_size ,gamma , tau):

        obs, actions, rewards, next_obs, dones = self.sample(batch_size) 
        
        '''类似于使用了Double的技巧 + target网络技巧'''
        next_action , next_log_pi = self.agent.actor_target(next_obs)
        next_Q1_target, next_Q2_target = self.agent.critic_target(next_obs, next_action) # batch_size x 1
        next_Q_target = torch.min(next_Q1_target,next_Q2_target)
        
        '''SAC 特有'''
        entropy_next = - next_log_pi
        ## 先更新critic
        ''' 公式: LQ_w = E_{s,a,r,s',d}[(Q_w(s,a) - (r + gamma * (1 - d) * (Q_w'(s',a') - alpha * log_pi_a(s',a')))^2] '''
        target_Q = rewards + gamma  * (1 - dones) * (next_Q_target + self.alphas.alpha.detach() * entropy_next) # batch_size x 1
        current_Q1,current_Q2 = self.agent.critic(obs ,actions)# batch_size x 1
        critic_loss = F.mse_loss(current_Q1, target_Q.detach()) + F.mse_loss(current_Q2, target_Q.detach()) # 标量值
        self.agent.update_critic(critic_loss)

        ## 再更新actor
        '''公式: Lpi_θ = E_{s,a ~ D}[-Q_w(s,a) + alpha * log_pi_a(s,a)]  
        理解为 最大化函数V,V = Q + alpha * H
        '''
        new_action , new_log_pi= self.agent.actor(obs)
        entropy = -new_log_pi
        Q1_pi , Q2_pi = self.agent.critic(obs, new_action)
        ''' 注：在更新target_Q时sac原论文代码中使用的是min,这点和TD3一致;
        但在更新pi时TD3选取的是Q1,sac原论文代码取mean,网上一般做法是取min。
        '''
        Q_pi = torch.mean(torch.stack((Q1_pi, Q2_pi)), dim=0) # (Q1+Q2)/2# min
        actor_loss = (- Q_pi - self.alphas.alpha.detach() * entropy).mean()  #这里alpha一定要加detach(),因为在更新critic时,计算图被丢掉了
        self.agent.update_actor(actor_loss)

        self.update_target(tau)

        ## 更新alpha
        if self.adaptive_alpha:
            '''公式: Lα = E_{s,a ~ D} [-α * log_pi_a(s,a) - α * H] = E_{s,a ~ D} [α * (-log_pi_a(s,a) - H)]'''
            alpha_loss = (self.alphas.alpha * (entropy - self.alphas.target_entropy).detach()).mean()
            self.alphas.update_alpha(alpha_loss)

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
        torch.save(self.agent.actor.state_dict(), os.path.join(model_dir,"SAC.pt"))
    
    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue ,model_dir,trick=None):
        policy = SAC(dim_info,is_continue,0,0,0,device = torch.device("cpu"), trick = trick)
        policy.agent.actor.load_state_dict(torch.load(os.path.join(model_dir,"SAC.pt")))
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
                is_continuous = False
            else: # 多重连续动作空间->多重离散动作空间
                dim_info = [obs_dim,2**action_dim]  # 离散动作空间
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
    pattern = re.compile(f'^{prefix}\d+') # ^ 表示开头，\d 表示数字，+表示至少一个
    existing_dirs = [d for d in os.listdir(env_dir) if pattern.match(d)]
    max_number = 0 if not existing_dirs else max([int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()])
    model_dir = os.path.join(env_dir, prefix + str(max_number + 1))
    os.makedirs(model_dir)
    return model_dir

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
环境见：
离散: CartPole-v1,MountainCar-v0,;LunarLander-v2,;FrozenLake-v1 
连续：Pendulum-v1,MountainCarContinuous-v0,BipedalWalker-v3
reward_threshold：https://github.com/openai/gym/blob/master/gym/envs/__init__.py 
介绍：https://gymnasium.farama.org/
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="MountainCarContinuous-v0") 
    # 共有参数
    parser.add_argument("--seed", type=int, default=0) # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(500))
    parser.add_argument("--save_freq", type=int, default=int(500//4))
    parser.add_argument("--start_steps", type=int, default=500) #ppo无此参数
    parser.add_argument("--random_steps", type=int, default=500)  #dqn 无此参数
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
    ## OU noise
    parser.add_argument("--ou_sigma", type=float, default=1) # 
    parser.add_argument("--ou_dt", type=float, default=1)
    parser.add_argument("--init_scale", type=float, default=1) # 若不设置衰减，则设置成None # maddpg值:ou_sigma 0.2 init_scale:0.3
    parser.add_argument("--final_scale", type=float, default=0.0)
    ## gauss noise
    parser.add_argument("--gauss_sigma", type=float, default=1) # 高斯标准差 #0.1 (1 for MountainCarContinuous-v0 )
    parser.add_argument("--gauss_scale", type=float, default=1)
    parser.add_argument("--gauss_init_scale", type=float, default=1) # 若不设置衰减，则设置成None
    parser.add_argument("--gauss_final_scale", type=float, default=0.0)
    # trick参数
    parser.add_argument("--policy_name", type=str, default='SAC')
    parser.add_argument("--trick", type=dict, default={'ObsNorm':False,'Batch_ObsNorm':False, # 两则择一 可不用
                                                       'OUNoise':True,'GaussNoise':False,  # 两则择一
                                                       }) # 两者择一

    # device参数
    parser.add_argument("--device", type=str, default='cpu') # cpu/cuda
    args = parser.parse_args()
    print(args)
    print('Algorithm:',args.policy_name)
    
    ## 环境配置
    env,dim_info,max_action,is_continue = get_env(args.env_name,args.is_dis_to_con)
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
    policy = SAC(dim_info, is_continue, actor_lr = args.actor_lr, critic_lr = args.critic_lr, buffer_size = args.buffer_size, device = device,trick=args.trick)

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
    
    if args.trick['ObsNorm']:
        obs_norm = Normalization(shape = obs_dim)
        obs = obs_norm(obs)
    
    if args.trick['OUNoise']:
        ou_noise = OUNoise(action_dim, sigma = args.ou_sigma, dt=args.ou_dt, scale = args.init_scale) #
    if args.trick['GaussNoise']:
        if args.gauss_init_scale is not None:
            args.gauss_scale = args.gauss_init_scale

    while episode_num < args.max_episodes:
        step +=1

        # 获取动作
        if step < args.random_steps:
            action_ = env.action_space.sample()  # [-max_action , max_action]
            action = action_ / max_action # -> [-1,1]
        else:
            action  = policy.select_action(obs)   # action (-1,1)
            if args.trick['OUNoise']:
                action_ = np.clip(action * max_action + ou_noise.noise()* max_action, -max_action, max_action)
            elif args.trick['GaussNoise']:
                action_ = np.clip(action * max_action + args.gauss_scale * np.random.normal(scale = args.gauss_sigma * max_action, size = action_dim), -max_action, max_action)
            else:
                action_ = np.clip(action * max_action , -max_action, max_action)
        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        if args.trick['ObsNorm']:
            next_obs = obs_norm(next_obs)
        done = terminated or truncated
        done_bool = terminated     ### truncated 为超过最大步数
        policy.add(obs, action, reward, next_obs, done_bool,)
        episode_reward += reward
        obs = next_obs
        
        # episode 结束
        if done:
            if args.trick['OUNoise']:
                ou_noise.reset()
                ## OUNoise scale若有 scale衰减 参考:https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/main.py#L71
                if args.init_scale is not None:
                    explr_pct_remaining = max(0, args.max_episodes - (episode_num + 1)) / args.max_episodes # 剩余探索百分比
                    ou_noise.scale = args.final_scale + (args.init_scale - args.final_scale) * explr_pct_remaining
            elif args.trick['GaussNoise']:
                ## gauss_noise 衰减
                if args.gauss_init_scale is not None:
                    explr_pct_remaining = max(0, args.max_episodes - (episode_num + 1)) / args.max_episodes
                    args.gauss_scale = args.gauss_final_scale + (args.gauss_init_scale - args.gauss_final_scale) * explr_pct_remaining

            ## 显示
            if  (episode_num + 1) % 100 == 0:
                print("episode: {}, reward: {}".format(episode_num + 1, episode_reward))
            writer.add_scalar('reward', episode_reward, episode_num + 1)
            train_return.append(episode_reward)

            episode_num += 1
            obs,info = env.reset(seed=args.seed)
            if args.trick['ObsNorm']:
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
    if args.trick['ObsNorm']:
        np.save(os.path.join(model_dir,f"{args.policy_name}_running_mean_std.npy"),np.array([obs_norm.running_ms.mean,obs_norm.running_ms.std]))
    if args.trick['Batch_ObsNorm']:
        np.save(os.path.join(model_dir,f"{args.policy_name}_running_mean_std_batch_size.npy"),np.array([policy.batch_size_obs_norm.running_ms.mean.detach().cpu().numpy(),policy.batch_size_obs_norm.running_ms.std.detach().cpu().numpy()]))