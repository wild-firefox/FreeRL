import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal,Categorical

from copy import deepcopy
import numpy as np
from Buffer import Buffer_for_PPO

import gymnasium as gym
import argparse

## tricks
from normalization import Normalization,Normalization_batch_size,RewardScaling
from torch.distributions import Beta

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
'''PPO论文链接:https://arxiv.org/pdf/1707.06347'''

'''PPO论文中提到，
1.如果使用AC网路共享参数，则
loss = - surrogate_objective + value_coefficient * value_loss - entropy_coefficient * entropy_loss
(其中 value_coefficient = 1, entropy_coefficient = 0.01),需要存储
2.如果不使用AC共享参数，则
actor_loss = - surrogate_objective - entropy_coefficient * entropy_loss
critic_loss =  (V - V_target) ** 2
这里使用第2种
'''
########## tricks ###############
'''tricks (本身算法已自带entropy_coef,由于grad_clip可以减少梯度消失的发生,算法默认加上)
1.adv_norm :提升PG算法性能
2.ObsNorm or Batch_ObsNorm ：统一到一个规模,利于训练
3.reward_norm or reward_scaling :调整reward的尺度，避免因过大或过小的reward对价值函数的训练产生负面影响 后者 --《PPO-Implementation matters in deep policy gradients A case study on PPO and TRPO》
4.lr_decay :一定程度增强训练后期的平稳性
5.orthogonal_init ：防止在训练开始时出现梯度消失、梯度爆炸等问题
6.adam_eps : 提高数值稳定性
7.tanh :对于性能有一定提升 --《PPO-Implementation matters in deep policy gradients A case study on PPO and TRPO》
附加:
Actor_Beta分布

参考:
1.https://zhuanlan.zhihu.com/p/512327050?utm_psn=1766822721810653184 # 已包含此10个
2.https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
'''

## 第一部分：定义Agent类
'''actor部分 与sac的相同和区别
相同：
1.测试时均是使用 action = tanh(mean)
2.高斯分布时 均是输出mean和std
不同：
1.sac需要对策略高斯分布采样(即使用重参数化技巧),而ppo不需要,因为ppo是更新替代策略(surrogate policy)
2.采样时sac action = tanh(rsample(mean,std))  ppo action = sample(mean,std) 
3.计算log_pi时 1.. sac的log_pi 是对tanh的对数概率，而ppo的log_pi是对动作的对数概率
        (重要) 2.. sac的log_pi 是直接对s_t得出的a_t计算的，而ppo的log_pi是对buffer中存储的s和a计算的(具体而言s->dist dist(a).log->log_pi)        
'''
def orthogonal_init(layer, gain=1.0):
    '''
    注：一般在初始化actor网络的输出层时，会把gain设置成0.01，actor网络的其他层和critic网络都使用Pytorch中正交初始化默认的gain=1.0。--tricks 参考1
    把gain设置成0.01：这个方法类似于ddpg中的net_init的方法，意思是保持初始输出actor的动作接近于0。
    '''
    nn.init.orthogonal_(layer.weight, gain=gain) #gain意思是权重矩阵的缩放因子 默认为1
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128,trick = None):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.mean_layer = nn.Linear(hidden_2, action_dim)        # 此方法称为对角高斯函数 的主流方法1.对于每个action维度都有独立的方差 第二种方法 2.self.log_std_layer = nn.Linear(hidden_2, action_dim) log_std 是环境状态的函数
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # 方法参考 1.https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py#L134C29-L134C41
                                                                      # 2.    https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/5.PPO-continuous/ppo_continuous.py#L56
                                                                      # 3.    https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py#L85
        self.trick = trick
        if self.trick['orthogonal_init']:
            orthogonal_init(self.l1)
            orthogonal_init(self.l2)
            orthogonal_init(self.mean_layer,gain=0.01)
        
        self.act_func = nn.Tanh() if self.trick['tanh'] else nn.ReLU()
    
    def forward(self, obs ):
        x = self.act_func(self.l1(obs))
        x = self.act_func(self.l2(x))
        mean = torch.tanh(self.mean_layer(x))  # 使得mean在-1,1之间

        log_std = self.log_std.expand_as(mean)  # 使得log_std与mean维度相同 输出log_std以确保std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2) # exp(-20) - exp(2) 等于 2e-9 - 7.4，确保std在合理范围内
        std = torch.exp(log_std)

        return mean, std
    
class Actor_discrete(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128):
        super(Actor_discrete, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim)

    def forward(self, obs ):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        a_prob = torch.softmax(self.l3(x), dim=1)
        return a_prob

class Actor_Beta(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128,trick = None):
        super(Actor_Beta, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.alpha_layer = nn.Linear(hidden_2, action_dim) #输出
        self.beta_layer = nn.Linear(hidden_2, action_dim) 

        self.trick = trick
        if self.trick['orthogonal_init']:
            orthogonal_init(self.l1)
            orthogonal_init(self.l2)
            orthogonal_init(self.alpha_layer,gain=0.01)
            orthogonal_init(self.beta_layer,gain=0.01)
        
        self.act_func = nn.Tanh() if self.trick['tanh'] else nn.ReLU()
    
    def forward(self, x ):
        x = self.act_func(self.l1(x))
        x = self.act_func(self.l2(x))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(x)) + 1.0
        beta = F.softplus(self.beta_layer(x)) + 1.0
        return alpha, beta 
    
    def mean(self, x):
        '''公式 beta分布的mean = alpha/(alpha+beta)
        mean输出范围:0-1
        '''
        alpha, beta = self.forward(x)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean 
    
'''
critic部分 与sac区别
区别:sac中critic输出Q1,Q2,而ppo中只输出V
'''    
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_1=128, hidden_2=128,trick = None):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)

        self.trick = trick
        if self.trick['orthogonal_init']:
            orthogonal_init(self.l1)
            orthogonal_init(self.l2)
            orthogonal_init(self.l3)
        
        self.act_func = nn.Tanh() if self.trick['tanh'] else nn.ReLU()

    def forward(self, x):
        x = self.act_func(self.l1(x))
        x = self.act_func(self.l2(x))
        value = self.l3(x)

        return value
    
class Agent:
    def __init__(self, obs_dim, action_dim, actor_lr, critic_lr, is_continue, device, trick, actor_dist):
        
        if is_continue:
            if actor_dist['Beta']:
                self.actor = Actor_Beta(obs_dim, action_dim, trick=trick).to(device)
            else:
                self.actor = Actor(obs_dim, action_dim, trick=trick).to(device)
        else:
            self.actor = Actor_discrete(obs_dim, action_dim, ).to(device)
        self.critic = Critic( obs_dim,trick=trick ).to(device)
        
        if trick['adam_eps']:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

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
class PPO:
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, horizon, device, trick = None,beta = False):

        self.actor_dist = {'Beta':True if beta else False} # 默认actor拟合Gaussian函数
        print('actor_dist:Beta') if self.actor_dist['Beta'] else print('actor_dist:Gaussian')
        
        obs_dim, action_dim = dim_info
        self.agent = Agent(obs_dim, action_dim,  actor_lr, critic_lr, is_continue, device, trick, actor_dist=self.actor_dist)
        self.buffer = Buffer_for_PPO(horizon, obs_dim, act_dim = action_dim if is_continue else 1, device = device) #Buffer中说明了act_dim和action_dim的区别
        self.device = device
        self.is_continue = is_continue
        print('actor_type:continue') if self.is_continue else print('actor_type:discrete')

        self.horizon = int(horizon)

        self.trick = trick
        if self.trick['Batch_ObsNorm']:
            self.batch_size_obs_norm = Normalization_batch_size(shape = obs_dim, device = device)
        
        if self.trick['lr_decay']:
            self.actor_lr = actor_lr
            self.critic_lr = critic_lr

    def select_action(self, obs):
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device) # 1xobs_dim
        if self.trick['Batch_ObsNorm']:
            obs = self.batch_size_obs_norm(obs,update=False)

        if self.is_continue: # dqn 无此项
            if self.actor_dist['Beta']:
                alpha, beta = self.agent.actor(obs)
                dist = Beta(alpha, beta)
            else:
                mean, std = self.agent.actor(obs)
                dist = Normal(mean, std)
            action = dist.sample()
            action_log_pi = dist.log_prob(action) # 1xaction_dim
        else:
            dist = Categorical(probs=self.agent.actor(obs))
            action = dist.sample()
            action_log_pi = dist.log_prob(action)
        # to 真实值
        action = action.detach().cpu().numpy().squeeze(0) # 1xaction_dim ->action_dim
        action_log_pi = action_log_pi.detach().cpu().numpy().squeeze(0) # 1xaction_dim ->action_dim
        return action , action_log_pi
    
    def evaluate_action(self, obs):
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
        if self.is_continue:
            if self.actor_dist['Beta']:
                mean = self.agent.actor.mean(obs)
                mean = 2 * (mean - 0.5)  # [0,1] ->[-1,1]
            else:
                mean, _ = self.agent.actor(obs)
            action = mean.detach().cpu().numpy().squeeze(0)
        else:
            a_prob = self.agent.actor(obs).detach().cpu().numpy().squeeze(0) 
            action = np.argmax(a_prob)
        return action
    
    ## buffer相关
    '''PPO论文中提到
    计算优势函数 有两种方法1.generalized advantage estimation 2.finite-horizon estimators
    第2种实现方法在许多代码上的实现方法不一,有buffer中存入return和value值的方法,也有在buffer里不存，而在在更新时计算的方法。
    这里我们选择第1种,在buffer中不会存在上述争议。
    通常ppo的buffer中存储的是obs, action, reward, next_obs, done, log_pi ;
    比较1.不存储log_pi,而是在更新时计算出log_pi_old, 2.存储log_pi，将此作为log_pi_old 发现2更好 采用2
    '''
    def add(self, obs, action, reward, next_obs, done, action_log_pi , adv_dones):
        self.buffer.add(obs, action, reward, next_obs, done, action_log_pi , adv_dones)
    
    ## ppo 无 buffer_sample

    ## PPO算法相关
    '''
    论文：GENERALIZED ADVANTAGE ESTIMATION:https://arxiv.org/pdf/1506.02438 提到
    先更新critic会造成额外的偏差，所以PPO这里 先更新actor，再更新critic ,且PPO主要是策略更新的方法
    lmbda = 0 时 GAE 为 one-step TD ; lmbda = 1时，GAE 为 MC
    '''
    def learn(self, minibatch_size, gamma, lmbda ,clip_param, K_epochs, entropy_coefficient):
        '''
        done : dead or win
        adv_done : dead or win or reach max step
        gae 公式：A_t = delta_t + gamma * lmbda * A_t+1 * (1 - adv_done) 
        '''
        obs, action, reward, next_obs, done , action_log_pi , adv_dones = self.buffer.all()
        if self.trick['Batch_ObsNorm']:
            obs = self.batch_size_obs_norm(obs)
            next_obs = self.batch_size_obs_norm(next_obs,update=False) #只对输入obs进行更新
        # 计算GAE
        with torch.no_grad():  # adv and v_target have no gradient
            adv = np.zeros(self.horizon,dtype=torch.float32)
            gae = 0
            vs = self.agent.critic(obs)
            vs_ = self.agent.critic(next_obs)
            td_delta = reward + gamma * (1.0 - done) * vs_ - vs
            td_delta = td_delta.reshape(-1).cpu().detach().numpy()
            adv_dones = adv_dones.reshape(-1).cpu().detach().numpy()
            for i in reversed(range(self.horizon)):
                gae = td_delta[i] + gamma * lmbda * gae * (1.0 - adv_dones[i])
                adv[i] = gae
            adv = torch.as_tensor(adv,dtype=torch.float32).reshape(-1, 1).to(self.device) ## cuda
            v_target = adv + vs  
            if self.trick['adv_norm']:  
                adv = ((adv - adv.mean()) / (adv.std() + 1e-8)) 

        # Optimize policy for K epochs:
        for _ in range(K_epochs): 
            # 随机打乱样本 并 生成小批量
            shuffled_indices = np.random.permutation(self.horizon)
            indexes = [shuffled_indices[i:i + minibatch_size] for i in range(0, self.horizon, minibatch_size)]
            for index in indexes:
                # 先更新actor
                if self.is_continue:
                    if self.actor_dist['Beta']:
                        alpha, beta = self.agent.actor(obs[index])
                        dist_now = Beta(alpha, beta)
                    else:
                        mean, std = self.agent.actor(obs[index])
                        dist_now = Normal(mean, std)
                    dist_entropy = dist_now.entropy().sum(dim = 1, keepdim=True)  # mini_batch_size x action_dim -> mini_batch_size x 1
                    action_log_pi_now = dist_now.log_prob(action[index]) # mini_batch_size x 1
                else:
                    dist_now = Categorical(probs=self.agent.actor(obs[index]))
                    dist_entropy = dist_now.entropy().reshape(-1,1) # mini_batch_size  -> mini_batch_size x 1
                    action_log_pi_now = dist_now.log_prob(action[index].reshape(-1)).reshape(-1,1) # mini_batch_size  -> mini_batch_size x 1
                '''
                公式： ratio = pi_now/pi_old = exp(log(a)-log(b))
                In multi-dimensional continuous action space，we need to sum up the log_prob
                Only calculate the gradient of 'a_logprob_now' in ratios
                '''
                ratios = torch.exp(action_log_pi_now.sum(dim = 1, keepdim=True) - action_log_pi[index].sum(dim = 1, keepdim=True))  # shape(mini_batch_size X 1)
                surr1 = ratios * adv[index]  
                surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * adv[index]  
                actor_loss = -torch.min(surr1, surr2).mean() - entropy_coefficient * dist_entropy.mean()  
                self.agent.update_actor(actor_loss)

                # 再更新critic
                v_s = self.agent.critic(obs[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                self.agent.update_critic(critic_loss)
        
        ## 清空buffer
        self.buffer.clear()

    
    def lr_decay(self, episode_num,max_episodes):
        lr_a_now = self.actor_lr * (1 - episode_num / max_episodes)
        lr_c_now = self.critic_lr * (1 - episode_num / max_episodes)
        for p in self.agent.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.agent.critic_optimizer.param_groups:
            p['lr'] = lr_c_now
    
    ## 保存模型
    def save(self, model_dir):
        torch.save(self.agent.actor.state_dict(), os.path.join(model_dir,"PPO.pt"))
    
    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue ,model_dir,trick,beta):
        policy = PPO(dim_info,is_continue,0,0,0,device = torch.device("cpu"),trick=trick,beta = beta)
        policy.agent.actor.load_state_dict(torch.load(os.path.join(model_dir,"PPO.pt")))
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
    print('existing_dirs:',existing_dirs)
    max_number = 0 if not existing_dirs else max([int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()])
    model_dir = os.path.join(env_dir, prefix + str(max_number + 1))
    os.makedirs(model_dir)
    return model_dir
    
''' 
环境见：
离散:CartPole-v1,MountainCar-v0,;LunarLander-v2,;FrozenLake-v1 
连续：Pendulum-v1,MountainCarContinuous-v0,BipedalWalker-v3
reward_threshold：https://github.com/openai/gym/blob/master/gym/envs/__init__.py 
介绍：https://gymnasium.farama.org/
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="CartPole-v1") 
    # 共有参数
    parser.add_argument("--seed", type=int, default=0) # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(500))
    parser.add_argument("--save_freq", type=int, default=int(500//4))
    parser.add_argument("--start_steps", type=int, default=0) #ppo无此参数
    parser.add_argument("--random_steps", type=int, default=0)  #dqn 无此参数
    parser.add_argument("--learn_steps_interval", type=int, default=0)  # 这个算法不方便用
    parser.add_argument("--is_dis_to_con", type=bool, default=False) # dqn 默认为True
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    ## A-C参数
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    # PPO独有参数
    parser.add_argument("--horizon", type=int, default=2048)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--K_epochs", type=int, default=10)
    parser.add_argument("--entropy_coefficient", type=float, default=0.01)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--lmbda", type=float, default=0.95) # GAE参数
    # trick参数
    parser.add_argument("--policy_name", type=str, default='PPO')
    parser.add_argument("--trick", type=dict, default={'adv_norm':False,
                                                       'ObsNorm':False,'Batch_ObsNorm':False,  # or 两者择1
                                                       'reward_norm':False, 'reward_scaling':False, # or 
                                                       'lr_decay':False,'orthogonal_init':False,'adam_eps':False,'tanh':False})
    parser.add_argument("--beta", type=bool, default=False)
    # device参数
    parser.add_argument("--device", type=str, default='cpu') # cpu/cuda
    args = parser.parse_args()
    # 检查 reward_norm 和 reward_scaling 的值
    if args.trick['reward_norm'] and args.trick['reward_scaling']:
        raise ValueError("reward_norm 和 reward_scaling 不能同时为 True")
    if args.trick['ObsNorm'] and args.trick['Batch_ObsNorm']:
        raise ValueError("ObsNorm 和 Batch_ObsNorm 不能同时为 True")
    
    print(args)
    print('-' * 50)
    print('Algorithm:',args.policy_name)
    
    ## 环境配置
    env,dim_info,max_action,is_continue = get_env(args.env_name,args.is_dis_to_con)
    obs_dim,action_dim = dim_info
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
    policy = PPO(dim_info, is_continue, actor_lr = args.actor_lr, critic_lr = args.critic_lr, horizon = args.horizon, device = device,trick = args.trick,beta = args.beta)

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

    if args.trick['reward_norm']:  
        reward_norm = Normalization(shape=1)
    elif args.trick['reward_scaling']:  
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    
    while episode_num < args.max_episodes:
        step +=1

        # 获取动作
        action , action_log_pi = policy.select_action(obs)   # action (-1,1)
        if is_continue:
            action_ = np.clip(action * max_action , -max_action, max_action)
        else:
            action_ = action
        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        if args.trick['ObsNorm']:
            next_obs = obs_norm(next_obs)
        if args.trick['reward_norm']:  
            reward_ = reward_norm(reward)
        elif args.trick['reward_scaling']:
            reward_ = reward_scaling(reward)
        done = terminated or truncated
        done_bool = terminated     ### truncated 为超过最大步数
        if args.trick['reward_norm'] or args.trick['reward_scaling']:
            policy.add(obs, action, reward_, next_obs, done_bool, action_log_pi,done)
        else:
            policy.add(obs, action, reward, next_obs, done_bool, action_log_pi,done)
        episode_reward += reward
        obs = next_obs
        
        # episode 结束
        if done:
            ## 显示
            if  (episode_num + 1) % 100 == 0:
                print("episode: {}, reward: {}".format(episode_num + 1, episode_reward))
            writer.add_scalar('reward', episode_reward, episode_num + 1)
            train_return.append(episode_reward)

            episode_num += 1
            obs,info = env.reset(seed=args.seed)
            if args.trick['ObsNorm']:
                obs = obs_norm(obs)
            if args.trick['reward_scaling']:
                reward_scaling.reset()
            episode_reward = 0
        
        # 满足step,更新网络
        if step % args.horizon == 0:
            policy.learn(args.minibatch_size, args.gamma, args.lmbda, args.clip_param, args.K_epochs, args.entropy_coefficient)
            if args.trick['lr_decay']:
                policy.lr_decay(episode_num,max_episodes=args.max_episodes)
        
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