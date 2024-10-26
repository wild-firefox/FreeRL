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
import re
import time
from torch.utils.tensorboard import SummaryWriter

'''TD3 修改于td3作者的代码，只是框架结构的改动,所以这里在DDPG_simple.py上改'''
'''TD3:  Twin Delayed Deep Deterministic policy gradient algorithm 
论文: https://arxiv.org/abs/1802.09477 代码:https://github.com/sfujim/TD3/blob/master/TD3.py
论文认为：使用高斯噪音比使用OUNoise更好。
创新点：
1.双截断Q网络  a clipped Double Q-learning variant Q高估时也能近似真值 论文中：CDQ
2.目标策略加噪声  policy noise (目标策略平滑正则化) target policy smoothin 论文中:TPS
3.延迟更新策略网络和目标网络 delaying policy updates 论文中:DP
可借鉴参数：
hidden：256-256
actor_lr = 1e-4
critic_lr = 1e-3
buffer_size = 1e6
batch_size = 256
gamma = 0.99
tau = 0.005
std = 0.1 # 高斯noise
### TD3独有
policy_std = 0.2 # 目标策略加噪声
noise_clip = 0.5 # 噪声截断
policy_freq = 2 # 延迟更新策略网络和目标网络

另外：论文中提出：可能使用组合的技巧，如：CDQ+TPS+DP 才会取得更好的效果,并在附录F中给出了消融实验。
在实验中可看出TD3作者写出的ourddpg(AHE),即DDPG_file中的DDPG_simple,比DDPG稍好一点(4个环境中1个环境落后一点，1个环境相差不大,一个环境领先一点，一个环境遥遥领先)
实验1中, 可得出单单加入创新点1,2,3 对于原算法,效果都有可能下降。
实验2中, 可得出同时加入创新点1,3,(1,2)(2,3) 对于原算法,效果都有提升。
实验3中, 可得出同时加入创新点2,3 对于原算法,效果有提升。
'''


## 第一部分：定义Agent类
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128 ):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim)

    def forward(self, obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        x = F.tanh(self.l3(x))

        return x
    
class Critic(nn.Module):
    def __init__(self, dim_info:list, hidden_1=128 , hidden_2=128):
        super(Critic, self).__init__()
        obs_act_dim = sum(dim_info)  
        
        self.l1 = nn.Linear(obs_act_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)

    def forward(self, o, a): # 传入观测和动作
        oa = torch.cat([o,a], dim = 1)
        
        q = F.relu(self.l1(oa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q

''' 创新点1：双截断Q网络'''
class Critic_TD3(nn.Module):
    def __init__(self, dim_info:list, hidden_1=128 , hidden_2=128):
        super(Critic_TD3, self).__init__()
        obs_act_dim = sum(dim_info)  
        
        # Q1
        self.l1 = nn.Linear(obs_act_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)

        # Q2
        self.l4 = nn.Linear(obs_act_dim, hidden_1)
        self.l5 = nn.Linear(hidden_1, hidden_2)
        self.l6 = nn.Linear(hidden_2, 1)


    def forward(self, o, a): # 传入观测和动作
        oa = torch.cat([o,a], dim = 1)
        
        q1 = F.relu(self.l1(oa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(oa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2
    
    def Q1(self, o, a):
        oa = torch.cat([o,a], dim = 1)
        
        q1 = F.relu(self.l1(oa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1
    
class Agent:
    def __init__(self, obs_dim, action_dim, dim_info,actor_lr, critic_lr, device, realize):   
        self.actor = Actor(obs_dim, action_dim,).to(device)
        if realize['clip_double']:
            self.critic = Critic_TD3( dim_info ).to(device)
        else:
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
class TD3: 
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, buffer_size, device, trick = None,realize = None):

        obs_dim, action_dim = dim_info
        self.agent = Agent(obs_dim, action_dim, dim_info, actor_lr, critic_lr, device, realize)
        self.buffer = Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = device) #Buffer中说明了act_dim和action_dim的区别
        self.device = device
        self.is_continue = is_continue

        self.trick = trick
        self.realize = realize
        self.total_it = 0 # 记录更新次数

    def select_action(self, obs):
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device) # 1xobs_dim
        # 先实现连续域下的ddpg
        if self.is_continue: # dqn 无此项
            action = self.agent.actor(obs).detach().cpu().numpy().squeeze(0) # 1xaction_dim -> action_dim
        else:
            action = self.agent.argmax(dim = 1).detach().cpu().numpy()[0] # []标量
        return action
    
    def evaluate_action(self, obs):
        '''确定性策略ddpg,在main中去掉noise'''
        return self.select_action(obs)

    ## buffer相关
    def add(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)
    
    def sample(self, batch_size):
        total_size = len(self.buffer)
        batch_size = min(total_size, batch_size) # 防止batch_size比start_steps大, 一般可去掉
        indices = np.random.choice(total_size, batch_size, replace=False)  #默认True 重复采样 
        obs, actions, rewards, next_obs, dones = self.buffer.sample(indices)

        return obs, actions, rewards, next_obs, dones
    
    ## 算法相关
    def learn(self,batch_size, gamma, tau, policy_noise, noise_clip,max_action, policy_freq,policy_noise_scale):

        self.total_it += 1

        obs, actions, rewards, next_obs, dones = self.sample(batch_size) 
        
        '''类似于使用了Double的技巧 + target网络技巧'''
        if self.realize['policy_noise']:
            noise = (policy_noise_scale * (torch.randn_like(actions) * policy_noise)).clamp(-noise_clip, noise_clip)
            next_action = (self.agent.actor_target(next_obs) * max_action + noise).clamp(-max_action, max_action) / max_action # 归一到[-1,1]
        else:
            next_action = self.agent.actor_target(next_obs)

        if self.realize['clip_double']:
            next_Q1, next_Q2 = self.agent.critic_target(next_obs, next_action)
            next_Q_target = torch.min(next_Q1, next_Q2) # batch_size x 1
        else:
            next_Q_target = self.agent.critic_target(next_obs, next_action) # batch_size x 1
        
        ## 先更新critic
        target_Q = rewards + gamma * next_Q_target * (1 - dones) # batch_size x 1
        if self.realize['clip_double']:
            current_Q1,current_Q2 = self.agent.critic(obs ,actions)# batch_size x 1
            critic_loss = F.mse_loss(current_Q1, target_Q.detach()) + F.mse_loss(current_Q2, target_Q.detach()) # 标量值
        else:
            current_Q = self.agent.critic(obs ,actions)# batch_size x 1
            critic_loss = F.mse_loss(current_Q, target_Q.detach()) # 标量值
        self.agent.update_critic(critic_loss)

        ## 再更新actor
        if self.realize['twin_delay']:
            pass
        else:
            policy_freq = 1

        if self.total_it % policy_freq == 0:
            new_action = self.agent.actor(obs)
            if self.realize['clip_double']:
                actor_loss = -self.agent.critic.Q1(obs, new_action).mean()
                '''github上也有 actor_loss = -min(Q1,Q2).mean() 的写法'''
            else:
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
        torch.save(self.agent.actor.state_dict(), os.path.join(model_dir,"TD3.pt"))
    
    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue ,model_dir,trick = None,realize = None):
        policy = TD3(dim_info,is_continue,0,0,0,device = torch.device("cpu"),trick = trick,realize = realize)
        policy.agent.actor.load_state_dict(torch.load(os.path.join(model_dir,"TD3.pt")))
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
    parser.add_argument("--seed", type=int, default=100) # 0 10 100
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
    parser.add_argument("--batch_size", type=int, default=64)  #保证比start_steps小 # 256 (64 for MountainCarContinuous-v0)
    # DDPG 独有参数 noise
    ## gauss noise
    parser.add_argument("--gauss_sigma", type=float, default=1) # 高斯标准差 # 0.1 (1 for MountainCarContinuous-v0 )
    parser.add_argument("--gauss_scale", type=float, default=1)
    parser.add_argument("--gauss_init_scale", type=float, default=1) # 若不设置衰减，则设置成None
    parser.add_argument("--gauss_final_scale", type=float, default=0.0)
    ## TD3独有参数
    parser.add_argument('--policy_noise',type=float,default=0.1) # 目标策略加噪声 # 0.2 (0.1 for MountainCarContinuous-v0)
    parser.add_argument('--noise_clip',type=float,default=0.5) # 噪声截断  # 0.5 ### 也加入衰减
    parser.add_argument('--policy_freq',type=int,default=2) # 延迟更新策略网络和目标网络
    parser.add_argument("--policy_noise_scale", type=float, default=1)  # 
    parser.add_argument("--policy_noise_init_scale", type=float, default=None) #若不设置衰减，则设置成None
    # trick参数
    parser.add_argument("--policy_name", type=str, default='TD3')
    parser.add_argument("--realize", type=dict, default={'clip_double':True,'policy_noise':True,'twin_delay':True}) 
    parser.add_argument("--trick", type=dict, default=None) 
    # device参数
    parser.add_argument("--device", type=str, default='cpu') # cpu/cuda
    
    args = parser.parse_args()
    print('Algorithm:',args.policy_name)
    if args.policy_name == 'TD3':
        args.realize = {'clip_double':True,'policy_noise':True,'twin_delay':True}
    print(args)


    
    ## 环境配置
    env,dim_info,max_action,is_continue = get_env(args.env_name,args.is_dis_to_con)
    action_dim = dim_info[1]
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
    policy = TD3(dim_info, is_continue, args.actor_lr, args.critic_lr, args.buffer_size, device, args.trick,args.realize)

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
    if args.policy_noise_init_scale is not None:
        args.policy_noise_scale = args.policy_noise_init_scale
    
    while episode_num < args.max_episodes:
        step +=1

        # 获取动作 区分动作action_为环境中的动作 action为要训练的动作
        if step < args.random_steps:
            action_ = env.action_space.sample()  # [-max_action , max_action]
            action = action_ / max_action # -> [-1,1]
        else:
            action = policy.select_action(obs)  # (-1,1)
            action_ = np.clip(action * max_action + args.gauss_scale * np.random.normal(scale = args.gauss_sigma * max_action, size = action_dim), -max_action, max_action)

        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        done = terminated or truncated
        done_bool = terminated     ### truncated 为超过最大步数
        policy.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        obs = next_obs
        
        # episode 结束
        if done:
            ## gauss_noise 衰减
            if args.gauss_init_scale is not None:
                explr_pct_remaining = max(0, args.max_episodes - (episode_num + 1)) / args.max_episodes
                args.gauss_scale = args.gauss_final_scale + (args.gauss_init_scale - args.gauss_final_scale) * explr_pct_remaining
            if args.policy_noise_scale is not None:
                explr_pct_remaining = max(0, args.max_episodes - (episode_num + 1)) / args.max_episodes
                args.policy_noise_scale = 0 + (args.policy_noise_scale - 0) * explr_pct_remaining
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
            episode_reward = 0
        
        # 满足step,更新网络
        if step > args.start_steps and step % args.learn_steps_interval == 0:
            policy.learn(args.batch_size, args.gamma, args.tau, args.policy_noise,args.noise_clip,max_action,args.policy_freq,args.policy_noise_scale)
        
        # 保存模型
        if episode_num % args.save_freq == 0:
            policy.save(model_dir)

    
    print('total_time:',time.time()-time_)
    policy.save(model_dir)
    ## 保存数据
    np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),np.array(train_return))