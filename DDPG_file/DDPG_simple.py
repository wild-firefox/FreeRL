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

'''simple版为类似td3作者的版本 易于简单了解其算法本质'''
'''ddpg ：Deep DPG
论文：CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING  链接：https://arxiv.org/pdf/1509.02971
创新点（特点，与DQN的区别）：1.提出一种a model-free, off-policy actor-critic 算法 ，并证实了即使使用原始像素的obs,也能使用continue action spaces 稳健解决各种问题
2.不需要更改环境的情况下，可以进行稳定的学习。
3.比DQN需要更少的经验步就可以收敛。
可借鉴参数：
hidden：400 -300
actor_lr = 1e-4
critic_lr = 1e-3
buffer_size = 1e6
gamma = 0.99
tau = 0.001
std = 0.2
补充：
1.对Q使用了正则项l2进行1e-2来权重衰减
2.使用时间相关噪声来进行探索used an Ornstein-Uhlenbeck process theta=0.15 std=0.2
3.使用批量归一化状态值
4.对于低维环境 对actor和critic的全连接层的最后一层使用uniform distribution[-3e-3,3e-3],其余层为[-1/sqrt(f),1/sqrt(f)],f为输入的维度
对于pixel case  最后一层[-3e-4,3e-4](这里论文笔误写成[3e-4,3e-4])，其余层[-1/sqrt(f),1/sqrt(f)]
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
    
class Agent:
    def __init__(self, obs_dim, action_dim, dim_info,actor_lr, critic_lr, device, ):   
        self.actor = Actor(obs_dim, action_dim,).to(device)
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
class DDPG: 
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, buffer_size, device, trick = None):

        obs_dim, action_dim = dim_info
        self.agent = Agent(obs_dim, action_dim, dim_info, actor_lr, critic_lr, device)
        self.buffer = Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = device) #Buffer中说明了act_dim和action_dim的区别
        self.device = device
        self.is_continue = is_continue

        self.trick = trick

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
    def load(dim_info, is_continue ,model_dir,trick = None):
        policy = DDPG(dim_info,is_continue,0,0,0,device = torch.device("cpu"),trick = trick)
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
    parser.add_argument("--env_name", type = str,default="Pendulum-v1") 
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
    parser.add_argument("--batch_size", type=int, default=256)  #保证比start_steps小 # 256 (64 for MountainCarContinuous-v0)
    # DDPG 独有参数 noise
    ## gauss noise
    parser.add_argument("--gauss_sigma", type=float, default=0.1) # 高斯标准差 # 0.1 (1 for MountainCarContinuous-v0 )
    parser.add_argument("--gauss_scale", type=float, default=1)
    parser.add_argument("--gauss_init_scale", type=float, default=None) # 若不设置衰减，则设置成None
    parser.add_argument("--gauss_final_scale", type=float, default=0.0)
    # trick参数
    parser.add_argument("--policy_name", type=str, default='DDPG_simple')
    parser.add_argument("--trick", type=dict, default=None) 
    # device参数
    parser.add_argument("--device", type=str, default='cuda') # cpu/cuda
    
    args = parser.parse_args()
    print(args)
    print('-'*50)
    print('Algorithm:',args.policy_name)
    
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
    policy = DDPG(dim_info, is_continue, args.actor_lr, args.critic_lr, args.buffer_size, device, args.trick)

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
            policy.learn(args.batch_size, args.gamma, args.tau)
        
        # 保存模型
        if episode_num % args.save_freq == 0:
            policy.save(model_dir)

    
    print('total_time:',time.time()-time_)
    policy.save(model_dir)
    ## 保存数据
    np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),np.array(train_return))