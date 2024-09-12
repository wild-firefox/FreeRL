import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
import numpy as np
from Buffer import Buffer

import gymnasium as gym
import argparse

## 其他
import time
from torch.utils.tensorboard import SummaryWriter

'''
一个深度强化学习算法分三个部分实现：
1.Agent类:包括actor、critic、target_actor、target_critic、actor_optimizer、critic_optimizer、
2.DQN算法类:包括select_action,learn、save、load等方法,为具体的算法细节实现
3.main函数:实例化DQN类,主要参数的设置,训练、测试、保存模型等
'''
'''DQN论文链接:https://arxiv.org/pdf/1312.5602'''
'''  参数修改 改三处 1.MLP的hidden  2.main中args 3.dis_to_con中的离散转连续空间维度 '''
## 第一部分：定义Agent类
class MLP(nn.Module):
    '''
    只有一层隐藏层的多层感知机。
    batch_size x obs_dim -> batch_size x hidden -> batch_size x action_dim
    公式：a = relu(W1*s + b1), q = W2*a + b2
    '''
    def __init__(self, obs_dim, action_dim, hidden = 128):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden)
        self.l2 = nn.Linear(hidden, action_dim)

    def forward(self, obs):
        a = F.relu(self.l1(obs))
        return self.l2(a)


class Agent:
    def __init__(self, obs_dim, action_dim, Qnet_lr , device):

        self.Qnet = MLP(obs_dim, action_dim).to(device)
        self.Qnet_target = deepcopy(self.Qnet)

        self.Qnet_optimizer = torch.optim.Adam(self.Qnet.parameters(), lr = Qnet_lr)
    
    def update_Qnet(self, loss):
        self.Qnet_optimizer.zero_grad()
        loss.backward()
        self.Qnet_optimizer.step()

## 第二部分：定义DQN算法类
class DQN:
    def __init__(self, dim_info, is_continue, Qnet_lr, buffer_size, device, trick = None):
        obs_dim, action_dim = dim_info
        self.agent = Agent(obs_dim, action_dim, Qnet_lr, device)
        self.buffer = Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = device) #Buffer中说明了act_dim和action_dim的区别
        self.device = device
        self.is_continue = is_continue

    def select_action(self, obs):
        '''
        输入的obs shape为(obs_dim) np.array reshape => (1,obs_dim) torch.tensor 
        若离散 则输出的action 为标量 <= [0]  (action_dim) np.array <= argmax (1,action_dim) torch.tensor 若keepdim=True则为(1,1) 否则为(1)
        若连续 则输出的action 为(1,action_dim) np.array <= (1,action_dim) torch.tensor
        is_continue 指定buffer和算法是否是连续动作 表示输出动作是否为连续动作
        '''
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
        if self.is_continue:  # dqn 均取离散动作
            action = self.agent.Qnet(obs).detach().cpu().numpy().squeeze(0) # 1xaction_dim ->action_dim
        else:
            action = self.agent.Qnet(obs).argmax(dim = 1).detach().cpu().numpy()[0] # []标量
        return action
    
    def evaluate_action(self, obs):
        '''DQN的探索策略是ε-greedy, 评估时,在main中去掉ε就行。类似于确定性策略ddpg。'''
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
    

    ## DQN算法相关
    def learn(self,batch_size, gamma, tau):
        '''
        公式: target_Q = rewards + gamma * max_a(Q_target(next_obs, a)) * (1 - dones)
        '''
        obs, actions, rewards, next_obs, dones = self.sample(batch_size) 

        next_target_Q = self.agent.Qnet_target(next_obs).max(dim = 1)[0].reshape(-1, 1) # batch_size x 1
        
        target_Q = rewards + gamma * next_target_Q * (1 - dones) # batch_size x 1

        current_Q = self.agent.Qnet(obs).gather(dim =1, index =actions.long()) # batch_size x action_dim -> batch_size x 1

        loss = F.mse_loss(current_Q, target_Q.detach()) # 标量值
        self.agent.update_Qnet(loss)
        self.update_target(tau)
    
    def update_target(self, tau):
        '''
        更新目标网络参数: θ_target = τ*θ_local + (1 - τ)*θ_target
        切断自举,缓解高估Q值 source -> target
        '''
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        soft_update(self.agent.Qnet_target, self.agent.Qnet, tau)

    ## 保存模型
    def save(self, model_dir):
        torch.save(self.agent.Qnet.state_dict(), os.path.join(model_dir,"DQN.pt"))
    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue ,model_dir):
        policy = DQN(dim_info,is_continue,0,0,device = torch.device("cpu"))
        policy.agent.Qnet.load_state_dict(torch.load(os.path.join(model_dir,"DQN.pt")))
        return policy

### 第三部分：main函数
## 环境配置
def get_env(env_name,is_dis_to_con = False):
    env = gym.make(env_name)
    if isinstance(env.observation_space, gym.spaces.Box):
        obs_dim = env.observation_space.shape[0]
    else:
        obs_dim = 1 # 例：Taxi-v3 离散状态空间
    if isinstance(env.action_space, gym.spaces.Box): # 是否动作连续环境
        action_dim = env.action_space.shape[0]
        dim_info = [obs_dim,action_dim]
        max_action = env.action_space.high[0]
        is_continue = True # 指定buffer和算法是否是连续动作 即输出动作是否是连续动作
        if is_dis_to_con :
            if action_dim == 1:
                dim_info = [obs_dim,16]  # 离散动作空间
                max_action = None
                is_continue = False
            else: # 多重连续动作空间->多重离散动作空间 例：BipedalWalker-v3
                dim_info = [obs_dim,2**action_dim]  # 离散动作空间
                max_action = None
                is_continue = False
    else:
        action_dim = env.action_space.n
        dim_info = [obs_dim,action_dim]
        max_action = None
        is_continue = False
    
    return env,dim_info, max_action, is_continue #dqn中均转为离散域.max_action没用到

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

## dis_to_cont
def dis_to_con(discrete_action, env, action_dim):  # 离散动作转连续的函数
    if env.action_space.shape[0] == 1:
        action_lowbound = env.action_space.low[0]  # 连续动作的最小值
        action_upbound = env.action_space.high[0]  # 连续动作的最大值
        return np.array([action_lowbound + (discrete_action /(action_dim - 1)) * (action_upbound -action_lowbound)])
    else:
        action_lowbound = env.action_space.low
        action_upbound = env.action_space.high
        action_dim_per = int(action_dim ** (1 / len(action_lowbound)))
        shape = env.action_space.shape[0]
        discrete_indices = [discrete_action // (action_dim_per ** i) % (action_dim_per) for i in range(shape)]  # 这里存储的是每个维度的离散索引 相当于8进制的表示
        continue_action = [action_lowbound[i] + discrete_indices[i] / (action_dim_per - 1) * (action_upbound[i] - action_lowbound[i]) for i in range(shape)]
        return np.array(continue_action)

''' 
环境见：CartPole-v1,Pendulum-v1,MountainCar-v0;LunarLander-v2,BipedalWalker-v3;FrozenLake-v1 
https://github.com/openai/gym/blob/master/gym/envs/__init__.py
FrozenLake-v1 在5000episode下比较好
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="Pendulum-v1") 
    parser.add_argument("--max_action", type=float, default=None)
    # 共有参数
    parser.add_argument("--seed", type=int, default=0) # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(500))
    parser.add_argument("--save_freq", type=int, default=int(500//4))
    parser.add_argument("--start_steps", type=int, default=500)
    parser.add_argument("--random_steps", type=int, default=0)  #dqn 无此参数
    parser.add_argument("--learn_steps_interval", type=int, default=1)
    parser.add_argument("--is_dis_to_con", type=bool, default=True) # dqn 默认为True
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    ## DQN参数
    parser.add_argument("--Qnet_lr", type=float, default=1e-3)
    ## buffer参数   
    parser.add_argument("--buffer_size", type=int, default=int(1e6)) #1e6默认是float,在bufffer中有int强制转换
    parser.add_argument("--batch_size", type=int, default=256)  #保证比start_steps小
    # DQN独有参数
    parser.add_argument("--epsilon", type=float, default=0.1)
    # trick参数
    parser.add_argument("--policy_name", type=str, default='DQN')
    parser.add_argument("--trick", type=dict, default={'Double':False,'Dueling':False,'PER':False,'Noisy':False,'N_Step':False,'Categorical':False})    
    # device参数
    parser.add_argument("--device", type=str, default='cpu')
    
    args = parser.parse_args()
    print(args)
    print('Algorithm:',args.policy_name)
    ## 环境配置
    env,dim_info,max_action,is_continue = get_env(args.env_name,args.is_dis_to_con)
    max_action = max_action if max_action is not None else args.max_action
    action_dim = dim_info[1]
    print(f'Env:{args.env_name}  obs_dim:{dim_info[0]}  action_dim:{dim_info[1]}  max_action:{max_action}  max_episodes:{args.max_episodes}')

    ## 随机数种子(cpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ### cuda
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random Seed:',args.seed)

    ## 保存文件夹
    model_dir = make_dir(args.env_name,policy_name = args.policy_name ,trick=args.trick)
    writer = SummaryWriter(model_dir)

    ## device参数
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    
    ## 算法配置
    policy = DQN(dim_info, is_continue, Qnet_lr = args.Qnet_lr, buffer_size = args.buffer_size,device = device,)

    env_spec = gym.spec(args.env_name)
    print('reward_threshold:',env_spec.reward_threshold if env_spec.reward_threshold else 'No Threshold = Higher is better')
    time_ = time.time()
    ## 训练
    episode_num = 0
    step = 0
    episode_reward = 0
    train_return = []
    obs,info = env.reset(seed=args.seed)
    while episode_num < args.max_episodes:
        step +=1

        # 获取动作 区分动作action_为环境中的动作 action为要训练的动作
        if step < args.random_steps:
            action = env.action_space.sample()
        else:
            if np.random.rand() < args.epsilon:
                action = np.random.randint(action_dim)
            else:
                action = policy.select_action(obs)   
        if args.is_dis_to_con and isinstance(env.action_space, gym.spaces.Box):
            action_ = dis_to_con(action, env, action_dim)
        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        done = terminated or truncated
        done_bool = terminated     ### truncated 为超过最大步数
        policy.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        obs = next_obs
        
        # episode 结束
        if done:
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
        
        if episode_num % args.save_freq == 0:
            policy.save(model_dir)

    
    print('total_time:',time.time()-time_)
    policy.save(model_dir)
    ## 保存数据
    np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),np.array(train_return))




