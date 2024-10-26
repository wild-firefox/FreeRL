'''
改成PPO_actor 输出mean 而std首先初始化然后衰减
'''
import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import MultivariateNormal

from copy import deepcopy
import numpy as np
from Buffer import Buffer_for_PPO

import gymnasium as gym
import argparse

## 其他
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
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.mean_layer = nn.Linear(hidden_2, action_dim)        # 此方法称为对角高斯函数 的主流方法1.对于每个action维度都有独立的方差 第二种方法 2.self.log_std_layer = nn.Linear(hidden_2, action_dim) log_std 是环境状态的函数

    def forward(self, x, ):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        mean = torch.tanh(self.mean_layer(x))  # 使得mean在-1,1之间
        return mean
    
'''
critic部分 与sac区别
区别:sac中critic输出Q1,Q2,而ppo中只输出V
'''    
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_1=128, hidden_2=128):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        value = self.l3(x)
        return value
    
class Agent:
    def __init__(self, obs_dim, action_dim, actor_lr, critic_lr, device):
        
        self.actor = Actor(obs_dim, action_dim, ).to(device)
        self.critic = Critic( obs_dim ).to(device)

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
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, horizon, device, trick = None):

        obs_dim, action_dim = dim_info
        self.agent = Agent(obs_dim, action_dim,  actor_lr, critic_lr, device)
        self.buffer = Buffer_for_PPO(horizon, obs_dim, act_dim = action_dim if is_continue else 1, device = device,) #Buffer中说明了act_dim和action_dim的区别
        self.device = device
        self.is_continue = is_continue

        self.horizon = int(horizon)
        self.trick = trick

        ## dacay std # 根据环境修改
        if self.is_continue:
            self.act_std_min=0.1
            self.act_std_init=0.4 # 0.4 
            self.act_std_temp = self.act_std_init
            #self.act_std = torch.ones(1,action_dim).to(device) * self.act_std_init
            self.act_std = torch.full((1,action_dim),self.act_std_temp)
            #self.action_var = torch.full((action_dim,), self.act_std **2).to(device)

    def select_action(self, obs):
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device) # 1xobs_dim
        ''' 这里先暂时实现连续动作空间的选择，后续再补充离散动作空间的选择 '''
        if self.is_continue: # dqn 无此项
            mean = self.agent.actor(obs) # 1xact_dim
            #cov_mat = torch.diag(self.action_var).unsqueeze(dim=0) # 协方差矩阵  act_dim x act_dim -> 1 x act_dim x act_dim
            std = self.act_std.expand_as(mean)
            dist = Normal(mean, std)
            action = dist.sample() # 1 x action
            action_log_pi = dist.log_prob(action) # 
            # to 真实值
            action = action.detach().cpu().numpy().squeeze(0) # 1xaction_dim ->action_dim
            action_log_pi = action_log_pi.detach().cpu().numpy().squeeze(0) # 1xaction_dim ->action_dim
        else:
            action = self.agent.Qnet(obs).argmax(dim = 1).detach().cpu().numpy()[0] # []->标量
        return action , action_log_pi
    
    def evaluate_action(self, obs):
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
        if self.is_continue:
            mean  = self.agent.actor(obs)
            action = mean.detach().cpu().numpy().squeeze(0)
        else:
            action = self.agent.Qnet(obs).argmax(dim = 1).detach().cpu().numpy()[0]
        return action
    
    ## buffer相关
    '''PPO论文中提到
    计算V_target 有两种方法1.generalized advantage estimation 2.finite-horizon estimators
    第2种实现方法在许多代码上的实现方法不一,有buffer中存入return和value值的方法,也有在buffer里不存，而在在更新时计算的方法。
    这里我们选择第1种,在buffer中不会存在上述争议。
    通常ppo的buffer中存储的是obs, action, reward, next_obs, done, log_pi ;
    比较1.不存储log_pi,而是在更新时计算出log_pi_old, 2.存储log_pi，将此作为log_pi_old 发现2更好 采用2
    '''
    def add(self, obs, action, reward, next_obs, done, action_log_pi , adv_dones):
        self.buffer.add(obs, action, reward, next_obs, done, action_log_pi , adv_dones)
    
    def sample(self,minibatch_size):
        indices = np.random.choice(self.horizon, minibatch_size, replace=False)
        obs, action, reward, next_obs, done = self.buffer.sample(indices)
        
        return obs, action, reward, next_obs, done,indices

    ## PPO算法相关
    '''
    论文：GENERALIZED ADVANTAGE ESTIMATION:https://arxiv.org/pdf/1506.02438 提到
    先更新critic会造成额外的偏差，所以PPO这里 先更新actor，再更新critic ,且PPO主要是策略更新的方法
    lmbda = 0 时 GAE 为 one-step TD = 1时，GAE 为 MC
    '''
    def learn(self, minibatch_size, gamma, lmbda ,clip_param, K_epochs, entropy_coefficient):
        '''
        done : dead or win
        adv_done : dead or win or reach max step
        gae 公式：A_t = delta_t + gamma * lmbda * A_t+1 * (1 - adv_done) 
        '''
        obs, action, reward, next_obs, done , action_log_pi , adv_dones = self.buffer.all()
        # 计算GAE
        with torch.no_grad():  # adv and v_target have no gradient
            adv = torch.zeros(self.horizon,dtype=torch.float32)
            gae = 0
            vs = self.agent.critic(obs)
            vs_ = self.agent.critic(next_obs)
            td_delta = reward + gamma * (1.0 - done) * vs_ - vs
            td_delta = td_delta.reshape(-1).cpu().detach().numpy()
            adv_dones = adv_dones.reshape(-1).cpu().detach().numpy()
            for i in reversed(range(self.horizon)):
                gae = td_delta[i] + gamma * lmbda * gae * (1.0 - adv_dones[i])
                adv[i] = gae
            adv = adv.reshape(-1, 1).to(self.device) ## cuda
            v_target = adv + vs  
            # if self.trick['adv_norm']:  # Trick 1:advantage normalization
            #     adv = ((adv - adv.mean()) / (adv.std() + 1e-5)) 

        # Optimize policy for K epochs:
        for _ in range(K_epochs): 
            # 随机打乱样本 并 生成小批量
            shuffled_indices = np.random.permutation(self.horizon)
            indexes = [shuffled_indices[i:i + minibatch_size] for i in range(0, self.horizon, minibatch_size)]
            for index in indexes:
                # 先更新actor
                mean = self.agent.actor(obs[index])
                # action_var = self.action_var.expand_as(mean) # 
                # cov_mat = torch.diag_embed(action_var).to(device) #
                std = self.act_std.expand_as(mean)
                dist_now = Normal(mean, std)
                dist_entropy = dist_now.entropy().sum(dim = 1, keepdim=True)  # mini_batch_size 
                action_log_pi_now = dist_now.log_prob(action[index])
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
    
    ## 保存模型
    def save(self, model_dir):
        torch.save(self.agent.actor.state_dict(), os.path.join(model_dir,"PPO.pt"))
    
    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue ,model_dir):
        policy = PPO(dim_info,is_continue,0,0,0,device = torch.device("cpu"))
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
环境见：CartPole-v1,Pendulum-v1,MountainCar-v0;LunarLander-v2,BipedalWalker-v3;FrozenLake-v1 
https://github.com/openai/gym/blob/master/gym/envs/__init__.py 
此算法 只写了连续域 BipedalWalker-v3  Pendulum-v1
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="BipedalWalker-v3") 
    parser.add_argument("--max_action", type=float, default=None)
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
    parser.add_argument("--horizon", type=int, default=2048) #根据环境更改：max_episodes数量级在百单位时horizon = 2048 minibatch_size = 64 小于百单位时，horizon = 128  minibatch_size = 32
    parser.add_argument("--clip_param", type=float, default=0.2) 
    parser.add_argument("--K_epochs", type=int, default=80)
    parser.add_argument("--entropy_coefficient", type=float, default=0.01)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--lmbda", type=float, default=0.95) # GAE参数
    # trick参数
    parser.add_argument("--policy_name", type=str, default='PPO_decaystd')
    parser.add_argument("--decaystd_freq", type=int, default=int(500//12))
    parser.add_argument("--trick", type=dict, default={'adv_norm':False}) 

    # device参数
    parser.add_argument("--device", type=str, default='cpu') # cpu/cuda
    args = parser.parse_args()
    
    ## 环境配置
    env,dim_info,max_action,is_continue = get_env(args.env_name,args.is_dis_to_con)
    max_action = max_action if max_action is not None else args.max_action
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
    writer = SummaryWriter(model_dir)

    ## device参数
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    
    ## 算法配置
    policy = PPO(dim_info, is_continue, actor_lr = args.actor_lr, critic_lr = args.critic_lr, horizon = args.horizon, device = device)

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

        # 获取动作
        action , action_log_pi = policy.select_action(obs)   # action (-1,1)
        action_ = np.clip(action * max_action , -max_action, max_action)
        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        done = terminated or truncated
        done_bool = done if not truncated  else False    ### truncated 为超过最大步数
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
            episode_reward = 0

            #if is_continue :
            # decay std
            if episode_num % args.decaystd_freq == 0:
                policy.act_std_temp = max((policy.act_std_temp - 0.04),policy.act_std_min)#(policy.act_std_init-policy.act_std_min)/args.max_episodes
                policy.act_std = torch.full((1,action_dim),policy.act_std_temp)
                #policy.action_var = torch.full((action_dim,), policy.act_std **2).to(device)
        
        # 满足step,更新网络
        if step % args.horizon == 0:
            policy.learn(args.minibatch_size, args.gamma, args.lmbda, args.clip_param, args.K_epochs, args.entropy_coefficient)

        
        # 保存模型
        if episode_num % args.save_freq == 0:
            policy.save(model_dir)

    
    print('total_time:',time.time()-time_)
    policy.save(model_dir)
    ## 保存数据
    np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),np.array(train_return))