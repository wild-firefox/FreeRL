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

## 进化算法相关
from ES import sepCEM

'''
CEM + GD3PG = CEM-GD3PG   论文名: Novel Evolutionary Deep Reinforcement Learning Algorithm:CEM-GD3PG
CEM：Cross Entropy Method + dual-replay buffer
GD3PG：Guide double actor ddpg
参考原始论文 改写  --针对连续域
'''
## 第一部分：定义Agent类
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128,layer_norm=False):
        super(Actor, self).__init__()
        if layer_norm :
            self.l1 = nn.Sequential(nn.Linear(obs_dim, hidden_1),nn.LayerNorm(hidden_1))
            self.l2 = nn.Sequential(nn.Linear(hidden_1, hidden_2),nn.LayerNorm(hidden_2))
            self.l3 = nn.Linear(hidden_2, action_dim)
        else:
            self.l1 = nn.Linear(obs_dim, hidden_1)
            self.l2 = nn.Linear(hidden_1, hidden_2)
            self.l3 = nn.Linear(hidden_2, action_dim)

        self.lambda_ =  10  #  the weighting parameter  原代码的 beta

    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
      
        x = F.tanh(self.l3(x)) # 输出动作(-1,1)
        return x
    
    def set_params(self, params):
        '''
        用于设置参数 将params参数设置到网络中
        '''
        cpt = 0
        for param in self.parameters():
            tmp = np.prod(param.size()) # product -> prod
            param.data.copy_(torch.as_tensor(params[cpt:cpt + tmp],dtype = torch.float32).reshape(param.size())) # to(device)
            cpt += tmp

    def get_params(self):
        '''
        用于获取参数 np.hstack = np.concatenate(axis=1)(后者必须二维)
        '''
        [param.detach().cpu().numpy().reshape(-1) for param in self.parameters()]
        return deepcopy(np.hstack([param.detach().cpu().numpy().reshape(-1) for param in self.parameters()]))
    
    def get_size(self):
        '''
        用于获取参数数量
        '''
        return len(self.get_params())

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128,layer_norm=False):
        super(Critic, self).__init__()
        if layer_norm :
            self.l1 = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_1),nn.LayerNorm(hidden_1))
            self.l2 = nn.Sequential(nn.Linear(hidden_1, hidden_2),nn.LayerNorm(hidden_2))
            self.l3 = nn.Linear(hidden_2, 1)
        else:
            self.l1 = nn.Linear(obs_dim + action_dim, hidden_1)
            self.l2 = nn.Linear(hidden_1, hidden_2)
            self.l3 = nn.Linear(hidden_2, 1)
        self.beta =  10 ###

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = self.l3(x)
        return x
    
class Agent:
    def __init__(self, obs_dim, action_dim, actor_lr, critic_lr, device):
        
        self.actor_1 = Actor(obs_dim, action_dim, )
        self.actor_2 = Actor(obs_dim, action_dim, )
        self.critic = Critic( obs_dim, action_dim, )

        self.actor_1_optimizer = torch.optim.Adam(self.actor_1.parameters(), lr=actor_lr)
        self.actor_2_optimizer = torch.optim.Adam(self.actor_2.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_1_target = deepcopy(self.actor_1)
        self.actor_2_target = deepcopy(self.actor_2)
        self.critic_target = deepcopy(self.critic)

        self.actor_domain = deepcopy(self.actor_2) # 挑选出来的actor

    def update_actor_1(self, loss):
        self.actor_1_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_1.parameters(), 0.5)
        self.actor_1_optimizer.step()

    def update_actor_2(self, loss):
        self.actor_2_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_2.parameters(), 0.5)
        self.actor_2_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
    
## 第二部分：定义DQN算法类
class CEM_GD3PG:
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, buffer_size, device, trick = None):
        obs_dim, action_dim = dim_info
        self.agent = Agent(obs_dim, action_dim, actor_lr, critic_lr, device)
        self.buffer = Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = device) #Buffer中说明了act_dim和action_dim的区别
        self.buffer_domain = Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = device)
        self.device = device
        self.is_continue = is_continue

    def select_action(self, obs , actor):
        '''
        输入的obs shape为(obs_dim) np.array reshape => (1,obs_dim) torch.tensor 
        若离散 则输出的action 为标量 <= [0]  (action_dim) np.array <= argmax (1,action_dim) torch.tensor 若keepdim=True则为(1,1) 否则为(1)
        若连续 则输出的action 为(1,action_dim) np.array <= (1,action_dim) torch.tensor
        '''
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
        if self.is_continue: # dqn 无此项
            action = actor(obs).detach().cpu().numpy().squeeze(0) # 1xaction_dim ->action_dim
        else:
            action = actor(obs).argmax(dim = 1).detach().cpu().numpy()[0] # []标量
        return action
    
    def evaluate_action(self, obs, actor):
        ''' 确定性策略 两者一致 改main中的noise的std'''
        return self.select_action(self, obs, actor)
    
    ## buffer相关
    def add(self, buffer , obs, action, reward, next_obs, done):
        buffer.add(obs, action, reward, next_obs, done)
    
    def sample(self, batch_size):
        total_size = len(self.buffer_domain)  # 取 暂时会较小的buffer
        batch_size = min(total_size, batch_size) # 防止batch_size比start_steps大, 一般可去掉

        half_size = batch_size // 2
        indices_buffer = np.random.choice(total_size, half_size, replace=False)  #默认True 重复采样 
        indices_buffer_domain = np.random.choice(total_size, half_size, replace=False)
        obs, actions, rewards, next_obs, dones = self.buffer.sample(indices_buffer)
        obs_domain, actions_domain, rewards_domain, next_obs_domain, dones_domain = self.buffer_domain.sample(indices_buffer_domain)
        
        obs = torch.cat([obs, obs_domain], dim = 0)
        actions = torch.cat([actions, actions_domain], dim = 0)
        rewards = torch.cat([rewards, rewards_domain], dim = 0)
        next_obs = torch.cat([next_obs, next_obs_domain], dim = 0)
        dones = torch.cat([dones, dones_domain], dim = 0)

        return obs, actions, rewards, next_obs, dones
    

    ## DQN算法相关
    def learn(self,batch_size, gamma, tau, is_F1_more, delta ,actor_domain):

        obs, actions, rewards, next_obs, dones = self.sample(batch_size) 

        # 先更新critic
        next_target_Q1 = self.agent.critic_target(next_obs,self.agent.actor_1_target(next_obs)) # batch_size x 1
        next_target_Q2 = self.agent.critic_target(next_obs,self.agent.actor_2_target(next_obs)) # batch_size x 1
        next_target_Q = torch.min(next_target_Q1, next_target_Q2) # batch_size x 1
        
        target_Q = rewards + gamma * next_target_Q * (1 - dones) # batch_size x 1

        current_Q = self.agent.critic(obs, actions) # batch_size x 1

        loss = F.mse_loss(current_Q, target_Q.detach()) # 标量值
        self.agent.update_critic(loss)
        
        # 再更新actor
        ''' 如果actor'''
        q_pi_1 = self.agent.critic(obs, self.agent.actor_1(obs)).mean()
        q_pi_2 = self.agent.critic(obs, self.agent.actor_2(obs)).mean()

        if is_F1_more:
            actor_loss_1 = -q_pi_1
            self.agent.update_actor_1(actor_loss_1)

            c = self.agent.actor_2(obs) - actor_domain(obs)
            KL = torch.sqrt(torch.sum(c * c) / len(c)) # 比原来的快
            actor_loss_2 = -q_pi_2 + self.agent.actor_2.lambda_ * delta * KL
            self.agent.update_actor_2(actor_loss_2)
        else:
            actor_loss_2 = -q_pi_2
            self.agent.update_actor_2(actor_loss_2)

            c = self.agent.actor_1(obs) - actor_domain(obs)
            KL = torch.sqrt(torch.sum(c * c) / len(c))
            actor_loss_1 = -q_pi_1 + self.agent.actor_1.lambda_ * delta * KL
            self.agent.update_actor_1(actor_loss_1)

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
        soft_update(self.agent.actor_1_target, self.agent.actor_1, tau)
        soft_update(self.agent.actor_2_target, self.agent.actor_2, tau)

    
    ## 保存模型
    def save(self, model_dir,actor_domain):
        ''' 保存 引导者 actor_domain '''
        torch.save(actor_domain.state_dict(), os.path.join(model_dir,"CEM_GD3PG.pt"))
    ## 加载模型
    @staticmethod 
    def load(dim_info, model_dir):
        policy = CEM_GD3PG(dim_info,0,0,0,0,device = torch.device("cpu"))
        policy.agent.actor_domain.load_state_dict(torch.load(os.path.join(model_dir,"CEM_GD3PG.pt")))
        return policy
    
## 第三部分：定义训练函数
''' 这里不用离散转连续域技巧'''
def get_env(env_name,dis_to_con_b = False):
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
        if dis_to_con_b :
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
    existing_dirs = [d for d in os.listdir(env_dir) if d.startswith(prefix)]
    max_number = 0 if not existing_dirs else max([int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()])
    model_dir = os.path.join(env_dir, prefix + str(max_number + 1))
    os.makedirs(model_dir)
    return model_dir
''' 
环境见：CartPole-v1,Pendulum-v1,MountainCar-v0;LunarLander-v2,BipedalWalker-v3;FrozenLake-v1 
https://github.com/openai/gym/blob/master/gym/envs/__init__.py 
此算法 只适用连续域 BipedalWalker-v3  Pendulum-v1
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="BipedalWalker-v3") 
    parser.add_argument("--max_action", type=float, default=None)
    # 共有参数
    parser.add_argument("--seed", type=int, default=0) # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(500))
    parser.add_argument("--start_steps", type=int, default=1000)
    parser.add_argument("--random_steps", type=int, default=0)  #dqn 无此参数
    parser.add_argument("--learn_steps_interval", type=int, default=1)  # 这个算法不方便用
    parser.add_argument("--dis_to_con_b", type=bool, default=False) # dqn 默认为True
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    ## A-C参数
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    ## buffer参数   
    parser.add_argument("--buffer_size", type=int, default=1e6) #1e6默认是float,在bufffer中有int强制转换
    parser.add_argument("--batch_size", type=int, default=256)  #保证比start_steps小
    # DDPG 独有参数 noise
    parser.add_argument("--gauss_sigma", type=float, default=0.1)
    # trick参数
    parser.add_argument("--policy_name", type=str, default='CEM_GD3PG')
    parser.add_argument("--trick", type=dict, default={'Double':False,'Dueling':False,'PER':False,'HER':False,'Noisy':False,'n_step':False})  

    # 进化算法独有参数
    parser.add_argument('--pop_size', type=int, default=10,)
    parser.add_argument('--elitism', dest="elitism", action='store_true')  #未用命令行 这里 False
    parser.add_argument('--n_grad', type=int, default=5,)  # 未用到
    parser.add_argument('--sigma_init', type=float, default=1e-3,)
    parser.add_argument('--damp', type=float, default=1e-3,)
    parser.add_argument('--damp_limit', type=float, default=1e-5,)
    parser.add_argument('--mult_noise', dest='mult_noise', action='store_true')  #未用命令行 这里 False


    args = parser.parse_args()

    ## 环境配置
    env,dim_info,max_action,is_continue = get_env(args.env_name,args.dis_to_con_b)
    max_action = max_action if max_action is not None else args.max_action

    action_dim = dim_info[1]

    ## 随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## 保存文件夹
    model_dir = make_dir(args.env_name,policy_name= args.policy_name ,trick=args.trick)
    writer = SummaryWriter(model_dir)

    ##
    device = torch.device('cpu')#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## 算法配置
    policy = CEM_GD3PG(dim_info,is_continue,args.actor_lr,args.critic_lr,args.buffer_size,device,args.trick)

    env_spec = gym.spec(args.env_name)
    print('reward_threshold:',env_spec.reward_threshold if env_spec.reward_threshold else 'No Threshold = Higher is better')
    time_ = time.time()

    ## 训练
    episode_num = 0
    step = 0
    #episode_reward = 0
    train_return = []
    obs,info = env.reset(seed=args.seed)

    def eval_actor(policy , actor, env , max_action , args , buffer = None ,noise = None,  ):
        ''' 评估一次'''
        episode_reward = 0
        steps = 0
        obs,info = env.reset(seed=args.seed)
        done = False
        while not done:
            # 获取动作
            if noise is not None: ### 将环境中的action_ 和要训练的action分开
                action = policy.select_action(obs,actor) 
                action_ = np.clip(action * max_action + np.random.normal(scale = args.gauss_sigma * max_action, size = action_dim), -max_action, max_action)
            else:
                action = policy.select_action(obs,actor) 
                action_ = action * max_action
            
            next_obs, reward, terminated, truncated, infos = env.step(action_)
            done = terminated or truncated
            done_bool = done if not truncated  else False
            if buffer is not None:
                policy.add(buffer , obs, action, reward, next_obs, done_bool)
            episode_reward += reward
            steps += 1

            obs = next_obs
            # episode 结束
        
        return episode_reward , steps

    ## 开始
    ## 进化算法 相关
    actor_1 = policy.agent.actor_1
    actor_2 = policy.agent.actor_2
    actor_domain = policy.agent.actor_domain 
    buffer = policy.buffer
    buffer_domain = policy.buffer_domain
    es = sepCEM(actor_1.get_size(), mu_init = actor_1.get_params(), sigma_init = args.sigma_init, damp = args.damp,damp_limit = args.damp_limit,
            pop_size = args.pop_size, antithetic = not args.pop_size % 2, parents = args.pop_size // 2, elitism=args.elitism)
    #f_domain = 0
    f1_total = 0
    f2_total = 0
    fitness = []
    cnt_es = 0 # 计数器
    es_params = es.ask(args.pop_size * 2)  # 生成参数 2*n 个种群
    fitness = []
    ### 初始化
    for i in range(args.pop_size):
        actor_domain.set_params(es_params[i])
        f, _ = eval_actor(policy,actor_domain,env,max_action,args,buffer = buffer)  # 这里造成buffer先比buffer_domain 多10个轮次的经验
        fitness.append(f)
    
    while episode_num < args.max_episodes:
        
        if cnt_es == args.pop_size:
            es.tell(es_params, fitness) # 更新分布
            es_params_half = es.ask(args.pop_size) # 生成一半种群
            fitness = []
            cnt_es = 0
            for params in es_params_half:
                actor_domain.set_params(params)
                f, _ = eval_actor(policy,actor_domain,env,max_action,args,buffer = buffer)
                fitness.append(f)
            
            idx_sort = np.argsort(fitness) # 按从小到大排序，返回索引
            max_idx = idx_sort[-1]
            print(f'select best actor : Actor{max_idx} ,fitness :{fitness[max_idx]}')
            if f1_total >= f2_total:
                actor_2.set_params((es_params_half[max_idx] + actor_2.get_params()) / 2) # 论文里的beta = 0.5 公式: pi_poor = (1 - beta) * pi_best + beta * pi_poor 
            else:
                actor_1.set_params((es_params_half[max_idx] + actor_1.get_params()) / 2)
            es_params[:args.pop_size] = es_params_half # 更新 前一半种群
        
        # 分别计算actor1和actor2的fitness
        f1 , _ = eval_actor(policy,actor_1,env,max_action,args,buffer = None)
        f2 , _ = eval_actor(policy,actor_2,env,max_action,args,buffer = None)
        f_best = max(f1,f2)
        f1_total = 0.8 * f1_total + 0.2 * f1  # 论文里的alpha = 0.2 公式: f_total = (1 - alpha) * f_total + alpha * f
        f2_total = 0.8 * f2_total + 0.2 * f2
        if f1_total >= f2_total:
            #f_domain = f1
            actor_domain.set_params(actor_1.get_params())
            if f1_total > 0:
                delta = 1 if (1 - f2_total / f1_total) > 1 else 1 - f2_total / f1_total
            else:
                delta = 1 - f1_total / f2_total 
        else:
            #f_domain = f2
            actor_domain.set_params(actor_2.get_params())
            if f2_total > 0:
                delta = 1 if (1 - f1_total / f2_total) > 1 else 1 - f1_total / f2_total
            else:
                delta = 1 - f2_total / f1_total

        
        es_params[cnt_es + args.pop_size] = actor_domain.get_params() # 更新 后半部分种群
        fitness.append(f_best)
        cnt_es += 1

        #
        episode_reward ,steps = eval_actor(policy,actor_domain,env,max_action,args,buffer = buffer_domain,noise = True)
        args.gauss_sigma = max(0.05, args.gauss_sigma * 0.999)
        ## 显示
        if  (episode_num + 1) % 100 == 0:
            print("episode: {}, reward: {}".format(episode_num + 1, episode_reward))
        writer.add_scalar('reward_domain', episode_reward, episode_num + 1)
        train_return.append(episode_reward)
        episode_num += 1
        step += steps

        if step > args.start_steps :
            for _ in range(steps):
                policy.learn(args.batch_size, args.gamma, args.tau, f1_total >= f2_total, delta,actor_domain)


    print('total_time:',time.time()-time_)
    policy.save(model_dir,actor_domain)
    ## 保存数据
    np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),np.array(train_return))



    
