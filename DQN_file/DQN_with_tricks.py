import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
import numpy as np
from Buffer import Buffer,PER_Buffer,N_Step_Buffer,N_Step_PER_Buffer
from Noisy_net import NoisyLinear

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
'''  参数修改 改三处 1.MLP的hidden  2.main中args 3.dis_to_con中的离散转连续空间的维度 '''
'''
rainbow-dqn : trick 1-6                                                链接: https://arxiv.org/abs/1710.02298 
tricks实现:
1.Double DQN:将Q值的选择和评估分开，减少Q值的高估                        链接: https://arxiv.org/abs/1509.06461
2.Dueling DQN:将Q值分解为状态值V和优势值A，学习更好的Q值                 链接: https://arxiv.org/abs/1511.06581
3.Prioritized Experience Replay:优先级经验回放，根据TD误差来更新优先级   链接: https://arxiv.org/abs/1511.05952
4.Noise DQN:添加噪声，增强探索能力，提高鲁棒性                          链接: https://arxiv.org/abs/1706.10295
5.N-step:将Q值计算到未来N步，提供更准确的状态价值估计，更快收敛          链接: https://arxiv.org/abs/1901.07510
6.Categorical:将Q值表示为一组分布，理解不同动作的潜在价值               链接: https://arxiv.org/abs/1707.06887
'''

## 第一部分：定义Agent类
class MLP(nn.Module):
    '''
    只有一层隐藏层的多层感知机。
    batch_size x obs_dim -> batch_size x hidden -> batch_size x action_dim
    公式：a = relu(W1*s + b1), q = W2*a + b2
    '''
    def __init__(self, obs_dim, action_dim, hidden = 128,trick = None,):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden)
        ''' NoisyLinear:可以将所有层都改成NoisyLinear , 也可以只将输出层改成NoisyLinear '''
        if trick['Noisy']:
            self.l2 = NoisyLinear(hidden, action_dim,)
        else:
            self.l2 = nn.Linear(hidden, action_dim)
        self.trick = trick

    def forward(self, obs):
        a = F.relu(self.l1(obs))
        return self.l2(a)

class Dueling(nn.Module):
    '''
    Dueling DQN的网络结构  学Q值的时候，分开学V和A
    公式：Q = V + A - A.mean()
    '''
    def __init__(self, obs_dim, action_dim, hidden = 128,trick = None,):
        super(Dueling, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden)
        if trick['Noisy']:
            self.V = NoisyLinear(hidden, 1,)
            self.A = NoisyLinear(hidden, action_dim,)
        else:
            self.V = nn.Linear(hidden, 1)
            self.A = nn.Linear(hidden, action_dim)

    def forward(self, obs):
        a = F.relu(self.l1(obs))
        V = self.V(a)
        A = self.A(a)
        return V + A - A.mean(dim = 1, keepdim = True)

class Categorical(nn.Module):
    '''
    Categorical DQN的网络结构 使用softmax估计概率
    公式: Q = z * p(z|s,a)
    参考：https://github.com/XinJingHao/DRL-Pytorch/blob/main/2.4_Categorical-DQN_C51/Categorical_DQN.py
    '''
    def __init__(self, obs_dim, action_dim, hidden = 128,atoms_num = 51,v_max = 100,v_min = -100,trick = None, batch_size = None ,device = None) :
        super(Categorical, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden)
        
        if trick['Noisy'] and trick['Dueling']:
            self.V = NoisyLinear(hidden, atoms_num,)
            self.A = NoisyLinear(hidden, action_dim * atoms_num)
        elif trick['Noisy']:
            self.l2 = NoisyLinear(hidden, action_dim,)
        elif trick['Dueling']:
            self.V = nn.Linear(hidden, atoms_num,)
            self.A = nn.Linear(hidden, action_dim * atoms_num)
        else:
            self.l2 = nn.Linear(hidden, action_dim * atoms_num)

        self.action_dim = action_dim
        self.trick = trick
        self.device = device
        
        self.atoms_num = atoms_num
        self.v_min = v_min
        self.v_max = v_max
        self.z = torch.linspace(v_min, v_max, steps = atoms_num).to(device)  ## cude
        self.delta_z = (v_max - v_min) / (atoms_num - 1)
        self.offset = torch.linspace(0, (batch_size - 1) * atoms_num, batch_size,).reshape(-1,1).long().to(device) # (batch_size,1)  ## cuda
        
    def _predict(self, obs):
        x = F.relu(self.l1(obs))
        if self.trick['Dueling']:
            V = self.V(x)
            A = self.A(x)
            V = V.reshape(-1, 1, self.atoms_num)
            A = A.reshape(-1, self.action_dim, self.atoms_num)
            logits = V + A - A.mean(dim = 1, keepdim = True)
        else:
            logits = self.l2(x)

        dist = torch.softmax(logits.reshape(-1, self.action_dim, self.atoms_num), dim=2) # batch_size x action_dim x atoms_num
        q = (dist * self.z).sum(dim=2) # batch_size x action_dim
        return dist, q

    def forward(self, obs, action=None):
        dist, q = self._predict(obs)
        if action is None:
            action = torch.argmax(q, dim=1) # batch_size
        return action, dist[torch.arange(len(obs)), action.reshape(-1).long()]
        #return action, dist.gather(1, action.reshape(-1,1,1).expand(-1,1,self.atoms_num).long()).squeeze(1) #与上一行 两者结果一致
    
    def projection_dist(self, Qnet, Qnet_target, next_obs, rewards, dones, gamma,):
        """
        计算目标分布，公式：m = (u - b) * next_dist + (1 - u + b) * next_dist
        """
        batch_size = len(next_obs)
        m = torch.zeros(batch_size,self.atoms_num).to(self.device)  ## cude
        if self.trick['Double']:
            next_actions , _ = Qnet(next_obs)
            _ , next_dist = Qnet_target(next_obs,next_actions)
        else:
            _, next_dist = Qnet_target(next_obs)  # batch_size x atoms_num

        t_z = (rewards + gamma * self.z * (1-dones)).clamp(self.v_min, self.v_max) # batch_size x atoms_num
        b = (t_z - self.v_min)/self.delta_z # b∈[0,atoms_num-1]; batch_size x atoms_num
        l = b.floor().long()  # 向上取整
        u = b.ceil().long()  # batch_size x atoms_num

        # When bj is exactly an integer, then bj.floor() == bj.ceil(), then u should +1. Eg: bj=1, l=1, u should = 2
        delta_m_l = (u + (l == u) - b) * next_dist 
        delta_m_u = (b - l) * next_dist # batch_size x atoms_num

        # Distribute probability with tensor operation. Much more faster than the For loop in the original paper.
        m.reshape(-1).index_add_(0, (l + self.offset).reshape(-1), delta_m_l.reshape(-1)) # dim index source
        m.reshape(-1).index_add_(0, (u + self.offset).reshape(-1), delta_m_u.reshape(-1))
        
        return m

class Agent:
    def __init__(self, obs_dim, action_dim, Qnet_lr ,device, trick = None ,batch_size = None):
        if trick['Categorical']:
            self.Qnet = Categorical(obs_dim, action_dim, trick = trick , batch_size = batch_size, device = device).to(device)
            self.projection_dist = self.Qnet.projection_dist
        else:
            if trick['Dueling']:
                self.Qnet = Dueling(obs_dim, action_dim, trick = trick ).to(device)
            else:
                self.Qnet = MLP(obs_dim, action_dim, trick = trick).to(device)
        self.Qnet_target = deepcopy(self.Qnet)

        self.Qnet_optimizer = torch.optim.Adam(self.Qnet.parameters(), lr = Qnet_lr)
    
    def update_Qnet(self, loss):
        self.Qnet_optimizer.zero_grad()
        loss.backward()
        self.Qnet_optimizer.step()

## 第二部分：定义DQN算法类
class DQN:
    def __init__(self, dim_info, is_continue, Qnet_lr, buffer_size, device , trick = None ,gamma = None , batch_size = None):
        obs_dim, action_dim = dim_info
        self.agent = Agent(obs_dim, action_dim, Qnet_lr, device = device ,trick = trick, batch_size = batch_size)
        if trick['PER'] and trick['N_Step']:
            self.buffer = N_Step_PER_Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = device, gamma = gamma)
        elif trick['PER']:
            self.buffer = PER_Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = device)
        elif trick['N_Step']:
            self.buffer = N_Step_Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = device, gamma = gamma)
        else:
            self.buffer = Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = device) #Buffer中说明了act_dim和action_dim的区别
        self.device = device
        self.is_continue = is_continue
        self.trick = trick

    def select_action(self, obs):
        '''
        输入的obs shape为(obs_dim) np.array reshape => (1,obs_dim) torch.tensor 
        若离散 则输出的action 为标量 <= [0]  (action_dim) np.array <= argmax (1,action_dim) torch.tensor 若keepdim=True则为(1,1) 否则为(1)
        若连续 则输出的action 为(1,action_dim) np.array <= (1,action_dim) torch.tensor
        is_continue 指定buffer和算法是否是连续动作 表示输出动作是否为连续动作
        '''
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
        if self.is_continue: # dqn 无此项
            action = self.agent.Qnet(obs).detach().cpu().numpy().squeeze(0) # 1xaction_dim -> action_dim
        else:
            if self.trick['Categorical']:
                action, _ = self.agent.Qnet(obs)
                action = action.detach().cpu().numpy()[0] # 1 []-> scalar
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
        if batch_size > total_size: # 防止batch_size比start_steps大, 一般可去掉
            batch_size = total_size
        
        if self.trick['PER']:
            indices, is_weight = self.buffer.sample(batch_size)
            obs, actions, rewards, next_obs, dones = self.buffer.buffer.sample(indices)
            return obs, actions, rewards, next_obs, dones, is_weight, indices
        else:
            indices = np.random.choice(total_size, batch_size, replace=False)  #默认True 重复采样 
            obs, actions, rewards, next_obs, dones = self.buffer.sample(indices)
            return obs, actions, rewards, next_obs, dones,
    

    ## DQN算法相关
    def learn(self,batch_size, gamma, tau):
        if self.trick['PER']: # 使用is_weight来矫正PER的采样偏差,indices用于更新优先级
            obs, actions, rewards, next_obs, dones, is_weight, indices = self.sample(batch_size)
        else:
            obs, actions, rewards, next_obs, dones = self.sample(batch_size)
        
        if self.trick['Categorical']:
            if self.trick['N_Step']:
                gamma = self.buffer.n_step_gamma
            m = self.agent.projection_dist(self.agent.Qnet,self.agent.Qnet_target,next_obs,rewards,dones ,gamma)
            _, dist = self.agent.Qnet(obs, actions) # batch_size x atoms_num

            if self.trick['PER']:
                error = (m * dist.clamp(1e-5, 1-1e-5).log()).sum(1)
                loss = -(m.detach() * dist.clamp(1e-5, 1-1e-5).log() * is_weight.reshape(-1,1)).sum(1).mean()
                self.buffer.update_priorities(indices, error.detach().cpu().numpy())
            else:
                ''' cross entropy loss'''
                loss =  -(m.detach() * dist.clamp(1e-5, 1-1e-5).log()).sum(1).mean() # clamp(1e-5, 1-1e-5) more stable
        else:
            '''  非分布型 Q网络 '''
            if self.trick['Double']: # 使用当前网络选择动作，使用目标网络评估Q值
                next_actions = self.agent.Qnet(next_obs).argmax(dim = 1).reshape(-1, 1) # batch_size x 1
                next_Q_target = self.agent.Qnet_target(next_obs).gather(dim = 1, index = next_actions.long()) # batch_size x 1
            else:
                next_Q_target = self.agent.Qnet_target(next_obs).max(dim = 1)[0].reshape(-1, 1) # batch_size x 1
            
            if self.trick['N_Step']:
                target_Q = rewards + self.buffer.n_step_gamma * next_Q_target * (1 - dones) # batch_size x 1
            else:
                target_Q = rewards + gamma * next_Q_target * (1 - dones) # batch_size x 1

            current_Q = self.agent.Qnet(obs).gather(dim =1, index =actions.long()) # batch_size x 1

            if self.trick['PER']:
                td_error = (current_Q -target_Q.detach())
                loss = (is_weight * (td_error ** 2)).mean() 
                self.buffer.update_priorities(indices, td_error.detach().cpu().numpy())
            else:
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
    def load(dim_info, is_continue,model_dir,trick = None):
        policy = DQN(dim_info,is_continue,0,0,device = torch.device("cpu"),trick = trick)
        policy.agent.Qnet.load_state_dict(torch.load(os.path.join(model_dir,"DQN.pt")))
        if trick['Noisy']:
            for module in policy.agent.Qnet.children(): # 只迭代返回子模块
                if isinstance(module,NoisyLinear): # 判断是否是某个类的实例
                    module.is_train = False
        return policy

### 第三部分：main函数
## 环境配置
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
        is_continue = True # 指定buffer和算法是否用于连续动作
        if is_dis_to_con :
            if action_dim == 1:
                dim_info = [obs_dim,16]  # 离散动作空间
                max_action = None
                is_continue = False
            else: # 多重连续动作空间->多重离散动作空间
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
    if trick is None or any(trick.values()) is False:
        prefix = policy_name + '_'
    elif trick['Double'] and trick['Dueling'] and trick['PER'] and trick['Noisy'] and trick['N_Step'] and trick['Categorical']:
        prefix = policy_name + '_Rainbow_' 
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
    parser.add_argument("--env_name", type = str,default="CartPole-v1") 
    parser.add_argument("--max_action", type=float, default=None)
    # 共有参数
    parser.add_argument("--seed", type=int, default=0) # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(500))
    parser.add_argument("--start_steps", type=int, default=500)
    parser.add_argument("--save_freq", type=int, default=int(500//4))
    parser.add_argument("--random_steps", type=int, default=0)  #可选择是否使用 dqn论文无此参数
    parser.add_argument("--learn_steps_interval", type=int, default=1)
    parser.add_argument("--is_dis_to_con", type=bool, default=True) # dqn 默认为True
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    ## DQN参数
    parser.add_argument("--Qnet_lr", type=float, default=1e-3) # 1e-3
    ## buffer参数   
    parser.add_argument("--buffer_size", type=int, default=1e6) #1e6默认是float,在bufffer中有int强制转换
    parser.add_argument("--batch_size", type=int, default=256)  #保证比start_steps小
    # DQN独有参数
    parser.add_argument("--epsilon", type=float, default=0.1)
    # trick参数
    parser.add_argument("--policy_name", type=str, default='DQN')                                
    parser.add_argument("--trick", type=dict, default={'Double':True,'Dueling':True,'PER':True,'Noisy':True,'N_Step':True,'Categorical':True})  
    # device参数
    parser.add_argument("--device", type=str, default='cpu') # cuda/cpu

    args = parser.parse_args()
    
    print(args)
    print('Algorithm:',args.policy_name)
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
    model_dir = make_dir(args.env_name,policy_name = args.policy_name ,trick = args.trick)
    writer = SummaryWriter(model_dir)

    ## device参数
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    
    ## 算法配置
    policy = DQN(dim_info, is_continue, Qnet_lr = args.Qnet_lr, buffer_size = args.buffer_size,device = device,trick= args.trick , gamma = args.gamma ,batch_size = args.batch_size)

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
    while episode_num < args.max_episodes:
        step +=1

        # 获取动作 区分动作action_为环境中的动作 action为要训练的动作
        if step < args.random_steps:
            action = env.action_space.sample()
        else:
            if args.trick['Noisy']:
                action = policy.select_action(obs)
            else:
                if np.random.rand() < args.epsilon:
                    action = np.random.randint(action_dim)
                else:
                    action = policy.select_action(obs)   
        action_ = action
        if args.is_dis_to_con and isinstance(env.action_space, gym.spaces.Box):
            action_ = dis_to_con(action, env, action_dim)
        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        done = terminated or truncated
        done_bool = terminated    ### truncated 为超过最大步数
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

    
    print('total_time:',time.time()-time_)
    policy.save(model_dir)
    ## 保存数据
    np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),np.array(train_return))




