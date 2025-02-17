import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
import numpy as np
from Buffer import MO_Buffer

import gymnasium as gym
import mo_gymnasium as mo_gym 
import argparse

## 其他
import re
import time
from torch.utils.tensorboard import SummaryWriter
from collections import deque

'''
一个深度强化学习算法分三个部分实现：
1.Agent类:包括actor、critic、target_actor、target_critic、actor_optimizer、critic_optimizer、
2.DQN算法类:包括select_action,learn、save、load等方法,为具体的算法细节实现
3.main函数:实例化DQN类,主要参数的设置,训练、测试、保存模型等
'''
'''ENVELOPE_MORL:论文链接：https://arxiv.org/abs/1908.08342 源代码:https://github.com/RunzheYang/MORL
多目标强化学习算法：pip install mo-gymnasium 
'''


'''  参数修改 改三处 1.MLP的hidden  2.main中args 3.dis_to_con中的离散转连续空间维度 '''
## 第一部分：定义Agent类
class MLP(nn.Module):
    def __init__(self, obs_dim, action_dim, reward_dim, hidden_1 = 256 ,hidden_2=256):
        super(MLP, self).__init__()
        self.action_dim = action_dim
        self.reward_dim = reward_dim

        self.l1 = nn.Linear(obs_dim + reward_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim * reward_dim)

    def forward(self, obs, preference):
        '''对比源代码
        1.减少了两层hidden层
        2.将hq的代码写到buffer的add中，增加理解性
        '''
        x = torch.cat((obs, preference), dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        q = self.l3(x)

        q = q.reshape(-1, self.action_dim, self.reward_dim) # batch_size x action_dim x reward_dim
        
        
        return q


class Agent:
    def __init__(self, obs_dim, action_dim, reward_dim ,Qnet_lr , device):

        self.Qnet = MLP(obs_dim, action_dim, reward_dim).to(device)
        self.Qnet_target = deepcopy(self.Qnet)

        self.Qnet_optimizer = torch.optim.Adam(self.Qnet.parameters(), lr = Qnet_lr)
    
    def update_Qnet(self, loss):
        self.Qnet_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.Qnet.parameters(), 1.0) # 梯度裁剪
        loss.backward()
        self.Qnet_optimizer.step()

## 第二部分：定义DQN算法类
class ENVELOPE:
    def __init__(self, dim_info, is_continue, Qnet_lr, buffer_size, device, beta , max_episodes, trick = None):
        obs_dim, action_dim , reward_dim = dim_info #
        self.agent = Agent(obs_dim, action_dim, reward_dim, Qnet_lr, device)
        self.buffer = MO_Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, preference_dim = reward_dim ,device = device) #Buffer中说明了act_dim和action_dim的区别
        self.device = device
        self.is_continue = is_continue

        self.reward_dim = reward_dim
        self.action_dim = action_dim    
        self.priority_mem = deque(maxlen=buffer_size) # 优先级buffer
        #self.w_kept = None # 保留的权重
        self.update_cnt = 0
        # 同伦参数
        self.homotopy  = True
        self.beta            = beta # 关于同伦的参数
        self.beta_init       = beta
        self.beta_uplim      = 1.00
        self.tau             = 1000.
        self.beta_expbase    = float(np.power(self.tau*(self.beta_uplim-self.beta), 1./max_episodes))
        self.beta_delta      = self.beta_expbase / self.tau




    def select_action(self, obs):
        '''
        输入的obs shape为(obs_dim) np.array reshape => (1,obs_dim) torch.tensor 
        若离散 则输出的action 为标量 <= [0]  (action_dim) np.array <= argmax (1,action_dim) torch.tensor 若keepdim=True则为(1,1) 否则为(1)
        若连续 则输出的action 为(1,action_dim) np.array <= (1,action_dim) torch.tensor
        is_continue 指定buffer和算法是否是连续动作 表示输出动作是否为连续动作
        注：与DQN区别：需要多增加一个随机的权重向量(preference) 用于计算q值
        '''
        preference = torch.randn(self.reward_dim).to(self.device)
        preference = (torch.abs(preference) / torch.norm(preference, p=1)) # 同样归一化

        preference = preference.reshape(1,-1)#.to(self.device)

        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
        if self.is_continue:  # dqn 均取离散动作
            print('ENVELOPE不适用于连续空间环境，请使用dis_to_cont方法')
            print('ENVELOPE is not suitable for continuous space environment, please use dis_to_cont method')
            #action = self.agent.Qnet(obs).detach().cpu().numpy().squeeze(0) # 1xaction_dim ->action_dim
        else:
            q = self.agent.Qnet(obs,preference)         # dqn:.argmax(dim = 1).detach().cpu().numpy()[0] # []标量
            q = q.reshape(-1,self.reward_dim) # action_dim x reward_dim
            q = q.data @ preference.reshape(-1,1) # action_dim 
            action = q.argmax().detach().cpu().numpy() # []标量
        return action
    
    def evaluate_action(self, obs, preference):
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
        preference = torch.as_tensor(preference,dtype=torch.float32).reshape(1, -1).to(self.device)

        q = self.agent.Qnet(obs,preference)         # dqn:.argmax(dim = 1).detach().cpu().numpy()[0] # []标量
        q = q.reshape(-1,self.reward_dim) # action_dim x reward_dim
        q = q.data @ preference.reshape(-1,1) # action_dim 
        action = q.argmax().detach().cpu().numpy() # []标量

        return action

    ## buffer相关
    def add(self, obs, action, reward, next_obs, done ,gamma ):
        ''' 这里done为terminal'''
        self.buffer.add(obs, action, reward, next_obs, done)
        ## 类似HER的做法
        '''
        总体做法：
        随机一个权重 W 容易得到  WQ (q选择当前动作的q值) :1 Wr:1
        WTnext_Q 得到所有动作的q值，选择最大的hq:1
        将优先级p = abs(Wr + gamma * hq - WQ)加入到优先级队列中
        '''
        '''
        使用这种方式：randn 返回均值为 0 且方差为 1 的正态分布
        preference = torch.randn(self.model_.reward_size)
        preference = (torch.abs(preference) / torch.norm(preference, p=1)).type(FloatTensor) # 同样归一化
        注意：不能使用以下这种方式：因为rand会返回 [0,1) 上均匀分布的随机数 会导致接近1附近的权重较小
        # preference = torch.rand(self.reward_dim).to(self.device) # 随机一个权重
        # preference = preference / preference.sum() # reward_dim
        '''
        with torch.no_grad():
            preference = torch.randn(self.reward_dim).to(self.device)
            preference = (torch.abs(preference) / torch.norm(preference, p=1)) # 同样归一化

            q = self.agent.Qnet(torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device),preference.reshape(1,-1)) # 1xaction_dim x reward_dim
            q = q[0,action].data # reward_dim 
            wq = preference.dot(q) # 1  # wTQ

            wr = preference.dot(torch.tensor(reward).to(self.device)) # 1 # wTr 加权之后的奖励
            if not done:
                next_q = self.agent.Qnet(torch.as_tensor(next_obs,dtype=torch.float32).reshape(1, -1).to(self.device),preference.reshape(1,-1)) # 1xaction_dim x reward_dim
                reQ_ext = next_q.squeeze(0) # action_dim x reward_dim
                w_ext = preference.reshape(1,-1,1).repeat(1,self.action_dim,1) # 1 x (action_dim x reward_dim) x 1
                w_ext = w_ext.reshape(-1,self.reward_dim) # action_dim x reward_dim
                prod = torch.bmm(reQ_ext.unsqueeze(1), w_ext.unsqueeze(2)).squeeze() # action_dim x 1 x reward_dim * action_dim x reward_dim x 1 -> action_dim x 1 x 1 -> action_dim
                
                prod = prod.reshape(1,-1) # 1 x action_dim
                inds = torch.argmax(prod) # 1
                hq = reQ_ext[inds] # reward_dim 
                whq = preference.dot(hq) # 1
                p = abs(wr + gamma * whq - wq)
            else:
                #self.w_kept = None
                # if self.epsilon_decay: # 暂不需要 已经实现
                #     self.epsilon -= self.epsilon_delta
                if self.homotopy:
                    self.beta += self.beta_delta
                    self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta
                p = abs(wr - wq)
            p += 1e-5

            self.priority_mem.append(p.detach().cpu().numpy())

    
    def sample(self, batch_size):
        total_size = len(self.buffer)
        batch_size = min(total_size, batch_size) # 防止batch_size比start_steps大, 一般可去掉
        priority_mem = np.array(self.priority_mem)
        indices = np.random.choice(range(total_size), batch_size, replace=False, p=priority_mem / priority_mem.sum() )
       
        #indices = np.random.choice(total_size, batch_size, replace=False)  #默认True 重复采样 
        obs, actions, rewards, next_obs, dones = self.buffer.sample(indices)

        return obs, actions, rewards, next_obs, dones
    

    ## ENVELOPE算法相关
    def learn(self,batch_size, gamma, tau, weight_num ,update_freq):
        '''
        公式: target_Q = rewards + gamma * max_a(Q_target(next_obs, a)) * (1 - dones)
        '''
        obs, actions, rewards, next_obs, dones = self.sample(batch_size) 

        # batch_size x obs_dim -> (batch_size x weight_num) X obs_dim
        obs = obs.repeat(weight_num,1)
        actions = actions.repeat(weight_num,1) # batch_size x 1 -> (batch_size x weight_num) x 1
        rewards = rewards.repeat(weight_num,1) # batch_size x reward_dim -> (batch_size x weight_num) x reward_dim
        next_obs = next_obs.repeat(weight_num,1) # batch_size x obs_dim -> (batch_size x weight_num) x obs_dim
        dones = dones.repeat(weight_num,1) # batch_size x 1 -> (batch_size x weight_num) x 1

        # 随机weight_num个权重
        # w_batch=  torch.randn(weight_num, self.reward_dim).to(self.device) # weight_num x reward_dim
        # w_batch = (torch.abs(w_batch) / torch.norm(w_batch, p=1))
        # w_batch = w_batch.repeat(batch_size,1) # weight_num x reward_dim -> (batch_size x weight_num) x reward_dim
        w_batch = np.random.randn(weight_num, self.reward_dim) # 多样性 每个样本拿到不同的权重
        w_batch = np.abs(w_batch) / np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)  # L1范数
        w_batch = torch.as_tensor(w_batch.repeat(batch_size, axis=0),dtype=torch.float32).to(self.device) # weight_num x reward_dim -> (batch_size x weight_num) x reward_dim
        
        # 与add类似 拓展维度
        w_ext = w_batch.unsqueeze(2).repeat(1, self.action_dim, 1)  # (batch_size x weight_num) x (action_dim x reward_dim) x 1
        w_ext = w_ext.reshape(-1, self.reward_dim) # (batch_size x weight_num x action_dim) x reward_dim

        #current_Q = self.agent.Qnet(obs,w_batch) #.gather(dim =1, index =actions.long()) # batch_size x action_dim -> batch_size x 1

        # 相当于double dqn # 当前网路选择动作，目标网络计算Q值
        tmpQ = self.agent.Qnet(next_obs,w_batch) # (batch_size x weight_num) x action_dim x reward_dim
        tmpQ = tmpQ.reshape(-1,self.reward_dim) # (batch_size x weight_num x action_dim) x reward_dim
        next_actions = torch.bmm(tmpQ.unsqueeze(1), w_ext.unsqueeze(2)).reshape(-1,self.action_dim).max(1)[1] # (batch_size x weight_num x action_dim) ->... -> batch_size x weight_num
        next_Q = self.agent.Qnet_target(next_obs,w_batch) # (batch_size x weight_num) x action_dim x reward_dim
        ## next_Q_target :HQ  next_Q:DQ
        next_Q_target = next_Q.gather(dim =1, index =next_actions.long().reshape(-1,1,1).expand(-1,1,self.reward_dim)).squeeze(1) # (batch_size x weight_num) x reward_dim
        

        target_Q = rewards + gamma * next_Q_target * (1 - dones) # (batch_size x weight_num) x reward_dim

        current_Q = self.agent.Qnet(obs,w_batch).gather(dim =1, index =actions.long().reshape(-1,1,1).expand(-1,1,self.reward_dim)).squeeze(1)  # (batch_size x weight_num) x reward_dim
        #print('current_Q:',current_Q.shape)   
        ## 辅助损失
        wQ = torch.bmm(current_Q.unsqueeze(1), w_batch.unsqueeze(2)).squeeze() # (batch_size x weight_num) 
        WTQ = torch.bmm(target_Q.unsqueeze(1), w_batch.unsqueeze(2)).squeeze() # (batch_size x weight_num) 
        
        #print(self.beta)
        self.loss = self.beta * F.mse_loss(wQ, WTQ.detach()) + (1-self.beta) * F.mse_loss(current_Q, target_Q.detach()) # 标量值
        
        self.agent.update_Qnet(self.loss)

        # self.update_cnt += 1
        # if self.update_cnt % update_freq == 0:
        self.update_target(tau)

    def update_target(self, tau):
        '''
        更新目标网络参数: θ_target = τ*θ_local + (1 - τ)*θ_target
        切断自举,缓解高估Q值 source -> target
        '''
        self.soft_update(self.agent.Qnet_target, self.agent.Qnet, tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)



    ## 保存模型
    def save(self, model_dir):
        torch.save(self.agent.Qnet.state_dict(), os.path.join(model_dir,"ENVELOPE_DQN.pt"))
    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue ,model_dir, trick = None):
        policy = ENVELOPE(dim_info,is_continue,0,0,device = torch.device("cpu"),trick = trick, beta=0,max_episodes=1)
        policy.agent.Qnet.load_state_dict(torch.load(os.path.join(model_dir,"ENVELOPE_DQN.pt")))
        return policy

### 第三部分：main函数
## 环境配置
def get_env(env_name,is_dis_to_con = False):
    env = mo_gym.make(env_name)
    # 增加 reward_dim
    # env.reset()
    # next_obs, vector_reward, terminated, truncated, info = env.step(env.action_space.sample())
    # reward_dim = len(vector_reward)

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
                is_continue = False
            else: # 多重连续动作空间->多重离散动作空间 例：BipedalWalker-v3 连续动作空间4维
                '''
                2**action_dim ->将每个维度分成2个离散动作 这个2这个系数可以调节。
                例:BipedalWalker-v3 4维分成了离散的16维 这个环境下3,4会造成维度爆炸，效果不太好。
                '''
                dim_info = [obs_dim,2**action_dim]  # 离散动作空间 
                is_continue = False
    else:
        action_dim = env.action_space.n
        dim_info = [obs_dim,action_dim]
        max_action = None
        is_continue = False
    
    # 增加reward_dim
    dim_info.append(env.unwrapped.reward_dim)

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
    pattern = re.compile(f'^{prefix}\d+') # ^ 表示开头，\d 表示数字，+表示至少一个
    existing_dirs = [d for d in os.listdir(env_dir) if pattern.match(d)]
    max_number = 0 if not existing_dirs else max([int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()])
    model_dir = os.path.join(env_dir, prefix + str(max_number + 1))
    os.makedirs(model_dir)
    return model_dir

## dis_to_cont
def dis_to_con(discrete_action, env, action_dim):  # 离散动作转连续的函数
    if env.action_space.shape[0] == 1: # 连续动作空间是一维
        action_lowbound = env.action_space.low[0]  # 连续动作的最小值
        action_upbound = env.action_space.high[0]  # 连续动作的最大值
        return np.array([action_lowbound + (discrete_action /(action_dim - 1)) * (action_upbound -action_lowbound)])
    else: # 连续动作空间是多维
        action_lowbound = env.action_space.low
        action_upbound = env.action_space.high
        action_dim_per = int(action_dim ** (1 / len(action_lowbound))) ## -> 反求出1维的动作空间用几个离散动作表示
        shape = env.action_space.shape[0]
        '''
        例：BipedalWalker-v3 连续动作空间4维 Box(-1.0, 1.0, (4,), float32) -> 2**4 = 64维
        action = 0 -> discrete_indices = [0,0,0,0] -> continue_action = [-1,-1,-1,-1]
        action = 1 -> discrete_indices = [1,0,0,0] -> continue_action = [-1,-1,-1,-1]
        action = 2 -> discrete_indices = [0,1,0,0] -> continue_action = [-1, 1, -1, -1]
        action = 3 -> discrete_indices = [1,1,0,0] -> continue_action = [1, 1, -1, -1]
        action = 4 -> discrete_indices = [0,0,1,0] -> continue_action = [-1, -1, 1, -1]
        ... action_dim_per=2时 类似于用2进制表示 
        action = 63 -> discrete_indices = [1,1,1,1] -> continue_action = [1,1,1,1]
        '''
        discrete_indices = [discrete_action // (action_dim_per ** i) % (action_dim_per) for i in range(shape)]  # 这里存储的是每个维度的离散索引 
        continue_action = [action_lowbound[i] + discrete_indices[i] / (action_dim_per - 1) * (action_upbound[i] - action_lowbound[i]) for i in range(shape)]
        return np.array(continue_action)

def eval_model(policy, env, w, seed = 0):
    ## 评估一次
    obs_e,info = env.reset(seed = seed)
    episode_reward = 0
    done_e = False
    #reward_vec = np.array([0,0])
    while not done_e:
        action = policy.evaluate_action(obs_e,w)
        action_ = action

        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        done_e = terminated or truncated
        #reward_vec = np.add(reward_vec,reward)
        episode_reward += np.dot(reward, w)
        obs_e = next_obs
    ## 评估一次
    return episode_reward
''' 

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="deep-sea-treasure-v0") 
    # 共有参数
    parser.add_argument("--seed", type=int, default=0) # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(5000))
    parser.add_argument("--save_freq", type=int, default=int(500//4))
    parser.add_argument("--start_steps", type=int, default=500)
    parser.add_argument("--random_steps", type=int, default=0)  #可选择是否使用 dqn论文无此参数
    parser.add_argument("--learn_steps_interval", type=int, default=1)
    parser.add_argument("--is_dis_to_con", type=bool, default=True) # dqn 默认为True
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01) # 0.01 注意这里是1 不是0.01
    ## DQN参数
    parser.add_argument("--Qnet_lr", type=float, default=1e-3)
    ## buffer参数   
    parser.add_argument("--buffer_size", type=int, default=int(1000000)) #1e6默认是float,在bufffer中有int强制转换
    parser.add_argument("--batch_size", type=int, default=256)  #保证比start_steps小
    # ENVELOPE独有参数
    parser.add_argument("--epsilon", type=float, default=0.4) # 0.2
    parser.add_argument("--epsilon_decay", type=bool, default=True)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--weight_num", type=int, default=128) #64 #32
    parser.add_argument("--update_freq", type=int, default=1)

    # trick参数
    parser.add_argument("--policy_name", type=str, default='ENVELOPE_DQN')
    parser.add_argument("--trick", type=dict, default=None) # 无trick
    #parser.add_argument("--trick", type=dict, default={'Double':False,'Dueling':False,'PER':False,'Noisy':False,'N_Step':False,'Categorical':False})    
    # device参数
    parser.add_argument("--device", type=str, default='cuda')
    # 评估参数
    parser.add_argument("--evaluate_interval", type=float, default=10)
    
    args = parser.parse_args()
    print(args)
    print('-'*50)
    print('Algorithm:',args.policy_name)
    ## 环境配置
    env,dim_info,max_action,is_continue = get_env(args.env_name,args.is_dis_to_con)
    action_dim = dim_info[1]
    print(f'Env:{args.env_name}  obs_dim:{dim_info[0]}  action_dim:{dim_info[1]}  max_action:{max_action}  max_episodes:{args.max_episodes}')
    if args.epsilon_decay:
        epsilon_delta = (args.epsilon - 0.05) / args.max_episodes

    ## 随机数种子(cpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ### cuda
    torch.cuda.manual_seed(args.seed) # 经过测试,使用cuda时,只加这句就能保证两次结果一致
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
    policy = ENVELOPE(dim_info, is_continue, Qnet_lr = args.Qnet_lr, buffer_size = args.buffer_size,device = device, beta = args.beta, max_episodes = args.max_episodes,)

    env_spec = gym.spec(args.env_name)
    print('reward_threshold:',env_spec.reward_threshold if env_spec.reward_threshold else 'No Threshold = Higher is better')
    time_ = time.time()
    ## 训练
    if args.env_name == 'deep-sea-treasure-v0':
        w_set_1 = [0.5,0.5] # 两个权重  treasure_value, time_penalty
        w_set_2 = [0.8,0.2]
        w_eval = w_set_2
    
    episode_num = 0
    step = 0
    episode_reward_1 = 0
    episode_reward_2 = 0
    loss = 0

    train_return = []
    obs,info = env.reset(seed=args.seed)  # 针对obs复现
    env.action_space.seed(seed=args.seed) if args.random_steps > 0 else None # 针对action复现:env.action_space.sample()
    while episode_num < args.max_episodes:
        step +=1
        # 获取动作 区分动作action_为环境中的动作 action为要训练的动作
        if step < args.random_steps:
            action = env.action_space.sample()
            if max_action is not None: # 若环境连续则将环境连续动作转为离散
                action = action / max_action
                ## con -> dis 使得random_steps 可用
                boundaries = np.linspace(-1, 1, action_dim + 1)
                # 使用 np.digitize 找到连续动作所在的区间 速度优于for循环
                discrete_action = np.digitize(action[0], boundaries) - 1
                action = np.clip(discrete_action, 0, action_dim - 1)
        else:
            if np.random.rand() < args.epsilon:
                action = np.random.randint(action_dim)
            else:
                action = policy.select_action(obs)   
        # 此时输出action为离散动作
        action_ = action            
        if args.is_dis_to_con and isinstance(env.action_space, gym.spaces.Box): # = .... and max_action is not None:  #如果环境连续，且算法（用于离散）使用离散转连续域技巧 
            action_ = dis_to_con(action_, env, action_dim)
        #print('action:',action_)
        # 探索环境
        next_obs, reward, terminated, truncated, infos = env.step(action_) 

        done = terminated or truncated
        done_bool = terminated     ### truncated 为超过最大步数
        #print('obs:',obs)
        policy.add(obs, action, reward, next_obs, done_bool, args.gamma)
        #print('reward:',reward)
         # 两个权重  treasure_value, time_penalty
        episode_reward_1 += reward.dot(w_set_1)
        episode_reward_2 += reward.dot(w_set_2)

        obs = next_obs
        # episode 结束
        if done:
            ## 保存
            if (episode_num ) % args.save_freq == 0:
                policy.save(model_dir)

            ## 评估模型 之后需要env_reset
            if episode_num % args.evaluate_interval == 0:
                #policy_e = policy.load(dim_info, is_continue ,model_dir)
                episode_reward = eval_model(policy, env, w = w_eval, seed = args.seed)
                writer.add_scalar('eval_reward', episode_reward, episode_num )

            ## 显示
            if  (episode_num + 1) % 100 == 0:
                print("episode: {}, reward: {}".format(episode_num + 1, episode_reward_1))

            ## decay epsilon
            if args.epsilon_decay:
                args.epsilon -= epsilon_delta

            writer.add_scalar('reward', episode_reward_1, episode_num + 1)
            writer.add_scalar('reward_2', episode_reward_2, episode_num + 1)
            writer.add_scalar('loss', loss, episode_num + 1)

            train_return.append(episode_reward_1)

            episode_num += 1
            obs,info = env.reset(seed=args.seed)
            episode_reward_1 = 0
            episode_reward_2 = 0
            loss = 0


                
        
        # 满足step,更新网络
        if step > args.start_steps and step % args.learn_steps_interval == 0:
            policy.learn(args.batch_size, args.gamma, args.tau, args.weight_num ,args.update_freq)
            loss += policy.loss.item()
        
        # # 保存模型
        # if episode_num % args.save_freq == 0:
        #     policy.save(model_dir)
        



    print('total_time:',time.time()-time_)
    policy.save(model_dir)
    ## 保存数据
    np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),np.array(train_return))




