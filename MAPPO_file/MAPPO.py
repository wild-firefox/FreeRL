import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal,Categorical

import numpy as np
from Buffer import Buffer_for_PPO

from copy import deepcopy
import pettingzoo #动态导入
import gymnasium as gym
import importlib
import argparse

## tricks
from normalization import Normalization,RewardScaling

## 其他
from torch.utils.tensorboard import SummaryWriter
import time
import re
import pickle 

''' mappo:论文链接：https://arxiv.org/pdf/2103.01955 代码链接：https://github.com/marlbenchmark/on-policy/
提出的建议
1.使用值value归一化来稳定价值函数。（注：不是状态归一，技巧为PopArt，(Preserving Outputs Precisely, while Adaptively Rescaling Targets) 来自文献Learning values across many orders of magnitude:https://arxiv.org/pdf/1602.07714  / 拓展：Multi-task Deep Reinforcement Learning with popart）
2.如果可以，使用必要的全局状态和本地状态。
3.在困难环境中最多使用10个epoch(10 or 5)，在简单环境中最多使用15个epoch;避免分成多个batch(1个效果最好)（PPO原论文中是分成horizon//mini_batch_size个)
4.为获得最高PPO性能，将clip值在0.2以下范围内调整。（论文中0.2和0.05效果还行）
5.使用一个较大的batch_size(即horizon值)以达到最好的效果，然后调整它来优化效果。

论文代码中使用的PPO技巧trick: 参考知乎:https://zhuanlan.zhihu.com/p/386559032
1.Generalized Advantage Estimation：这个技巧来自文献：Hign-dimensional continuous control using generalized advantage estimation。
2.Input Normalization
3.Value Clipping：与策略截断类似，将值函数进行一个截断。
4.Relu activation with Orthogonal Initialization： 论文：https://arxiv.org/pdf/2004.05867v2
5.Gredient Clipping：梯度更新不要太大。
6.Layer Normalization：## 来自同名论文 ###后面文献并无提及LayerNorm 怀疑作者贴错了  这个技巧来自文献：Regularization matters in policy optimization-an empirical study on continuous control。
7.Soft Trust-Region Penalty：## 这个技巧来自文件：Revisiting design choices in proximal policy optimization。
## 还有一些其他技巧 知乎上没提到但是论文和代码上有,补充如下
8.huber_loss: 用于替代mse_loss，减少outlier的影响。
9.reward normalization: 对reward进行归一化。
10.use_feature_normalization :类似于Layer Normalization,在特征输入前采取的归一化操作
11.adm_eps: 与PPO_with_tricks一致
12.lr_decay: 与PPO_with_tricks一致
'''

## 先写一个简单的continue环境实现：
'''
关于建议
1.由于在ddpg中发现建议1中的popart效果并不理想，所以这里不使用popart
2.知乎上的评论：论证了建议2中可以不用特别加入agent-specific information,所以按一般MA拼接状态和动作处理，PPO无动作 只拼接状态
3.设置epocch = 5 ; minibatch_size = 1 (即:不再进行对batch_size进行小批量,在我这相当于该值与horizon相等 )#后者我做过实验，效果确实会好一点点,见PPO_file中note
4.设置clip = 0.2
5.和原ppo一样看情况调整horizon

综上:关注3，4即可

关于args.trick:  trick为PPO_file中PPO_with_tricks.py中已实现过的trick
1.使用GAE 与原PPO一致
2.Input Normalization 即trick中的ObsNorm 确实重要
3.Value Clipping 新增
4.Orthogonal Initialization 即trick中的orthogonal_init
5.Gredient Clipping PPO中默认加入以防止梯度爆炸和消失 
6.Layer Normalization 新增
7.Soft Trust-Region Penalty 新增 ## 在trpo用到，在PPO没有用到
8.huber_loss 新增
9.reward normalization 即trick中reward_norm or reward_scaling
10.use_feature_normalization 新增
11.adm_eps 即trick中的adam_eps
12.lr_decay 即trick中的lr_decay

# 综上：重点关注3,6,7 (实际加入2,3,4,6,7,8,9,10,11 trick) 1 5 默认加入
综上：PPO_with_tricks.py中的7个trick全部加入了,(其中tanh的trick集成在net_init中,且默认使用relu,故上述中没有提及)
重点关注3,6,7,8,10

另外:
mappo与rmappo区别见原代码:https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/scripts/train/train_mpe.py#L68
区别：rmappo使用了RNN
原代码中shared文件夹和separated文件夹的区别:
shared文件夹：使用一个共享的actor-critic网络
separated文件夹：每个agent使用一个独立的actor-critic网络 
这里是实现的separated的形式

注：这里MAPPO 为复刻的论文代码
MAPPO_simple为不加任何trick的代码

疑问：
从代码：https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/runner/separated/mpe_runner.py#L126
（action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)）
及 https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/envs/mpe/environment.py#L206
（if action[0] == 1:）
这里两处矛盾 推出原论文作者并没有使用mpe的离散环境做实验，
且此代码实现效果和https://blog.csdn.net/onlyyyyyyee/article/details/139331501类似 所以推测复现代码实现并无错误。  
## 已解决：2025.4.17 已解决 原论文应该是支持的，此代码已支持离散和连续环境。

此代码的log_std 使用对角高斯函数的效果好，此与MASAC不同
'''


## 第一部分：定义Agent类
def net_init(m,gain=None,use_relu = True):
    '''网络初始化
    m:layer = nn.Linear(3, 2) # 按ctrl点击Linear 可知默认初始化为 nn.init.kaiming_uniform_(self.weight) ,nn.init.uniform_(self.bias) 此初始化的推荐的非线性激活函数方式为relu,和leaky_relu)
    参考2：Orthogonal Initialization trick:（原代码也是如此）
    critic: gain :nn.init.calculate_gain(['tanh', 'relu'][use_ReLU]) ; weight: nn.init.orthogonal_(self.weight, gain) ; bias: nn.init.constant_(self.bias, 0)
    actor: 其余层和critic一样，最后输出层gain = 0.01
    参考：
    1.https://zhuanlan.zhihu.com/p/210137182， -> orthogonal_ 优于 xavier_uniform_
    2.https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/  -> kaiming_uniform_ 替代 xavier_uniform_
    代码参考 原论文代码：https://github.com/marlbenchmark/on-policy/
    '''
    use_orthogonal = True # -> 1
    use_relu = use_relu

    init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_]
    activate_fuction = ['tanh','relu', 'leaky_relu']  # relu 和 leaky_relu 的gain值一样
    gain = gain if gain is not None else  nn.init.calculate_gain(activate_fuction[use_relu]) # 根据的激活函数设置
    
    init_method[use_orthogonal](m.weight, gain=gain)
    nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128,trick = None):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.mean_layer = nn.Linear(hidden_2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim)) # 与PPO.py的方法一致：对角高斯函数
        #self.log_std_layer = nn.Linear(hidden_2, action_dim) # 式2

        self.trick = trick
        # 使用 orthogonal_init
        if trick['orthogonal_init']:
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.mean_layer, gain=0.01)   

    def forward(self, x, ):
        if self.trick['feature_norm']:
            x = F.layer_norm(x, x.size()[1:])
        x = F.relu(self.l1(x))
        if self.trick['LayerNorm']:
            x = F.layer_norm(x, x.size()[1:])
        x = F.relu(self.l2(x))
        if self.trick['LayerNorm']:
            x = F.layer_norm(x, x.size()[1:])

        mean = torch.tanh(self.mean_layer(x))  # 使得mean在-1,1之间

        log_std = self.log_std.expand_as(mean)  # 使得log_std与mean维度相同 输出log_std以确保std=exp(log_std)>0
        #log_std = self.log_std_layer(x) # 式2
        log_std = torch.clamp(log_std, -20, 2) # exp(-20) - exp(2) 等于 2e-9 - 7.4，确保std在合理范围内
        std = torch.exp(log_std)

        return mean, std    
    
class Actor_discrete(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128,trick = None):
        super(Actor_discrete, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim)

        self.trick = trick
        # 使用 orthogonal_init
        if trick['orthogonal_init']:
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.l3, gain=0.01) 

    def forward(self, obs ):
        if self.trick['feature_norm']:
            obs = F.layer_norm(obs, obs.size()[1:])
        x = F.relu(self.l1(obs))
        if self.trick['LayerNorm']:
            x = F.layer_norm(x, x.size()[1:])
        x = F.relu(self.l2(x))
        if self.trick['LayerNorm']:
            x = F.layer_norm(x, x.size()[1:])
        a_prob = torch.softmax(self.l3(x), dim=1)
        return a_prob
        
class Critic(nn.Module):
    def __init__(self, dim_info:dict[str,list], hidden_1=128 , hidden_2=128,trick = None):
        super(Critic, self).__init__()
        global_obs_dim = sum(val[0] for val in dim_info.values())  
        
        self.l1 = nn.Linear(global_obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)
        
        self.trick = trick
        # 使用 orthogonal_init
        if trick['orthogonal_init']:
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.l3)  
        
    def forward(self, s): # 传入全局观测和动作
        s = torch.cat(list(s), dim = 1)
        #sa = torch.cat([s,a], dim = 1)
        if self.trick['feature_norm']:
            s = F.layer_norm(s, s.size()[1:])
        
        q = F.relu(self.l1(s))
        if self.trick['LayerNorm']:
            q = F.layer_norm(q, q.size()[1:])
        q = F.relu(self.l2(q))
        if self.trick['LayerNorm']:
            q = F.layer_norm(q, q.size()[1:])
        q = self.l3(q)

        return q
    
class Agent:
    def __init__(self, obs_dim, action_dim, dim_info,actor_lr, critic_lr, is_continue, device,trick):   
        
        if is_continue:
            self.actor = Actor(obs_dim, action_dim,trick=trick ).to(device)
        else:
            self.actor = Actor_discrete(obs_dim, action_dim, trick=trick).to(device)
        self.critic = Critic( dim_info ,trick=trick).to(device)

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr= actor_lr)

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

    def update_ac(self, loss):
        self.ac_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10)
        self.ac_optimizer.step()

## 第二部分：定义DQN算法类
def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

class MAPPO: 
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, horizon, device, trick = None):        
        self.agents  = {}
        self.buffers = {}
        for agent_id, (obs_dim, action_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, action_dim, dim_info, actor_lr, critic_lr, is_continue, device,trick)
            self.buffers[agent_id] = Buffer_for_PPO(horizon, obs_dim, act_dim = action_dim if is_continue else 1, device = device)

        self.device = device
        self.is_continue = is_continue
        print('actor_type:continue') if self.is_continue else print('actor_type:discrete')

        self.horizon = int(horizon)

        self.trick = trick
        self.num_agents = len(self.agents) # 智能体数量

        if self.trick['lr_decay']:
            self.actor_lr = actor_lr
            self.critic_lr = critic_lr


    def select_action(self, obs):
        actions = {}
        action_log_pis = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue: 
                mean, std = self.agents[agent_id].actor(obs)
                dist = Normal(mean, std)
                action = dist.sample()
                action_log_pi = dist.log_prob(action) # 1xaction_dim
            else:
                dist = Categorical(probs=self.agents[agent_id].actor(obs))
                action = dist.sample()
                action_log_pi = dist.log_prob(action)
            # to 真实值
            actions[agent_id] = action.detach().cpu().numpy().squeeze(0) # 1xaction_dim ->action_dim
            action_log_pis[agent_id] = action_log_pi.detach().cpu().numpy().squeeze(0)

        return actions , action_log_pis
    
    def evaluate_action(self,obs):
        actions = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue: 
                mean, _ = self.agents[agent_id].actor(obs)
                action = mean.detach().cpu().numpy().squeeze(0)
            else:
                a_prob = self.agents[agent_id].actor(obs).detach().cpu().numpy().squeeze(0)
                action = np.argmax(a_prob)

            actions[agent_id] = action
        return actions
    
    ## buffer 相关
    def add(self, obs, action, reward, next_obs, done, action_log_pi , adv_dones):
        for agent_id, buffer in self.buffers.items():
            buffer.add(obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id], action_log_pi[agent_id] , adv_dones[agent_id])

    def all(self):
        obs = {}
        action = {}
        reward = {}
        next_obs = {}
        done = {}
        action_log_pi = {}
        adv_dones = {}
        for agent_id, buffer in self.buffers.items():
            obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id], action_log_pi[agent_id], adv_dones[agent_id] = buffer.all()
        return obs, action, reward, next_obs, done, action_log_pi, adv_dones

    ## PPO算法相关
    def learn(self, minibatch_size, gamma, lmbda ,clip_param, K_epochs, entropy_coefficient, huber_delta = None):
        # 多智能体特有-- 集中式训练critic:要用到所有智能体next状态和动作
        obs, action, reward, next_obs, done , action_log_pi , adv_dones = self.all()
        # 计算GAE
        with torch.no_grad():  # adv and v_target have no gradient
            adv = torch.zeros(self.horizon, self.num_agents)
            gae = 0
            vs = []
            vs_ = []
            for agent_id  in self.buffers.keys():
                vs.append(self.agents[agent_id].critic(obs.values()))  # batch_size x 1
                vs_.append(self.agents[agent_id].critic(next_obs.values()))
            
            vs = torch.cat(vs, dim = 1) # batch_size x 3
            vs_ = torch.cat(vs_, dim = 1) # batch_size x 3

            reward = torch.cat(list(reward.values()), dim = 1) # 
            done = torch.cat(list(done.values()), dim = 1)
            adv_dones = torch.cat(list(adv_dones.values()), dim = 1)

            td_delta = reward + gamma * (1.0 - done) * vs_ - vs  #这里可能使用全局的reward
            
            for i in reversed(range(self.horizon)):
                gae = td_delta[i] + gamma * lmbda * gae * (1.0 - adv_dones[i])
                adv[i] = gae

            adv = adv.to(self.device)  # batch_size x 3
            v_target = adv + vs  # batch_size x 3
            if self.trick['adv_norm']:  
                adv = ((adv - adv.mean()) / (adv.std() + 1e-8)) 

        for agent_id, agent in self.agents.items():
            # Optimize policy for K epochs:
            for _ in range(K_epochs): 
                # 随机打乱样本 并 生成小批量
                shuffled_indices = np.random.permutation(self.horizon)
                indexes = [shuffled_indices[i:i + minibatch_size] for i in range(0, self.horizon, minibatch_size)]
                for index in indexes:
                    # 先更新actor
                    if self.is_continue:
                        mean, std = self.agents[agent_id].actor(obs[agent_id][index])
                        dist_now = Normal(mean, std)
                        dist_entropy = dist_now.entropy().sum(dim = 1, keepdim=True)  # mini_batch_size x action_dim -> mini_batch_size x 1
                        action_log_pi_now = dist_now.log_prob(action[agent_id][index]) # mini_batch_size x action_dim
                    else:
                        dist_now = Categorical(probs=self.agents[agent_id].actor(obs[agent_id][index]))
                        dist_entropy = dist_now.entropy().reshape(-1,1) # mini_batch_size  -> mini_batch_size x 1
                        action_log_pi_now = dist_now.log_prob(action[agent_id][index].reshape(-1)).reshape(-1,1) # mini_batch_size  -> mini_batch_size x 1

                    ratios = torch.exp(action_log_pi_now.sum(dim = 1, keepdim=True) - action_log_pi[agent_id][index].sum(dim = 1, keepdim=True))  # shape(mini_batch_size X 1)
                    surr1 = ratios * adv[index]   # mini_batch_size x 1
                    surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * adv[index]  
                    actor_loss = -torch.min(surr1, surr2).mean() - entropy_coefficient * dist_entropy.mean()
                    agent.update_actor(actor_loss)

                    # 再更新critic
                    obs_ = {agent_id: obs[agent_id][index] for agent_id in obs.keys()}

                    v_s = self.agents[agent_id].critic(obs_.values()) # mini_batch_size x 1
                    v_s = v_s.repeat(1,self.num_agents) # mini_batch_size x 3

                    v_target_ = v_target[index]
                    if self.trick['ValueClip']:
                        ''' 参考原mappo代码,原代码存储了return和value值,故实现上和如下有些许差异'''
                        v_target_clip = torch.clamp(v_target_, v_s - clip_param, v_s + clip_param)
                        if self.trick['huber_loss']:
                            critic_loss_clip = huber_loss(v_target_clip-v_s,huber_delta).mean()
                            critic_loss_original = huber_loss(v_target_-v_s,huber_delta).mean()
                        else:
                            critic_loss_clip = F.mse_loss(v_target_clip, v_s)
                            critic_loss_original = F.mse_loss(v_target_, v_s)
                        critic_loss = torch.max(critic_loss_original,critic_loss_clip)
                    else:
                        if self.trick['huber_loss']:
                            critic_loss = huber_loss(v_target_-v_s,huber_delta).mean()
                        else:
                            critic_loss = F.mse_loss(v_target_, v_s)
                    agent.update_critic(critic_loss)
        

        ## 清空buffer
        for buffer in self.buffers.values():
            buffer.clear()
    
    def lr_decay(self,episode_num,max_episodes):
        lr_a_now = self.actor_lr * (1 - episode_num / max_episodes)
        lr_c_now = self.critic_lr * (1 - episode_num / max_episodes)
        for agent in self.agents.values():
            for p in agent.actor_optimizer.param_groups:
                p['lr'] = lr_a_now
            for p in agent.critic_optimizer.param_groups:
                p['lr'] = lr_c_now
 

    def save(self, model_dir):
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            os.path.join(model_dir, 'MAPPO.pth')
        )
        
    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue, model_dir,trick=None):
        policy = MAPPO(dim_info, is_continue = is_continue, actor_lr = 0, critic_lr = 0, horizon = 0, device = 'cpu',trick=trick)
        data = torch.load(os.path.join(model_dir, 'MAPPO.pth'))
        for agent_id, agent in policy.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return policy


## 第三部分 mian 函数
## 环境配置
def get_env(env_name,env_agent_n = None,continuous_actions=True):
    # 动态导入环境
    module = importlib.import_module(f'pettingzoo.mpe.{env_name}')
    print('env_agent_n or num_good:',env_agent_n) 
    if env_agent_n is None: #默认环境
        env = module.parallel_env(max_cycles=25, continuous_actions=continuous_actions)
    elif env_name == 'simple_spread_v3' or 'simple_adversary_v3': 
        env = module.parallel_env(max_cycles=25, continuous_actions=continuous_actions, N = env_agent_n)
    elif env_name == 'simple_tag_v3': 
        env = module.parallel_env(max_cycles=25, continuous_actions=continuous_actions, num_good= env_agent_n, num_adversaries=3)
    elif env_name == 'simple_world_comm_v3':
        env = module.parallel_env(max_cycles=25, continuous_actions=continuous_actions, num_good= env_agent_n, num_adversaries=4)
    env.reset()
    dim_info = {}
    for agent_id in env.agents:
        dim_info[agent_id] = []
        if isinstance(env.observation_space(agent_id), gym.spaces.Box):
            dim_info[agent_id].append(env.observation_space(agent_id).shape[0])
        else:
            dim_info[agent_id].append(1)
        if isinstance(env.action_space(agent_id), gym.spaces.Box):
            dim_info[agent_id].append(env.action_space(agent_id).shape[0])
        else:
            dim_info[agent_id].append(env.action_space(agent_id).n)
    if continuous_actions:
        return env,dim_info, 1, True # pettingzoo.mpe 环境中，max_action均为1 , 选取连续环境is_continue = True
    else:
        return env,dim_info, None, False

## make_dir 
def make_dir(env_name,policy_name = 'DQN',trick = None):
    script_dir = os.path.dirname(os.path.abspath(__file__)) # 当前脚本文件夹
    env_dir = os.path.join(script_dir,'./results', env_name)
    os.makedirs(env_dir) if not os.path.exists(env_dir) else None
    print('trick:',trick)
    # 确定前缀
    if trick is None or not any(trick.values()) or policy_name =='MAPPO':
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
环境见:simple_adversary_v3,simple_crypto_v3,simple_push_v3,simple_reference_v3,simple_speaker_listener_v3,simple_spread_v3,simple_tag_v3
具体见:https://pettingzoo.farama.org/environments/mpe
注意：环境中N个智能体的设置   

7： mappo 1e-3
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="simple_adversary_v3") 
    parser.add_argument("--N", type=int, default=None) # 环境中智能体数量 默认None 这里用来对比设置
    parser.add_argument("--continuous_actions", type=bool, default=True ) #默认True 
    # 共有参数
    parser.add_argument("--seed", type=int, default=100) # 0 10 100  
    parser.add_argument("--max_episodes", type=int, default=int(120000))
    parser.add_argument("--save_freq", type=int, default=int(5000//4))
    parser.add_argument("--start_steps", type=int, default=0) # 满足此开始更新 此算法不用
    parser.add_argument("--random_steps", type=int, default=0)  # 满足此开始自己探索
    parser.add_argument("--learn_steps_interval", type=int, default=0) # 这个算法不方便用
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    ## A-C参数   
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    # PPO独有参数
    parser.add_argument("--horizon", type=int, default=256) # 
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--K_epochs", type=int, default=15) # 15 # 困难任务建议设置为5
    parser.add_argument("--entropy_coefficient", type=float, default=0.01)
    parser.add_argument("--minibatch_size", type=int, default=256)  
    parser.add_argument("--lmbda", type=float, default=0.95) # GAE参数
    ## mappo 参数
    parser.add_argument("--huber_delta", type=float, default=10.0) # huber_loss参数
    # trick参数
    parser.add_argument("--policy_name", type=str, default='MAPPO')
    parser.add_argument("--trick", type=dict, default={'adv_norm':False,
                                                        'ObsNorm':False,
                                                        'reward_norm':False,'reward_scaling':False,    # or
                                                        'orthogonal_init':False,'adam_eps':False,'lr_decay':False, # 原代码中设置为False
                                                        # 以上均在PPO_with_tricks.py中实现过
                                                       'ValueClip':False,'huber_loss':False,
                                                       'LayerNorm':False,'feature_norm':False,
                                                       })  
    # device参数   
    parser.add_argument("--device", type=str, default='cuda') # cpu/cuda

    args = parser.parse_args()
    # 检查 reward_norm 和 reward_scaling 的值
    if args.trick['reward_norm'] and args.trick['reward_scaling']:
        raise ValueError("reward_norm 和 reward_scaling 不能同时为 True")
    
    if  args.policy_name == 'MAPPO' or ((args.trick['lr_decay'] is False ) and all(value  for key, value in args.trick.items() if key not in ['reward_norm','lr_decay'])) :
        args.policy_name = 'MAPPO'
        for key in args.trick.keys():
            if key not in ['reward_norm','lr_decay']:
                args.trick[key] = True
            else:
                args.trick[key] = False
    
    if args.policy_name == 'MAPPO_simple' or (not any(args.trick.values())) : # if all(value is False for value in args.trick.values()):
        args.policy_name = 'MAPPO_simple'
        for key in args.trick.keys():
            args.trick[key] = False

    print(args)
    print('-' * 50)
    print('Algorithm:',args.policy_name)

    ## 环境配置
    env,dim_info,max_action,is_continue = get_env(args.env_name, env_agent_n = args.N, continuous_actions = args.continuous_actions)
    print(f'Env:{args.env_name}  dim_info:{dim_info}  max_action:{max_action}  max_episodes:{args.max_episodes}')

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
    policy = MAPPO(dim_info, is_continue, args.actor_lr, args.critic_lr, args.horizon, device, args.trick)

    time_ = time.time()
    ## 训练
    episode_num = 0
    step = 0
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}
    obs,info = env.reset(seed = args.seed) # env.reset(seed = args.seed)  # 针对obs复现:env.reset()
    {agent: env.action_space(agent).seed(seed = args.seed) for agent in env_agents}  # 针对action复现:env.action_space.sample()

    if args.trick['ObsNorm']:
        obs_norm = {agent_id  :Normalization(shape = dim_info[agent_id][0]) for agent_id in env_agents }
        obs = {agent_id : obs_norm[agent_id](obs[agent_id]) for agent_id in env_agents }
    
    if args.trick['reward_norm'] :
        reward_norm = {agent_id : Normalization(shape=1) for agent_id in env_agents}
    elif args.trick['reward_scaling']:
        reward_norm = {agent_id : RewardScaling(shape=1, gamma=args.gamma,) for agent_id in env_agents}
    
    while episode_num < args.max_episodes:
        step +=1

        # 获取动作
        action , action_log_pi = policy.select_action(obs)   # action (-1,1)
        if is_continue:
            action_ = {agent_id: np.clip(action[agent_id] * max_action, -max_action, max_action,dtype= np.float32) for agent_id in action}
            action_ = {agent_id: (action_[agent_id] + 1) / 2 for agent_id in env_agents}  # [-1,1] -> [0,1]
        else:
            action_ = { agent_id: int(action[agent_id]) for agent_id in env_agents} ## 针对PettingZoo离散动作空间 np.array(0) -> int(0)
            
        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        if args.trick['ObsNorm']:
            next_obs = {agent_id : obs_norm[agent_id](next_obs[agent_id]) for agent_id in env_agents }
        if args.trick['reward_norm'] or args.trick['reward_scaling']:
            reward_ = {agent_id : reward_norm[agent_id](reward[agent_id]) for agent_id in env_agents}

        done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
        done_bool = {agent_id: terminated[agent_id]  for agent_id in env_agents} ### truncated 为超过最大步数
        if args.trick['reward_norm'] or args.trick['reward_scaling']:
            policy.add(obs, action, reward_, next_obs, done_bool, action_log_pi, done)
        else:
            policy.add(obs, action, reward, next_obs, done_bool, action_log_pi, done)

        episode_reward = {agent_id: episode_reward[agent_id] + reward[agent_id] for agent_id in env_agents}
        obs = next_obs

        if any(done.values()):
            ## 显示
            if  (episode_num + 1) % 100 == 0:
                print("episode: {}, reward: {}".format(episode_num + 1, episode_reward))
            for agent_id in env_agents:
                writer.add_scalar(f'reward_{agent_id}', episode_reward[agent_id], episode_num + 1)
                train_return[agent_id].append(episode_reward[agent_id])

            episode_num += 1
            obs,info = env.reset(seed = args.seed) # env.reset(seed = args.seed)  # 针对obs复现:env.reset()
            if args.trick['ObsNorm']:
                obs = {agent_id : obs_norm[agent_id](obs[agent_id]) for agent_id in env_agents }
            if args.trick['reward_scaling']:
                {agent_id : reward_norm[agent_id].reset() for agent_id in env_agents }
            episode_reward = {agent_id: 0 for agent_id in env_agents}
        
        # 满足step,更新网络
        if step % args.horizon == 0:
            policy.learn(args.minibatch_size, args.gamma, args.lmbda, args.clip_param, args.K_epochs, args.entropy_coefficient,args.huber_delta)
            if args.trick['lr_decay']:
                policy.lr_decay(episode_num,max_episodes=args.max_episodes)

        # 保存模型
        if episode_num % args.save_freq == 0:
            policy.save(model_dir)

    print('total_time:',time.time()-time_)
    policy.save(model_dir)
    ## 保存数据
    train_return_ = np.array([train_return[agent_id] for agent_id in env.agents])
    if args.N is None:
        np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),train_return_)
    else:
        np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}_N_{len(env_agents)}.npy"),train_return_)
    
    if args.trick['ObsNorm']:
        obs_norm_ = {agent_id: [obs_norm[agent_id].running_ms.mean, obs_norm[agent_id].running_ms.std] for agent_id in env_agents}
        pickle.dump(obs_norm_, open(os.path.join(model_dir,'obs_norm.pkl'), 'wb'))








