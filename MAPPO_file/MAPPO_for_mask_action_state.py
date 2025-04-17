import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal,Categorical

import numpy as np
from Buffer import Buffer_for_PPO_mask,Buffer_for_PPO_state_mask

from copy import deepcopy

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

## 环境
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

'''
将此代码改成兼容SMACV2游戏环境的版本，增加了对action_mask的支持。
其他的实验效果链接：https://zhuanlan.zhihu.com/p/560339507
加上了环境中的state状态
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
        # a_prob = torch.softmax(self.l3(x), dim=1)
        # return a_prob
        return self.l3(x)
        
class Critic(nn.Module):
    def __init__(self, dim_info:dict[str,list], hidden_1=128 , hidden_2=128,trick = None,state_dim=None):
        super(Critic, self).__init__()
        global_obs_dim = sum(val[0] for val in dim_info.values())  
        if state_dim is not None: # 全局状态维度
            global_obs_dim += state_dim
        
        self.l1 = nn.Linear(global_obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)
        
        self.trick = trick
        # 使用 orthogonal_init
        if trick['orthogonal_init']:
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.l3)  
        
    def forward(self, s,state): # 传入全局观测和动作
        s = torch.cat(list(s), dim = 1)
        s = torch.cat([s, state], dim = 1) #if state is not None else s # 如果有全局状态则拼接
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
    def __init__(self, obs_dim, action_dim, dim_info,actor_lr, critic_lr, is_continue, device,trick,state_dim):   
        
        if is_continue:
            self.actor = Actor(obs_dim, action_dim,trick=trick ).to(device)
        else:
            self.actor = Actor_discrete(obs_dim, action_dim, trick=trick).to(device) 
        self.critic = Critic( dim_info ,trick=trick, state_dim=state_dim).to(device) ## 不能少写等于

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


class CategoricalMasked(Categorical):
    '''
    参考：https://github.com/vwxyzjn/invalid-action-masking/blob/master/invalid_action_masking/ppo_10x10.py#L201-L216
    validate_args为Categorical的参数:表示是否验证参数
    probs为归一化的概率值，logits为未归一化的对数概率，
    probs->logits的转换:
    logits = torch.tensor([2.0, 3.0, -1e8])
    probs = torch.softmax(logits, dim=-1)  # 输出: [0.2689, 0.7311, 0.0]
    masks:例：[True, True, False]
    '''
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = self.masks.bool().to(device) #masks.type(torch.BoolTensor)#.to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device)) 
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs # 父类中得到
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.))
        return -p_log_p.sum(-1)
    
## 第二部分：定义DQN算法类
def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

class MAPPO: 
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, horizon, device, trick = None,state_dim=None):        
        self.agents  = {}
        self.buffers = {}
        for agent_id, (obs_dim, action_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, action_dim, dim_info, actor_lr, critic_lr, is_continue, device,trick,state_dim)
            if state_dim is None:
                self.buffers[agent_id] = Buffer_for_PPO_mask(horizon, obs_dim, act_dim = action_dim if is_continue else 1, device = device ,mask_act_dim = action_dim)
            else:
                self.buffers[agent_id] = Buffer_for_PPO_state_mask(horizon, obs_dim, act_dim = action_dim if is_continue else 1, device = device ,mask_act_dim = action_dim,state_dim=state_dim)

        self.device = device
        self.is_continue = is_continue
        print('actor_type:continue') if self.is_continue else print('actor_type:discrete')

        self.horizon = int(horizon)

        self.trick = trick
        self.num_agents = len(self.agents) # 智能体数量

        if self.trick['lr_decay']:
            self.actor_lr = actor_lr
            self.critic_lr = critic_lr
        
        self.state_dim = state_dim # 全局状态维度


    def select_action(self, obs, avail_actions):
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
                masks = torch.as_tensor(avail_actions[agent_id], dtype=torch.bool).reshape(1, -1).to(self.device) # 1xaction_dim
                dist = CategoricalMasked(logits=self.agents[agent_id].actor(obs),masks=masks)
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
    def add(self, obs, action, reward, next_obs, done, action_log_pi , adv_dones,avail_actions,state=None):
        for agent_id, buffer in self.buffers.items():
            buffer.add(obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id], action_log_pi[agent_id] , adv_dones[agent_id], avail_actions[agent_id],state)

    def all(self):
        obs = {}
        action = {}
        reward = {}
        next_obs = {}
        done = {}
        action_log_pi = {}
        adv_dones = {}
        avail_actions = {}
        for agent_id, buffer in self.buffers.items():
            obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id], action_log_pi[agent_id], adv_dones[agent_id] ,avail_actions[agent_id] , state = buffer.all()
        return obs, action, reward, next_obs, done, action_log_pi, adv_dones , avail_actions , state

    ## PPO算法相关
    def learn(self, minibatch_size, gamma, lmbda ,clip_param, K_epochs, entropy_coefficient, huber_delta = None):
        # 多智能体特有-- 集中式训练critic:要用到所有智能体next状态和动作
        obs, action, reward, next_obs, done , action_log_pi , adv_dones , avail_actions ,state = self.all()
        
        self.horizon = len(obs[list(obs.keys())[0]])  # horizon = batch_size # 修改成片段式学习
        minibatch_size = self.horizon
        # 计算GAE
        with torch.no_grad():  # adv and v_target have no gradient
            adv = torch.zeros(self.horizon, self.num_agents)
            gae = 0
            vs = []
            vs_ = []
            for agent_id  in self.buffers.keys():
                #state= torch.tensor(state, dtype=torch.float32) ### 维度不对 之后改
                vs.append(self.agents[agent_id].critic(obs.values(),state))  # batch_size x 1
                vs_.append(self.agents[agent_id].critic(next_obs.values(),state))
            
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
                        masks = torch.as_tensor(avail_actions[agent_id][index], dtype=torch.bool).reshape(minibatch_size,-1).to(self.device) # mini_batch_size x action_dim
                        dist_now = CategoricalMasked(logits=self.agents[agent_id].actor(obs[agent_id][index]),masks=masks)
                        dist_entropy = dist_now.entropy().reshape(-1,1) # mini_batch_size  -> mini_batch_size x 1
                        action_log_pi_now = dist_now.log_prob(action[agent_id][index].reshape(-1)).reshape(-1,1) # mini_batch_size  -> mini_batch_size x 1

                    ratios = torch.exp(action_log_pi_now.sum(dim = 1, keepdim=True) - action_log_pi[agent_id][index].sum(dim = 1, keepdim=True))  # shape(mini_batch_size X 1)
                    surr1 = ratios * adv[index]   # mini_batch_size x 1
                    surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * adv[index]  
                    actor_loss = -torch.min(surr1, surr2).mean() - entropy_coefficient * dist_entropy.mean()
                    agent.update_actor(actor_loss)

                    # 再更新critic
                    obs_ = {agent_id: obs[agent_id][index] for agent_id in obs.keys()}

                    v_s = self.agents[agent_id].critic(obs_.values(),state) # mini_batch_size x 1
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
    
def get_smac_env(env_agent_n = None,n_enemies = None):
    
    n_units = env_agent_n if env_agent_n is not None else 3 # 默认3个智能体
    n_enemies = n_enemies if n_enemies is not None else 3 # 默认3个敌人

    ## 自定义地图和配置
    distribution_config = {
            "n_units": n_units,
            "n_enemies": n_enemies,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"], # 
                "exception_unit_types": ["medivac"],
                "weights": [0.45, 0.45, 0.1],
                "observe": True,
            },
            ## 起始位置
            "start_positions": {
                "dist_type": "surrounded_and_reflect", # 围绕地图四周生成
                "p": 0.5, # 生成在地图四周的概率
                "n_enemies": n_enemies,
                "map_x": 32, # 地图大小
                "map_y": 32,
            },
        }
    env = StarCraftCapabilityEnvWrapper(
        capability_config=distribution_config,
        map_name="10gen_terran",
        debug=True, # 默认  True 开启的话会打印动作信息
        conic_fov=False,
        obs_own_pos=True,
        use_unit_ranges=True,
        min_attack_range=2,
    )
    env_info = env.get_env_info()
    print("Environment Info:")
    print(f"number of agents: {env_info['n_agents'] }")
    print(f"action_dim: {env_info['n_actions']}")
    print(f"obs_dim: {env_info['obs_shape']}")
    print(f"state_dim: {env_info['state_shape']}")
    print(f"episode_limit: {env_info['episode_limit']}")
    dim_info = {}
    for agent_id in range(env_info["n_agents"]):
        dim_info[agent_id] = []
        dim_info[agent_id].append(env_info['obs_shape'])
        dim_info[agent_id].append(env_info['n_actions']) # 离散动作空间

    return env, dim_info, None, False ,env_info# SMACV2环境中，max_action均为None , 选取离散环境is_continue = False
    



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
环境见:https://github.com/oxwhirl/smacv2

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="smacv2") 
    parser.add_argument("--N", type=int, default=None) # 环境中智能体数量 默认None 这里用来对比设置
    parser.add_argument("--continuous_actions", type=bool, default=True ) #默认True 
    # 共有参数
    parser.add_argument("--seed", type=int, default=100) # 0 10 100  
    parser.add_argument("--max_episodes", type=int, default=int(30000))
    parser.add_argument("--save_freq", type=int, default=int(5000//4))
    parser.add_argument("--start_steps", type=int, default=0) # 满足此开始更新 此算法不用
    parser.add_argument("--random_steps", type=int, default=0)  # 满足此开始自己探索
    parser.add_argument("--learn_steps_interval", type=int, default=0) # 这个算法不方便用
    parser.add_argument("--learn_episode_interval", type=int, default=1) # 使用这个而不是 horizon minibatch_size #8
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    ## A-C参数   
    parser.add_argument("--actor_lr", type=float, default=5e-4)
    parser.add_argument("--critic_lr", type=float, default=5e-4)
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
    parser.add_argument("--device", type=str, default='cpu') # cpu/cuda

    ## 其他
    parser.add_argument("--use_state", type=bool, default=True) # 在train中改的话会有有太多if判断，故这里只支持True。

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

    # 此环境不进行observation归一化,因为环境里归一化过了
    args.trick['ObsNorm'] = False


    print(args)
    print('-' * 50)
    print('Algorithm:',args.policy_name)

    ## 环境配置
    env,dim_info,max_action,is_continue ,env_info= get_smac_env()
    print(f'Env:{args.env_name}  dim_info:{dim_info}  max_action:{max_action}  max_episodes:{args.max_episodes}')
    state_dim = env_info['state_shape'] if args.use_state else None # 全局状态维度

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
    policy = MAPPO(dim_info, is_continue, args.actor_lr, args.critic_lr, args.horizon, device, args.trick,state_dim)

    time_ = time.time()
    ## 训练
    episode_num = 0
    episode_step = 0
    step = 0
    env_agents = [agent_id for agent_id in range(len(dim_info))]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}
    win_list = [] # 胜利列表
    env.reset() # env.reset(seed = args.seed)  # 针对obs复现:env.reset()
    #{agent: env.action_space(agent).seed(seed = args.seed) for agent in env_agents}  # 针对action复现:env.action_space.sample()

    obs_n = env.get_obs() # obs_n.shape=(N,obs_dim)
    # if args.use_state:
    #     state = env.get_state()
    ## 环境包装
    obs = {agent_id: obs_n[agent_id] for agent_id in env_agents} 

    if args.trick['ObsNorm']:
        obs_norm = {agent_id  :Normalization(shape = dim_info[agent_id][0]) for agent_id in env_agents }
        obs = {agent_id : obs_norm[agent_id](obs[agent_id]) for agent_id in env_agents }
    
    if args.trick['reward_norm'] :
        reward_norm = {agent_id : Normalization(shape=1) for agent_id in env_agents}
    elif args.trick['reward_scaling']:
        reward_norm = {agent_id : RewardScaling(shape=1, gamma=args.gamma,) for agent_id in env_agents}
    
    while episode_num < args.max_episodes:
        step +=1
        episode_step += 1

        state = env.get_state() if args.use_state else None # 全局状态

        avail_a_n = env.get_avail_actions()  #avail_a_n.shape=(N,action_dim)
        ## 环境包装
        avail_actions = {agent_id: avail_a_n[agent_id] for agent_id in env_agents}

        # 获取动作
        action , action_log_pi = policy.select_action(obs,avail_actions)   # action 0-9

        #action_ = { agent_id: int(action[agent_id]) for agent_id in env_agents} ## 针对PettingZoo离散动作空间 np.array(0) -> int(0)
        action_ = [ action[agent_id] for agent_id in env_agents] 
        # 探索环境
        reward, env_done, infos = env.step(action_) 

        terminated = env_done
        truncated = False if episode_step != env_info['episode_limit'] else True 
        
        ## 环境包装
        terminated = {agent_id: terminated for agent_id in env_agents}
        truncated = {agent_id: truncated for agent_id in env_agents}

        reward = {agent_id: reward for agent_id in env_agents} # reward.shape=(1) -> {agent_id: reward[agent_id]}

        next_obs_n = env.get_obs() # next_obs_n.shape=(N,obs_dim)
        ## 环境包装
        next_obs = {agent_id: next_obs_n[agent_id] for agent_id in env_agents}

        if args.trick['ObsNorm']:
            next_obs = {agent_id : obs_norm[agent_id](next_obs[agent_id]) for agent_id in env_agents }
        if args.trick['reward_norm'] or args.trick['reward_scaling']:
            reward_ = {agent_id : reward_norm[agent_id](reward[agent_id]) for agent_id in env_agents}

        done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
        done_bool = {agent_id: terminated[agent_id]  for agent_id in env_agents} ### truncated 为超过最大步数
        if args.trick['reward_norm'] or args.trick['reward_scaling']:
            policy.add(obs, action, reward_, next_obs, done_bool, action_log_pi, done,avail_actions,state)
        else:
            policy.add(obs, action, reward, next_obs, done_bool, action_log_pi, done,avail_actions,state)

        episode_reward = {agent_id: episode_reward[agent_id] + reward[agent_id] for agent_id in env_agents}
        obs = next_obs

        if any(done.values()):
            #print(episode_step,truncated)
            ## 显示
            if  (episode_num + 1) % 100 == 0:
                print("episode: {}, reward: {}".format(episode_num + 1, episode_reward))
            
            for agent_id in env_agents:
                writer.add_scalar(f'reward_{agent_id}', episode_reward[agent_id], episode_num + 1)
                train_return[agent_id].append(episode_reward[agent_id])

            win_tag = 1 if env_done and 'battle_won' in infos and infos['battle_won'] else 0
            win_list.append(win_tag)
            if len(win_list) >= 100:
                win_rate = sum(win_list[-100:]) / 100
            else:
                win_rate = sum(win_list) / len(win_list)

            writer.add_scalar(f'win_rate', win_rate, episode_num + 1)
            
            episode_num += 1
            
            if args.trick['ObsNorm']:
                obs = {agent_id : obs_norm[agent_id](obs[agent_id]) for agent_id in env_agents }
            if args.trick['reward_scaling']:
                {agent_id : reward_norm[agent_id].reset() for agent_id in env_agents }
            episode_reward = {agent_id: 0 for agent_id in env_agents}
            episode_step = 0
        
            # 游戏结束 更新权重 
            if episode_num % args.learn_episode_interval == 0:
                policy.learn(args.minibatch_size, args.gamma, args.lmbda, args.clip_param, args.K_epochs, args.entropy_coefficient,args.huber_delta)
                if args.trick['lr_decay']:
                    policy.lr_decay(episode_num,max_episodes=args.max_episodes)
            
            # 更新完再开始
            env.reset() # env.reset(seed = args.seed)  # 针对obs复现:env.reset()

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








