import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal,Categorical

import numpy as np
from Buffer import Buffer_for_PPO , ReplayBuffer

from copy import deepcopy
import pettingzoo #动态导入
import gymnasium as gym
import importlib
import argparse

## tricks
from normalization import Normalization,RewardScaling

## 其他
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import BatchSampler, SequentialSampler
import time
import re
import pickle 


''' mappo:论文链接：https://arxiv.org/pdf/2103.01955 代码链接：https://github.com/marlbenchmark/on-policy/'''


''' 此离散代码 修改自github: https://github.com/Lizhi-sjtu/MARL-code-pytorch/tree/main/1.MAPPO_MPE 

与MAPPO.py中实现的discrete(不收敛)不同之处：（大方向的不同）
1.单智能体 actor discrete 1维 0-4  (输入 1xobs_dim 输出 1x1 ) -> 多智能体 actor discrete n维 [0-4,0-4,0-4,...]  (输入 nxobs_dim 输出nx1 ) (局限：各个obs_dim必须相同)
2.Buffer修改  多了智能体的n维 (局限：有episode_length限制)
3.critic 输入 nxobs_dim 输出nx1  (局限：各个obs_dim必须相同)
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
            x = F.layer_norm(obs, obs.size()[1:])
        x = F.relu(self.l1(obs))
        if self.trick['LayerNorm']:
            x = F.layer_norm(x, x.size()[1:])
        x = F.relu(self.l2(x))
        if self.trick['LayerNorm']:
            x = F.layer_norm(x, x.size()[1:])
        a_prob = torch.softmax(self.l3(x), dim=-1) ## ！！ 注意这里为-1
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
        #s = torch.cat(list(s), dim = 1)
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

        if trick['adam_eps']:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr= actor_lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr= actor_lr)

        '''
        # if trick['adam_eps']:
        #     self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        #     self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        # else:
        #     self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        #     self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    # def update_actor(self, loss):
    #     self.actor_optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
    #     self.actor_optimizer.step()

    # def update_critic(self, loss):
    #     self.critic_optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
    #     self.critic_optimizer.step()
    '''

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
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, horizon, device, trick = None ,buffer = None):        

        
        self.agent_x = list(dim_info.keys())[0]
        obs_dim, action_dim = dim_info[self.agent_x]
        self.N = len(dim_info)

        self.agent = Agent(obs_dim, action_dim, dim_info, actor_lr, critic_lr, is_continue, device,trick)
        self.buffer = buffer
        self.batch_size = buffer.batch_size
        self.episode_limit = buffer.episode_limit

        self.device = device
        self.is_continue = is_continue
        print('actor_type:continue') if self.is_continue else print('actor_type:discrete')

        self.horizon = int(horizon)

        self.trick = trick

        if self.trick['lr_decay']:
            self.actor_lr = actor_lr
            self.critic_lr = critic_lr

    def select_action(self, obs):
        # obs : Nxobs_dim
        obs = torch.as_tensor(np.array(obs),dtype=torch.float32).to(self.device)

        if self.is_continue: 
            mean, std = self.agent.actor(obs)
            dist = Normal(mean, std)
            action = dist.sample()
            action_log_pi = dist.log_prob(action) # 1xaction_dim
        else:
            dist = Categorical(probs=self.agent.actor(obs))
            action = dist.sample()
            action_log_pi = dist.log_prob(action)
        # to 真实值
        actions = action.detach().cpu().numpy() # 3 
        action_log_pis = action_log_pi.detach().cpu().numpy() # 3 

        return actions , action_log_pis
    
    def evaluate_action(self,obs):
        '''只对discrete有效'''
        # obs : Nxobs_dim
        obs = torch.as_tensor(np.array(obs),dtype=torch.float32).to(self.device)

        action = self.agent.actor(obs).argmax(dim=-1) # 1xaction_dim
        action = action.detach().cpu().numpy() # 3

        return action

    
    ## buffer 相关
    def get_value(self, s): # copy 
        with torch.no_grad():
            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1).to(self.device)  # (state_dim,)-->(N,state_dim)
            critic_inputs.append(s)
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_input.shape=(N, critic_input_dim)
            v_n = self.agent.critic(critic_inputs)  # v_n.shape(N,1)
        return v_n.detach().cpu().numpy().flatten()
        
    def get_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_inputs.shape=(batch_size, episode_limit, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_inputs.shape=(batch_size, episode_limit, N, critic_input_dim)
        return actor_inputs, critic_inputs
        
    def add(self, obs, action, reward, next_obs, done, action_log_pi , adv_dones ,episode_step):
        ''' 这里并没有使用adv_dones/dones 和 next_obs ，只是为了保持接口一致'''
        # obs : Nxobs_dim  
        obs_n = obs
        s = np.array(obs).flatten() 
        v_n = self.get_value(s) 
        a_n = action
        a_logprob_n = action_log_pi
        r_n = [v for v in reward.values()]
        done_n = [d for d in done.values()]   # 参考的代码用的是done


        self.buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

    
    ## PPO算法相关
    def learn(self, minibatch_size, gamma, lmbda ,clip_param, K_epochs, entropy_coefficient, huber_delta = None): # huber
        # 多智能体特有-- 集中式训练critic:要用到所有智能体next状态和动作

        batch = self.buffer.get_training_data()  # get training data

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            deltas = batch['r_n'] + gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]  # deltas.shape=(batch_size,episode_limit,N)
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + gamma * lmbda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,episode_limit,N)
            v_target = adv + batch['v_n'][:, :-1]  # v_target.shape(batch_size,episode_limit,N)
            if self.trick['adv_norm']:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # if self.use_adv_norm:  # Trick 1: advantage normalization
            #     adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs = self.get_inputs(batch)

        # Optimize policy for K epochs:
        for _ in range(K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), minibatch_size, False):
                """
                    get probs_now and values_now
                    probs_now.shape=(mini_batch_size, episode_limit, N, action_dim)
                    values_now.shape=(mini_batch_size, episode_limit, N)
                """
                probs_now = self.agent.actor(actor_inputs[index])
                values_now = self.agent.critic(critic_inputs[index]).squeeze(-1)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()  # dist_entropy.shape=(mini_batch_size, episode_limit, N)
                # batch['a_n'][index].shape=(mini_batch_size, episode_limit, N)
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])  # a_logprob_n_now.shape=(mini_batch_size, episode_limit, N)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())  # ratios.shape=(mini_batch_size, episode_limit, N)
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - entropy_coefficient * dist_entropy

                # if self.use_value_clip:
                #     values_old = batch["v_n"][index, :-1].detach()
                #     values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                #     values_error_original = values_now - v_target[index]
                #     critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                # else:
                if self.trick['ValueClip']:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -clip_param, clip_param) + values_old - v_target[index]
                    if self.trick['huber_loss']:
                        values_error_clip = huber_loss(values_error_clip, huber_delta).mean()
                        values_error_original = huber_loss(values_now - v_target[index], huber_delta).mean()
                    else:
                        values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2

                total_loss = actor_loss.mean() + critic_loss.mean()
                self.agent.update_ac(total_loss)  
                
                # self.agent.ac_optimizer.zero_grad()
                # ac_loss = actor_loss.mean() + critic_loss.mean()
                # ac_loss.backward()
                # # if self.use_grad_clip:  # Trick 7: Gradient clip
                # #     torch.nn.utils.clip_grad_norm_(self.ac_parameters, 0.5)#10.0)
                
                self.agent.ac_optimizer.step()
        
        self.buffer.reset_buffer()
                
    def lr_decay(self,episode_num,max_episodes):
        lr_a_now = self.actor_lr * (1 - episode_num / max_episodes)
        '''
        lr_c_now = self.critic_lr * (1 - episode_num / max_episodes)
        for agent in self.agents.values():
            for p in agent.actor_optimizer.param_groups:
                p['lr'] = lr_a_now
            for p in agent.critic_optimizer.param_groups:
                p['lr'] = lr_c_now
        '''
        for p in self.agent.ac_optimizer.param_groups:
            p['lr'] = lr_a_now
 

    def save(self, model_dir):
        torch.save(
            #{name: agent.actor.state_dict() for name, agent in self.agents.items()},
            self.agent.actor.state_dict(),
            os.path.join(model_dir, 'MAPPO_discrete.pth')
        )
        
    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue, model_dir,trick=None):
        policy = MAPPO(dim_info, is_continue = is_continue, actor_lr = 0, critic_lr = 0, horizon = 0, device = 'cpu',trick=trick)
        data = torch.load(os.path.join(model_dir, 'MAPPO_discrete.pth'))
        # for agent_id, agent in policy.agents.items():
        #     agent.actor.load_state_dict(data[agent_id])
        policy.agent.actor.load_state_dict(data)
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
    parser.add_argument("--env_name", type = str,default="simple_spread_v3") 
    parser.add_argument("--N", type=int, default=None) # 环境中智能体数量 默认None 这里用来对比设置
    parser.add_argument("--episode_limit", type=int, default=25)
    parser.add_argument("--continuous_actions", type=bool, default=False) 
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
    parser.add_argument("--actor_lr", type=float, default=1e-3)  ## 这里的AC参数是一起优化的,只用到了actor_lr
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

    ## buffer
    agent_x = list(dim_info.keys())[0]
    obs_dim, action_dim = dim_info[agent_x]
    buffer = ReplayBuffer(N = len(dim_info), obs_dim = obs_dim, state_dim = sum(val[0] for val in dim_info.values()), episode_limit = args.episode_limit ,batch_size = args.horizon ,device = args.device)

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
    policy = MAPPO(dim_info, is_continue, args.actor_lr, args.critic_lr, args.horizon, device, args.trick ,buffer)

    time_ = time.time()
    ## 训练
    episode_num = 0
    step = 0
    episode_step = 0
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}
    obs,info = env.reset()
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
        obs = [ v for v in obs.values()]
        action , action_log_pi = policy.select_action(obs)   # action (-1,1)
        if is_continue:
            action_ = {agent_id: np.clip(action[agent_id] * max_action, -max_action, max_action,dtype= np.float32) for agent_id in action}
            action_ = {agent_id: (action_[agent_id] + 1) / 2 for agent_id in env_agents}  # [-1,1] -> [0,1]
        else:
            action_ = { agent: action[i] for i,agent in enumerate(env_agents)}
            
            
        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        if args.trick['ObsNorm']:
            next_obs = {agent_id : obs_norm[agent_id](next_obs[agent_id]) for agent_id in env_agents }
        if args.trick['reward_norm'] or args.trick['reward_scaling']:
            reward_ = {agent_id : reward_norm[agent_id](reward[agent_id])[0] for agent_id in env_agents}

        done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
        done_bool = {agent_id: terminated[agent_id]  for agent_id in env_agents} ### truncated 为超过最大步数
        
        if args.trick['reward_norm'] or args.trick['reward_scaling']:
            policy.add(obs, action, reward_, next_obs, done_bool, action_log_pi, done , episode_step)
        else:
            policy.add(obs, action, reward, next_obs, done_bool, action_log_pi, done , episode_step)

        episode_step += 1
        episode_reward = {agent_id: episode_reward[agent_id] + reward[agent_id] for agent_id in env_agents}

        obs = next_obs

        if any(done.values()):
            
            obs = [ v for v in next_obs.values()]
            s = np.array(obs).flatten()
            v_n = policy.get_value(s)
            policy.buffer.store_last_value(episode_step , v_n)
            episode_step = 0

            ## 显示
            if  (episode_num + 1) % 100 == 0:
                print("episode: {}, reward: {}".format(episode_num + 1, episode_reward))
                for agent_id in env_agents:
                    writer.add_scalar(f'reward_{agent_id}', episode_reward[agent_id], episode_num + 1)
                    train_return[agent_id].append(episode_reward[agent_id])

            episode_num += 1
            obs,info = env.reset()
            if args.trick['ObsNorm']:
                obs = {agent_id : obs_norm[agent_id](obs[agent_id]) for agent_id in env_agents }
            if args.trick['reward_scaling']:
                {agent_id : reward_norm[agent_id].reset() for agent_id in env_agents }
            episode_reward = {agent_id: 0 for agent_id in env_agents}
        
        # 满足step,更新网络
        if policy.buffer.episode_num == args.horizon:
        #if step % args.horizon == 0:
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








