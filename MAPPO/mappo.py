import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
from Buffer import Buffer # 与DQN.py中的Buffer一样

from copy import deepcopy
import pettingzoo #动态导入
import gymnasium as gym
import importlib
import argparse
from torch.utils.tensorboard import SummaryWriter
import time

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
6.Layer Normalization：这个技巧来自文献：Regularization matters in policy optimization-an empirical study on continuous control。
7.Soft Trust-Region Penalty：这个技巧来自文件：Revisiting design choices in proximal policy optimization。
'''

## 先写一个简单的continue环境实现：
'''
关于建议
1.由于在ddpg中发现建议1中的popart效果并不理想，所以这里不使用popart
2.知乎上的评论：论证了建议2中可以不用特别加入agent-specific information,所以按一般MA拼接状态和动作处理
3.设置epocch = 5 ; minibatch_size = 1 #后者我做过实验，效果确实会好一点点
4.设置clip = 0.2
5.和原ppo一样看情况调整horizon

综上:关注3，4即可

关于trick:
1.使用GAE 与原PPO一致
2.Input Normalization 即trick中的ObsNorm 确实重要
3.Value Clipping 新增
4.Orthogonal Initialization 即trick中的orthogonal_init
5.Gredient Clipping PPO中默认加入以防止梯度爆炸和消失 
6.Layer Normalization 新增
7.Soft Trust-Region Penalty 新增

综上：重点关注3,6,7

另外:
mappo与rmappo区别见原代码:https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/scripts/train/train_mpe.py#L68
这里rmappo使用了RNN
'''


## 第一部分：定义Agent类
def net_init(m,gain=None):
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
    use_relu = True

    init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_]
    activate_fuction = ['tanh','relu', 'leaky_relu']  # relu 和 leaky_relu 的gain值一样
    gain = gain if gain is not None else gain = nn.init.calculate_gain(activate_fuction[use_relu]) # 根据的激活函数设置
    
    init_method[use_orthogonal](m.weight, gain=gain)
    nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128,trick = None):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.mean_layer = nn.Linear(hidden_2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim)) # 于PPO.py的方法一致：对角高斯函数

        # 使用 orthogonal_init
        if trick['orthogonal_init']:
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.mean_layer, gain=0.01)   

        
    
    def forward(self, x, ):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        mean = torch.tanh(self.mean_layer(x))  # 使得mean在-1,1之间

        log_std = self.log_std.expand_as(mean)  # 使得log_std与mean维度相同 输出log_std以确保std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2) # exp(-20) - exp(2) 等于 2e-9 - 7.4，确保std在合理范围内
        std = torch.exp(log_std)

        return mean, std    
        
class Critic(nn.Module):
    def __init__(self, dim_info:dict, hidden_1=128 , hidden_2=128):
        super(Critic, self).__init__()
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())  
        
        self.l1 = nn.Linear(global_obs_act_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)
        

        # 使用 orthogonal_init
        net_init(self.l1)
        net_init(self.l2)
        net_init(self.l3)  
        
    def forward(self, s, a): # 传入全局观测和动作
        sa = torch.cat(list(s)+list(a), dim = 1)
        #sa = torch.cat([s,a], dim = 1)
        
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q
    

    

    




