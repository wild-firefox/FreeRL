import torch
import torch.nn as nn
import torch.nn.functional as F

import math

'''
NoiseLinear层:
论文: Noisy Networks for Exploration  https://arxiv.org/pdf/1706.10295
论文中提出了两种noise 1. factorized Gaussian noise 2. independent Gaussian noise 论文主要选择了第1种用于DQN,第2种用于A3C,因为第1种耗时短
这里实现了第1种
公式: y = (u + sigma * epsilon)x + (v + sigma * epsilon)
论文中 均值u ,标准差sigma_init= 0.4
NoisyLinear:可以将所有层都改成NoisyLinear , 也可以只将输出层改成NoisyLinear(略微增加计算量,且达到增加探索的效果)
'''

class NoisyLinear(nn.Module):
    def __init__(self, in_dim, out_dim, sigma_init=0.05,):
        super(NoisyLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_dim, in_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_dim, in_dim))
        # 注册持久性缓冲区,不可用于反向传播,但可以保存在模型中
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_dim, in_dim))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_dim))

        # 初始化
        self.is_train = True
        self.reset_parameters()
        self.reset_noise()
        torch.manual_seed(100)

    def forward(self, x):
        if self.is_train:
            self.reset_noise()  # 每次前向传播都要重置noise
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon  # out_dim x in_dim * out_dim x in_dim
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon # out_dim * out_dim
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias) # 即 x * weight.t() + bias

    def reset_parameters(self):
        '''公式
        sample from u(-1/sqrt(in_dim), 1/sqrt(in_dim))
        sigma = sigma_init / sqrt(in_dim) 注：bias的sigma要除以out_dim , 因为bias是out_dim维度
        '''
        mu_range = 1 / math.sqrt(self.in_dim)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_dim))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_dim))  # 这里要除以out_dim

    def reset_noise(self):
        '''公式:
        episilon_(j,i)^w = f(out_dim)*f(in_dim) #ger:生成out_dim*in_dim的矩阵 result[i, j] = a[i] * b[j]
        episilon_(j)^b = f(out_dim)
        '''
        epsilon_i = self.scale_noise(self.in_dim)
        epsilon_j = self.scale_noise(self.out_dim)
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        '''公式: f(x) = sign(x) * sqrt(|x|)'''
        x = torch.randn(size)  # torch.randn产生标准高斯分布 均值为0 方差为1  unit Gaussian variables
        x = x.sign() * torch.sqrt(abs(x)) 
        return x

'''
另外一篇 noise 论文 openai 提出 https://openreview.net/forum?id=ByBAl2eAZ
'''