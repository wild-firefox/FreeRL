import torch
import numpy as np

'''
关于ObsNorm  原理和ddpg中使用的ObsNorm原理一致,这里直接复制过来。
但是PPO是online算法,每次得把buffer装满后才能更新,之后再把buffer清空,所以状态归一化的速度会很慢,不如使用Normalization。
而DDPG的mean和std的更新频率高，适合使用RunningMeanStd_batch_size。
'''
''' 
补充：ObsNorm 根据原论文的描述:此技术将小批量中样本的每个维度标准化为具有单位均值和方差,此外,它还维护平均值和方差的运行平均值。这个trick更像是RunningMeanStd 
这里选用参考3
参考1：https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/networks.py#L19 # 直接使用batchnorm 不符合原论文
参考2：https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg_learner.py#L103 # √
参考3：https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/5.PPO-continuous/normalization.py#L4
参考4：https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py#L62 与参考3类似
'''
class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False #是否更新均值和方差，在评估时，update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

'''modify 
根据上述RuningMeanStd的方法和ddpg原论文的描述，将RuningMeanStd的方法（一个state一个state的更新）改进成ddpg原论文描述的（一个batch_size的state一个batch_size的state的更新）
'''
class RunningMeanStd_batch_size:
    # Dynamically calculate mean and std
    def __init__(self, shape,device):  # shape:the dimension of input data
        self.n = 0
        self.mean = torch.zeros(shape).to(device)
        self.S = torch.zeros(shape).to(device)
        self.std = torch.sqrt(self.S).to(device)

    def update(self, x):
        x = x.mean(dim=0,keepdim=True)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean 
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n)


class Normalization_batch_size:
    def __init__(self, shape, device):
        self.running_ms = RunningMeanStd_batch_size(shape,device)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False #是否更新均值和方差，在评估时，update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x
    

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)