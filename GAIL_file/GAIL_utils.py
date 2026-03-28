import torch
from typing import Tuple
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from torch import nn


class ExpertDataset(torch.utils.data.Dataset):
    """一个标准的 PyTorch Dataset，用于从 .npz 文件加载专家数据。"""
    def __init__(self, file_path: str):
        expert_data = np.load(file_path)
        self.states = expert_data['states']
        self.actions = expert_data['actions']

        ### 归一化
        # 得到states的每一维的上下界
        self.s_space = [[],[]] # [min,max]
        self.a_space = [[],[]] # [min,max]
        for i in range(self.states.shape[1]):
            self.s_space[0].append(np.min(self.states[:,i]))
            self.s_space[1].append(np.max(self.states[:,i]))
        for i in range(self.actions.shape[1]):
            self.a_space[0].append(np.min(self.actions[:,i]))
            self.a_space[1].append(np.max(self.actions[:,i]))
        self.space_high = np.array(self.s_space[1])
        self.space_low = np.array(self.s_space[0])
        self.action_high = np.array(self.a_space[1])
        self.action_low = np.array(self.a_space[0])
        ## 先归一到-1 ~ 1
        self.states = 2 * (self.states - self.space_low) / (self.space_high - self.space_low) - 1
        self.actions = 2 * (self.actions - self.action_low) / (self.action_high - self.action_low) - 1
        ### 归一化


        print(f"成功从 '{file_path}' 加载 {len(self.states)} 条专家数据。")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # 返回 float32 类型的张量
        return torch.as_tensor(self.states[idx], dtype=torch.float32), torch.as_tensor(self.actions[idx], dtype=torch.float32)

class InfiniteUniformSampler(torch.utils.data.Sampler):
    """一个无限数据采样器，提供无限长度的数据索引。"""
    def __init__(self, data_source, number_samples):
        self.data_source = data_source
        self.number_samples = number_samples # 这里的 number_samples 就是 batch_size

    def __iter__(self):
        n = len(self.data_source)
        while True:
            # 每次迭代都生成一个批次的随机索引
            rand_tensor = torch.randint(high=n, size=(self.number_samples,), dtype=torch.int64)
            yield rand_tensor.tolist()

class InfiniteDataLoader:
    """一个包装器，实现无限预取功能。"""
    def __init__(self, dataloader : torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self.dataset = self.dataloader.dataset
        # 创建一个可以无限获取数据的迭代器
        self.iter = iter(self.dataloader)

    def __next__(self):
        """允许通过 next(loader) 的方式获取下一批数据。"""
        try:
            return next(self.iter)
        except StopIteration:
            # 理论上由于 InfiniteUniformSampler，这里不会被触发，但作为保障
            self.iter = iter(self.dataloader)
            return next(self.iter)

def create_expert_loader(dataset: ExpertDataset, config: dict):
    """创建专家数据的无限 DataLoader。"""
    batch_size = config['PPO']['horizon']+1 if config['algo'] == 'MAIL' else config['PPO']['horizon']# 每次采样的批次大小
    
    # 使用自定义的无限采样器  ### 没有归一化
    batch_sampler = InfiniteUniformSampler(dataset, batch_size)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        pin_memory=True,
        num_workers=config['num_workers']
    )
    # 使用包装器来简化 next() 调用
    return InfiniteDataLoader(loader)


class Env_model_trainEnvWrapper:
    def __init__(self, env, config,):
        self.env = env
        self.config = config
        expert_data = np.load(config['expert_data_path'])
        self.states = expert_data['states']
        self.actions = expert_data['actions']
        ## 划分训练的部分
        data_ratio = config.get('data_ratio', 1.0)
        self.states = self.states[:int(data_ratio*len(self.states))]
        self.actions = self.actions[:int(data_ratio*len(self.actions))]
        ### 归一化
        # 得到states的每一维的上下界
        self.s_space = [[],[]] # [min,max]
        self.a_space = [[],[]] # [min,max]
        for i in range(self.states.shape[1]):
            self.s_space[0].append(np.min(self.states[:,i]))
            self.s_space[1].append(np.max(self.states[:,i]))
        for i in range(self.actions.shape[1]):
            self.a_space[0].append(np.min(self.actions[:,i]))
            self.a_space[1].append(np.max(self.actions[:,i]))
        self.space_high = np.array(self.s_space[1])
        self.space_low = np.array(self.s_space[0])
        self.action_high = np.array(self.a_space[1])
        self.action_low = np.array(self.a_space[0])
        ## 先归一到-1 ~ 1
        self.states = 2 * (self.states - self.space_low) / (self.space_high - self.space_low) - 1
        self.actions = 2 * (self.actions - self.action_low) / (self.action_high - self.action_low) - 1
        ###
        data_ratio = config.get('data_ratio', 1.0)
        self.train_states , self.val_states = self.states[:int(data_ratio*len(self.states))], self.states[int(data_ratio*len(self.states)):]
        self.train_actions , self.val_actions = self.actions[:int(data_ratio*len(self.actions))], self.actions[int(data_ratio*len(self.actions)):]
        self.states = self.train_states
        #del expert_data , self.actions

        ## 设置horizon
        self.horizon = config['M']['env_horizon']
        self.step_count = 0

        # 通过 @reproducible.setter 装饰器触发 setter 逻辑
        self.reproducible = config['env_reproducible']
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_gpu', False) else "cpu")

        self.val = False

    @property
    def reproducible(self):
        """获取可复现性状态"""
        return self._reproducible

    @reproducible.setter
    def reproducible(self, value: bool):
        """
        设置可复现性状态。
        当这个属性被赋值时，此方法会自动被调用。
        """
        self._reproducible = value
        self.seed = self.config['seed'] if self._reproducible else None
        np.random.seed(self.seed)
        #print(f"Environment reproducibility set to {self._reproducible}. Seed is now: {self.seed}")

    @property
    def val(self):
        return self._val
    
    @val.setter
    def val(self, value: bool):
        self._val = value
        if self._val:
            self.states = self.val_states
        else:
            self.states = self.train_states




    def reset(self,): ## TODO
        # 抽一个state
        self.start_index = 0 if self.val else np.random.randint(0, len(self.states)) ## TODO 是否正确 先这样
        state = self.states[self.start_index]

        #state = self.env.reset(seed = self.seed)[0]
        ## 归一state 到[-1,1]
        #state = 2 * (state - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low) - 1
        return torch.as_tensor(state, dtype=torch.float32).to(self.device)

    def step(self, state, action):
        with torch.no_grad(): # 如果是环境模型有梯度，则要加上这个
            # 反归一action
            #action = torch.clamp(action, -1.0, 1.0) # 先去掉 裁剪
            #action = np.clip(action.cpu().numpy(), -1.0, 1.0)
            #action = action * (self.env.action_space.high - self.env.action_space.low) / 2 + (self.env.action_space.high + self.env.action_space.low) / 2
            
            state = torch.as_tensor(state, dtype=torch.float32)
            ## 注意 pi
            next_state, = self.env.P_model.pi(torch.cat([state, action], dim=-1).unsqueeze(0),return_mean=self._val)
            #next_state = torch.clamp(next_state.squeeze(0), -1.0, 1.0)

            self.step_count += 1
            if self.step_count >= self.horizon:
                self.step_count = 0
                done = True
            else:
                done = False

            reward = 0
            info = {}
            #info['terminate'] = torch.as_tensor(terminate, dtype=torch.float32).to(self.device)

            #next_state = 2 * (next_state - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low) - 1
            return (torch.as_tensor(next_state, dtype=torch.float32).to(self.device),
                    torch.as_tensor(reward, dtype=torch.float32).to(self.device),
                    torch.as_tensor(done, dtype=torch.float32).to(self.device),
                    info)