import numpy as np
import torch
'''
    这里将act_dim 和action_dim 区分开来
    1维离散空间 act_dim = 1  action_dim = 离散空间的维度       即 [0]
    3维离散空间 act_dim = 1  action_dim = 离散空间的维度 ** 3
    1维连续空间 act_dim = 1  action_dim = 1
    3维连续空间 act_dim = 3  action_dim = 3                   即 [0,0,0]
'''

class Buffer:
    """replay buffer for each agent"""

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity = int(capacity)

        self.obs = np.zeros((capacity, obs_dim))    # batch_size x state_dim
        self.actions = np.zeros((capacity, act_dim))  # batch_size x action_dim
        self.rewards = np.zeros(capacity)            # just a tensor with length: batch
        self.next_obs = np.zeros((capacity, obs_dim))  # batch_size x state_dim
        self.dones = np.zeros(capacity, dtype=bool)    # just a tensor with length: batch_size

        self._index = 0
        self._size = 0

        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        """ add an experience to the memory """ #
        self.obs[self._index] = obs
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.dones[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        obs = self.obs[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_obs = self.next_obs[indices]
        dones = self.dones[indices]

        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        obs = torch.as_tensor(obs,dtype=torch.float32).to(self.device)  # torch.Size([batch_size, state_dim])
        actions = torch.as_tensor(actions,dtype=torch.float32).to(self.device) # torch.Size([batch_size, action_dim])
        rewards = torch.as_tensor(rewards,dtype=torch.float32).reshape(-1,1).to(self.device)  # torch.Size([batch_size]) -> torch.Size([batch_size, 1])
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = torch.as_tensor(next_obs,dtype=torch.float32).to(self.device)  # torch.Size([batch_size, state_dim])
        dones = torch.as_tensor(dones,dtype=torch.float32).reshape(-1,1).to(self.device)

        return obs, actions, rewards, next_obs, dones

    # __len__ is a magic method in Python 可以让对象实现len()方法
    def __len__(self):
        return self._size
    
##

class PER_Buffer:
    '''
    原理:优先级经验回放
    论文名:PRIORITIZED EXPERIENCE REPLAY 论文链接：https://arxiv.org/pdf/1511.05952 
    论文提出了两种方法: 1. rank-based 2. proportional 这里仅仅实现了第2种方法
    公式: 优先级P = (p + epsilon) ** alpha
    1. 优先级越高，被采样的概率越大
    2. alpha 控制优先级的程度 0-1 0 完全随机 1 完全按优先级采样
    3. beta 控制采样的程度 0-1 0 不修正 1 完全修正采样偏差
    4. epsilon 防止优先级为0
    5. beta_increment 逐渐增加beta
    6. is_weight 重要性采样权重 为了修正因为优先级采样带来的偏差
    alpha = 0 beta = 1 等价于均匀采样 论文给出的参考超参数为alpha=0.6 beta=0.4
    '''
    def __init__(self, capacity, obs_dim, act_dim, device, alpha=0.1, beta=1, beta_increment=0.001, epsilon=0.01):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.device = device

        self.sumtree = SumTree(capacity)
        self.buffer = Buffer(capacity, obs_dim, act_dim, device)

    def add(self, obs, action, reward, next_obs, done): 
        '''
        第一条数据的优先级为1.0，后续数据的优先级为最大优先级,来保证所有的经验至少被采样一次
        '''
        max_priority = 1.0 if len(self.buffer) == 0 else self.sumtree.max()  
        self.sumtree.add(self.buffer._index, max_priority)
        self.buffer.add(obs, action, reward, next_obs, done)
        
        

    def sample(self, batch_size):
        '''
        分成batch_size个段，每个段均匀采样采样一个数s,减少采样偏差
        '''
        batch_indices = np.zeros(batch_size, dtype=np.int64) #整数类型
        priorities = np.zeros(batch_size, dtype=np.float32)
        segment = self.sumtree.sum() / batch_size   

        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            p,buffer_index = self.sumtree.get(s)
            priorities[i] = p
            batch_indices[i]= buffer_index

        sampling_probabilities = priorities / self.sumtree.sum() # 采样概率 #公式：P(i) = p(i) / sum(p(i))
        sampling_probabilities = np.clip(sampling_probabilities, 1e-7, None) # 防止出现0概率 #None 表示上限无限制
        is_weight = (len(self.buffer) * sampling_probabilities) ** (-self.beta)  # 重要性采样 # 公式：(N*P(i))**(-beta)
        is_weight /= is_weight.max() # 归一化
        is_weight = torch.as_tensor(is_weight, dtype=torch.float32).to(self.device)

        return batch_indices, is_weight
    
    def update_priorities(self, indices, td_error):
        priorities = (np.abs(td_error) + self.epsilon) ** self.alpha # 公式：P = (|TD| + epsilon) ** alpha
        for idx, priority in zip(indices, priorities):
            self.sumtree.add(idx, priority)

    def __len__(self):
        return len(self.buffer)

class SumTree:
    '''
    树索引idx:
         0         -> 存储优先级之和
        / \
      1     2
     / \   / \
    3   4 5   6    -> 存储转换的优先级
    0   1 2   3    -> 对应的buffer中数据索引 
    树存储优先级 7个优先级                            例: tree_index=4 
    buffer存储数据 4个数据 看作存储在树最后一层        buffer_index = tree_index - capacity + 1 = 4 - 4 + 1 = 1
    '''
    def __init__(self, capacity):
        self.capacity = capacity = int(capacity)
        self.tree = np.zeros(2 * capacity - 1) # 共2n-1个节点

    def add(self, buffer_index, priority):
        '''
        目的: 将优先级添加到树节点
        '''
        tree_index = buffer_index + self.capacity - 1
        self.update(tree_index, priority)

    def update(self, idx, priority):
        '''
        实现sumtree: 更新树节点-> 递归更新父节点
        例：索引5 和6的值都被加到了索引2的值上
        '''
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        '''
        逻辑：1. 从根节点开始
                2. 如果s小于左子树，则idx = 左子树
                3. 如果s大于左子树，则s -= 左子树，idx = 右子树
                4. 直到叶子节点
        返回：优先级，buffer_index
        '''
        idx = 0 
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree): 
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        buffer_idx = idx - self.capacity + 1
        return self.tree[idx], buffer_idx #

    def sum(self): # 优先级之和
        return self.tree[0]
    
    def max(self): # 最大优先级
        return np.max(self.tree[-self.capacity:])
    

class Buffer_for_PPO_d:
    """replay buffer for each agent
    无 action_log_probs  adv_dones for PPO_d.py
    """

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity = int(capacity)

        self.obs = np.zeros((capacity, obs_dim))    # batch_size x state_dim
        self.actions = np.zeros((capacity, act_dim))  # batch_size x action_dim
        self.rewards = np.zeros(capacity)            # just a tensor with length: batch
        self.next_obs = np.zeros((capacity, obs_dim))  # batch_size x state_dim
        self.dones = np.zeros(capacity, dtype=bool)    # just a tensor with length: batch_size

        self._index = 0
        self._size = 0

        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        """ add an experience to the memory """ #
        self.obs[self._index] = obs
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.dones[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        obs = self.obs[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_obs = self.next_obs[indices]
        dones = self.dones[indices]

        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        obs = torch.as_tensor(obs,dtype=torch.float32).to(self.device)  # torch.Size([batch_size, state_dim])
        actions = torch.as_tensor(actions,dtype=torch.float32).to(self.device) # torch.Size([batch_size, action_dim])
        rewards = torch.as_tensor(rewards,dtype=torch.float32).reshape(-1,1).to(self.device)  # torch.Size([batch_size]) -> torch.Size([batch_size, 1])
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = torch.as_tensor(next_obs,dtype=torch.float32).to(self.device)  # torch.Size([batch_size, state_dim])
        dones = torch.as_tensor(dones,dtype=torch.float32).reshape(-1,1).to(self.device)

        return obs, actions, rewards, next_obs, dones

    # __len__ is a magic method in Python 可以让对象实现len()方法
    def __len__(self):
        return self._size
    
    # 清空buffer
    def clear(self):
        self._index = 0
        self._size = 0

    def all(self):
        obs = torch.as_tensor(self.obs,dtype=torch.float32).to(self.device)  # torch.Size([batch_size, state_dim])
        actions = torch.as_tensor(self.actions,dtype=torch.float32).to(self.device) # torch.Size([batch_size, action_dim])
        rewards = torch.as_tensor(self.rewards,dtype=torch.float32).reshape(-1,1).to(self.device)  # torch.Size([batch_size]) -> torch.Size([batch_size, 1])
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = torch.as_tensor(self.next_obs,dtype=torch.float32).to(self.device)  # torch.Size([batch_size, state_dim])
        dones = torch.as_tensor(self.dones,dtype=torch.float32).reshape(-1,1).to(self.device)
        
        return obs, actions, rewards, next_obs, dones
    
class Buffer_for_PPO:
    """replay buffer for each agent"""

    def __init__(self, capacity, obs_dim, act_dim, device, trick = None):
        self.capacity = capacity = int(capacity)

        self.obs = np.zeros((capacity, obs_dim))    # batch_size x state_dim
        self.actions = np.zeros((capacity, act_dim))  # batch_size x action_dim
        self.rewards = np.zeros(capacity)            # just a tensor with length: batch
        self.next_obs = np.zeros((capacity, obs_dim))  # batch_size x state_dim
        self.dones = np.zeros(capacity, dtype=bool)    # just a tensor with length: batch_size
        if trick is not None and trick['decaystd']:
            self.action_log_probs = np.zeros((capacity)) 
        else:
            self.action_log_probs = np.zeros((capacity, act_dim)) 
        self.adv_dones = np.zeros(capacity, dtype=bool)    

        self._index = 0
        self._size = 0

        self.device = device

    def add(self, obs, action, reward, next_obs, done, action_log_probs, adv_done):
        """ add an experience to the memory """ #
        self.obs[self._index] = obs
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.dones[self._index] = done

        self.action_log_probs[self._index] = action_log_probs
        self.adv_dones[self._index] = adv_done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    # __len__ is a magic method in Python 可以让对象实现len()方法
    def __len__(self):
        return self._size
    
    # 清空buffer
    def clear(self):
        self._index = 0
        self._size = 0

    def all(self):
        obs = torch.as_tensor(self.obs,dtype=torch.float32).to(self.device)  # torch.Size([batch_size, state_dim])
        actions = torch.as_tensor(self.actions,dtype=torch.float32).to(self.device) # torch.Size([batch_size, action_dim])
        rewards = torch.as_tensor(self.rewards,dtype=torch.float32).reshape(-1,1).to(self.device)  # torch.Size([batch_size]) -> torch.Size([batch_size, 1])
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = torch.as_tensor(self.next_obs,dtype=torch.float32).to(self.device)  # torch.Size([batch_size, state_dim])
        dones = torch.as_tensor(self.dones,dtype=torch.float32).reshape(-1,1).to(self.device)

        actions_log_probs = torch.as_tensor(self.action_log_probs,dtype=torch.float32).to(self.device) # torch.Size([batch_size, action_dim])
        adv_dones = torch.as_tensor(self.adv_dones,dtype=torch.float32).reshape(-1,1).to(self.device)
        
        return obs, actions, rewards, next_obs, dones , actions_log_probs, adv_dones