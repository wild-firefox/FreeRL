
'''
使用一种全新的PPO结构
暂时适用：continue
PPO2_continue.py的train函数的对照组
'''


import torch
from typing import Tuple
import math
import torch.nn.functional as F
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from torch import nn
import importlib
from PPO2_utils import  mlp, Memory


class Actor(nn.Module):
    def __init__(self, s_dim: int, hidden_dims ,a_dim: int, act=None, out_act=None, dropout=0.,layernorm=True):
        super().__init__()

        self.backbone = mlp(s_dim, hidden_dims, a_dim, act=act ,out_act=out_act, dropout=dropout,layernorm=layernorm)

        self.log_std = nn.Parameter(torch.zeros(1, a_dim))

    def forward(self, s: torch.Tensor):
        # 这一步等价于原来的 l1、l2、mean_layer 连在一起
        mean = self.backbone(s)
        mean = torch.tanh(mean)  # 把动作均值压到 [-1, 1]

        # log_std 仍然是单独的参数，只在这里展开
        log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, self.config['log_std_min'], self.config['log_std_max'])
        std = torch.exp(log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, s_dim: int, hidden_dims ,a_dim: int, act=None, out_act=None, dropout=0.,layernorm=True):
        super().__init__()
        self.backbone = mlp(s_dim, hidden_dims, a_dim, act=act ,out_act=out_act, dropout=dropout,layernorm=layernorm)

    def forward(self, s: torch.Tensor):
        v_out = self.backbone(s)
        return v_out

    
class PolicyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.P = Actor(config['s_dim'] , 2*[config['policy_mlp_dim']], config['a_dim'],act=nn.ReLU(),layernorm=False)
        self.V = Critic(config['s_dim'] , 2*[config['policy_mlp_dim']], 1,act=nn.ReLU(),layernorm=False)

    def pi(self, s: torch.Tensor,return_dist: bool = False, return_mean:bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' 策略网络前向传播 '''
        mean, std = self.P(s)

        if return_mean:
            return mean

        dist = torch.distributions.Normal(mean, std)
        if return_dist:
            return dist

        action = dist.sample()
        return action
    
    
    def value(self, s: torch.Tensor):
        v_out = self.V(s)
        return v_out
    

class PPO:
    def __init__(self, config):
        self.global_seed(config['seed'])
        self.config = config
        
        self.P_model = PolicyModel(config)
        self.P = self.P_model.P
        self.V = self.P_model.V
        
        ## 优化器
        # 优化器写成一个但是分别设置两个lr
        self.optim = torch.optim.Adam(
            [
                {'params': self.P.parameters() , 'lr': config['PPO']['p_lr']},
                {'params': self.V.parameters(), 'lr': config['PPO']['v_lr']},
            ]
        )

    def global_seed(self, seed: int):
        ''' 设置随机种子 '''
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def load_model(self, path: str):
        ''' 加载模型参数 '''
        self.P_model.load_state_dict(torch.load(path))
        print(f"Loaded model from {path}")

    def save_model(self, path: str):
        pass

    
    def GAE(self,rewards, values, masks_1, masks_2, gamma, lam, last_value, MC=False,):
        ''' 计算GAE优势函数 
        一般来说
        mask_1 (td_delta) 表示 1-terminate
        mask_2 (gae)     表示  1-(terminate or truncation)
        令done = terminate or truncation 
        truncation 由这里的dataset['index'] 提供

        如果不提供 terminate 则都认为是0, 即 masks_1 全是1  # 这里不提供 若要提供 则需要另外学习这个terminate函数
        如果不提供 last_value 则可以写成是tensor(0)  # 这里逻辑提供了last_value

        return 
        {
            1.advantages 来更新P           使用GAE(λ)
            2.returns    来更新V           这里写两种更新V的方式1.TD(λ) 2.Monte-Carlo(TD(1))
        }
        '''

        with torch.no_grad():
            values = torch.cat([values, last_value.unsqueeze(0)])  # batch -> batch+1
            advantages = torch.zeros_like(rewards)
            last_advantage = 0
            if not MC:
                for t in reversed(range(len(rewards))):
                    delta = rewards[t] + gamma * values[t + 1] * masks_1[t] - values[t]
                    advantages[t] = last_advantage = delta + gamma * lam * masks_2[t] * last_advantage

                returns = advantages + values[:-1]
            else:

                future_return = 0
                returns = torch.zeros_like(rewards)
                
                for t in reversed(range(len(rewards))):
                    # 计算 TD(λ) 的优势函数
                    delta = rewards[t] + gamma * values[t + 1] * masks_1[t] - values[t]
                    advantages[t] = last_advantage = delta + gamma * lam * masks_2[t] * last_advantage

                    # 计算 Monte-Carlo 的 returns
                    future_return = rewards[t] + gamma * future_return * masks_2[t]
                    returns[t] = future_return

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            del rewards, masks_1, masks_2, values
        
        return advantages, returns # [batch,1] ,[batch,1]
    
    def PPO_learn(self,batch,last_value):
        batch_states = batch['states'] # [batch, s_dim]
        batch_actions = batch['actions'] # [batch, a_dim]
        rewards = batch['rewards']  # [batch, ]

        with torch.no_grad():

            # 计算价值
            value = self.P_model.value(batch_states).squeeze(-1)
            # 计算旧策略 log_prob
            old_dist = self.P_model.pi(batch_states, return_dist=True)
            p_log_probs = old_dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)

        # 计算优势函数
        ## mask 处理
        mask_1 = torch.ones_like(rewards)
        mask_2 = batch['masks_2'] # [batch, ]

        advantages, returns = self.GAE(rewards, value, mask_1, mask_2, gamma=self.config['PPO']['gamma'], lam=self.config['PPO']['lam'],last_value=last_value)        
        advantages = advantages.reshape(-1,1)
        returns = returns.reshape(-1,1)
        

        ppo_mini_batch_size = self.config['PPO']['mini_batch_size']

        for k in range(self.config['PPO']['K_epochs']):
            # 随机打乱数据顺序,
            new_index = torch.randperm(batch_states.shape[0])
            indexs = [new_index[i : i + ppo_mini_batch_size] for i in range(0, batch_states.shape[0], ppo_mini_batch_size)]
            # for i in range(optim_iter_num):
            #     index = new_index[i * ppo_mini_batch_size : min((i + 1) * ppo_mini_batch_size, batch_states.shape[0])]
            for index in indexs:
                mini_batch_states = batch_states[index]
                mini_batch_actions = batch_actions[index]
                mini_log_probs = p_log_probs[index]   ### 
                mini_advantages = advantages[index]
                mini_returns = returns[index]

                # 计算新的log概率
                p_dist = self.P_model.pi(mini_batch_states, return_dist=True)
                new_log_probs = p_dist.log_prob(mini_batch_actions).sum(dim=-1, keepdim=True)
                dist_entropy = (p_dist.entropy().sum(dim=-1)).mean()

                # 计算比例
                ratio = torch.exp(new_log_probs - mini_log_probs)

                # 计算策略损失
                surr1 = ratio * mini_advantages
                eps = self.config['PPO']['clip_epsilon']
                surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * mini_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.config['PPO']['ent_weight'] * dist_entropy

                # 计算价值损失
                value_pred = self.P_model.value(mini_batch_states) # [batch,1]
                value_loss = F.mse_loss(value_pred, mini_returns)

                value_loss = self.config['PPO']['value_loss_coef'] * value_loss

                loss = policy_loss + value_loss

                # 更新所有网络
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.P.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.V.parameters(), 0.5)
                self.optim.step()

    
    def _init_logging(self):
        current_time = time.strftime("%Y%m%d-%H%M%S")
        self.policy_log_dir = self.config["log_dir"] + f'/policy/{current_time}'
        os.makedirs(self.policy_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.policy_log_dir)
        
        print("Logging to:", self.policy_log_dir)
        self.model_pth = os.path.join(self.policy_log_dir, f'model.pth')

        self.max_reward = -float('inf')
        self.eval_history = []

    def train(self, env_wrapper, num_episodes):
        """训练逻辑改成类似 PPO.py：按 step 累积到 horizon 再更新一次。"""
        self._init_logging()

        horizon = self.config['PPO']['horizon']      # 你可以在 config 里加这个字段，或写死一个常数
        max_episodes = num_episodes

        episode_num = 0
        step = 0
        episode_reward = 0.0
        train_return = []
        

        # 初始化环境
        state = env_wrapper.reset()

        # 用 Memory 做 “一个 horizon 的 buffer”
        memory = Memory(fields=('state', 'action', 'reward', 'mask'))

        while episode_num < max_episodes:
            step += 1

            # 与环境交互一步
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action = self.P_model.pi(state_tensor).squeeze(0).detach() #.cpu().numpy()
            next_state, reward, done, info = env_wrapper.step(state, action)

            # mask: 1 - done，用于 GAE 中的 masks_2
            mask = 1.0 - float(done)
            memory.push(state, action, reward, mask)

            state = next_state
            episode_reward += reward

            # episode 结束：统计一下回报，重新 reset
            if done:
                episode_num += 1
                self.writer.add_scalar('Episode Reward', episode_reward, episode_num)
                train_return.append(episode_reward)
                print(f"Episode {episode_num}/{max_episodes}, Reward: {episode_reward}")

                state = env_wrapper.reset()
                episode_reward = 0.0

            # 每 horizon 步更新一次网络
            if step % horizon == 0:
                # 从 memory 里取出一个 horizon 的数据
                mem = memory.sample()
                # 计算最后一个状态的 value，用于 GAE bootstrap
                last_value = self.P_model.value(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze()

                batch = {
                    'states': torch.as_tensor(np.array(mem.state), dtype=torch.float32),
                    'actions': torch.as_tensor(np.array(mem.action), dtype=torch.float32),
                    'rewards': torch.as_tensor(np.array(mem.reward), dtype=torch.float32),
                    'masks_2': torch.as_tensor(np.array(mem.mask), dtype=torch.float32),
                }
                self.PPO_learn(batch, last_value)

                # # 评估当前策略
                if episode_num % self.config['PPO']['eval_interval'] == 0:
                    avg_reward_step = self.evaluate(env_wrapper, self.config['PPO']['eval_episodes'])
                    self.writer.add_scalar('Eval/Average Return', avg_reward_step, episode_num)
                    self.eval_history.append((episode_num, avg_reward_step))
                    # 保存最好模型
                    if avg_reward_step > self.max_reward:
                        self.max_reward = avg_reward_step
                        torch.save(self.P_model.state_dict(), self.model_pth)
                        self.writer.add_scalar('Eval/Best Return', self.max_reward, episode_num)
                        print(f"New best model saved with average return: {self.max_reward}")

                # 清空 buffer，开始下一个 horizon
                memory.clear()
            
    
    def evaluate(self, env_wrapper, num_episodes: int,for_real_env_eval=False) -> float:
        if for_real_env_eval:
            best_model_path = os.path.join(self.policy_log_dir, 'model.pth')
            self.load_model(best_model_path)
        
        total_reward = []
        total_step = []
        for episode in range(num_episodes):
            state = env_wrapper.reset()
            done = False
            episode_reward = 0.0
            episode_steps = 0
            while not done:
                state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                action = self.P_model.pi(state_tensor,return_mean=True).squeeze(0).detach()
                next_state, reward, done, info = env_wrapper.step(state, action)
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
            total_reward.append(episode_reward)
            total_step.append(episode_steps)
        avg_reward_step = [reward/step for reward, step in zip(total_reward, total_step)]
        if for_real_env_eval:
            episode_avg_rewards = sum(total_reward) / num_episodes
            np.savez(os.path.join(self.policy_log_dir, 'real_env_eval_rewards.npz'), episode_rewards=total_reward, episode_avg_rewards=episode_avg_rewards)
            return episode_avg_rewards , total_reward  

        return np.mean(avg_reward_step)
    
