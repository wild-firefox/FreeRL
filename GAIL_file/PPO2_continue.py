
'''
使用一种全新的PPO代码结构
暂时适用：continue
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
import copy
from PPO2_utils import  mlp, Memory, gymEnvWrapper # PolicyEnvWrapper, load_npz


class Actor(nn.Module):
    def __init__(self, config, s_dim: int, hidden_dims ,a_dim: int, act=None, out_act=None, dropout=0.,layernorm=True):
        super().__init__()
        self.config = config
        self.backbone = mlp(s_dim, hidden_dims, a_dim, act=act ,out_act=out_act, dropout=dropout,layernorm=layernorm)

        self.log_std = nn.Parameter(torch.zeros(1, a_dim))

    def forward(self, s: torch.Tensor):
        # 这一步等价于原来的 l1、l2、mean_layer 连在一起
        mean = self.backbone(s)
        mean = torch.tanh(mean)  # 把动作均值压到 [-1, 1]
        if s.dim() == 1:
            log_std = self.log_std.squeeze(0)
        else:
            log_std = self.log_std.expand_as(mean)
        # log_std 仍然是单独的参数，只在这里展开
        #log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, self.config['PPO']['log_std_min'], self.config['PPO']['log_std_max'])
        std = torch.exp(log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, config, s_dim: int, hidden_dims ,a_dim: int, act=None, out_act=None, dropout=0.,layernorm=True):
        super().__init__()
        self.config = config
        self.backbone = mlp(s_dim, hidden_dims, a_dim, act=act ,out_act=out_act, dropout=dropout,layernorm=layernorm)

    def forward(self, s: torch.Tensor):
        v_out = self.backbone(s)
        return v_out

    
class PolicyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        act = getattr(nn, config['PPO']['activation'])()
        self.P = Actor(config, config['s_dim'] , 2*[config['PPO']['policy_mlp_dim']], config['a_dim'],
                       act=act,dropout = config['PPO']['dropout'], layernorm=config['PPO']['layernorm'])
        self.V = Critic(config, config['s_dim'] , 2*[config['PPO']['policy_mlp_dim']], 1,
                        act=act,dropout = config['PPO']['dropout'] ,  layernorm=config['PPO']['layernorm'])

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

        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_gpu', False) else "cpu")
        
        self.P_model = PolicyModel(config).to(self.device)
        self.P = self.P_model.P
        self.V = self.P_model.V
        self.pi = self.P_model.pi
        self.value = self.P_model.value
        
        ## 优化器
        # 优化器写成一个但是分别设置两个lr
        # if self.config.get('D', {'gp_coef':0}).get('gp_coef', 0) > 0:
        #     betas = [0.5, 0.9]
        # else:
        betas = [0.9, 0.999]

        self.optim = torch.optim.Adam(
            [
                {'params': self.P.parameters() , 'lr': config['PPO']['p_lr'] , 'betas' : betas},
                {'params': self.V.parameters(), 'lr': config['PPO']['v_lr'] ,'betas' : betas},
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
        checkpoint = torch.load(path,weights_only=False)
        ## 解析存储地址
        self.config['log_dir'] = os.path.dirname(path)

        self.P.load_state_dict(checkpoint['P_state_dict'])
        self.V.load_state_dict(checkpoint['V_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
        torch.set_rng_state(checkpoint['cpu_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])

        iteration = checkpoint.get('iteration', 0)
        return iteration

    def save_model(self, path: str = None,iteration: int =0, name=''):
        '''
        iteration: 
            'best': 当前训练的episode数 以0开始
            'final': 总训练episode数 = 之后读取时要继续训练的开始episode数
        '''

        name = f'_{name}' if name != '' else ''
        self.model_pth = os.path.join(self.config["log_dir"] , f'model{name}.pth')
        path = path if path is not None else self.model_pth 

        P_model = copy.deepcopy(self.P_model).to('cpu')
        
        torch.save({
            'iteration': iteration,  # <--- 新增：保存当前步数
            'P_state_dict': P_model.P.state_dict(),
            'V_state_dict': P_model.V.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'cpu_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state()
        }, path)
        del P_model

    
    def GAE(self,rewards, values, masks_1, masks_2, gamma, lam, last_value=None, MC=False,):
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
        if last_value is None:
            last_value = torch.tensor(0.0).to(self.device)

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
    
    def PPO_learn(self,batch,last_value=None):
        batch_states = batch['states'] # [batch, s_dim]
        batch_actions = batch['actions'] # [batch, a_dim]
        rewards = batch['rewards']  # [batch, ]

        with torch.no_grad():

            # 计算价值
            value = self.value(batch_states).squeeze(-1)
            # 计算旧策略 log_prob
            old_dist = self.pi(batch_states, return_dist=True)
            p_log_probs = old_dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)

        # 计算优势函数
        ## mask 处理
        mask_1 = torch.ones_like(rewards) # 最好是使用mask_1 来处理 terminate
        mask_2 = batch['masks_2'] # [batch, ]

        advantages, returns = self.GAE(rewards, value, mask_1, mask_2, gamma=self.config['PPO']['gamma'], lam=self.config['PPO']['lam'],last_value=last_value)        
        advantages = advantages.reshape(-1,1)
        returns = returns.reshape(-1,1)
        

        ppo_mini_batch_size = self.config['PPO']['mini_batch_size']

        for k in range(self.config['PPO']['K_epochs']):
            # 随机打乱数据顺序,
            new_index = torch.randperm(batch_states.shape[0])
            indexs = [new_index[i : i + ppo_mini_batch_size] for i in range(0, batch_states.shape[0], ppo_mini_batch_size)]
            for index in indexs:
                mini_batch_states = batch_states[index]
                mini_batch_actions = batch_actions[index]
                mini_log_probs = p_log_probs[index]   ### 
                mini_advantages = advantages[index]
                mini_returns = returns[index]

                # 计算新的log概率
                p_dist = self.pi(mini_batch_states, return_dist=True)
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
                value_pred = self.value(mini_batch_states) # [batch,1]
                value_loss = F.mse_loss(value_pred, mini_returns)

                value_loss = self.config['PPO']['value_loss_coef'] * value_loss

                loss = policy_loss + value_loss

                # 更新所有网络
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.P.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.V.parameters(), 0.5)
                self.optim.step()

    
    def _init_logging(self ,resume: bool = False, start_iteration: int =0):
        '''
        resume: 是否是续训。如果是 True，则复用现有的 log_dir。
        '''
        if not resume:
            # 新训练：创建新目录
            current_time = time.strftime("%Y%m%d-%H%M%S")
            ## 当前文件地址的父文件夹
            current_dir = os.path.dirname(os.path.abspath(__file__)) + '/' 
            self.config["log_dir"] = current_dir + self.config["log_dir"] + '/' + self.config['algo'] + '/' + current_time
            self.config["log_dir"] = os.path.abspath(self.config["log_dir"])
        else:
            # 续训：log_dir 已经在 load_model 里被设置成了 checkpoint 所在的目录
            print(f"Resuming logging in existing directory: {self.config['log_dir']}")
        
        os.makedirs(self.config["log_dir"] , exist_ok=True)
        # purge_step: 如果 TensorBoard 发现步数冲突，会清除该步数之后的记录，防止曲线乱跳（均会清除start_iteration之后的内容）
        self.writer = SummaryWriter(log_dir=self.config["log_dir"] ,purge_step=start_iteration) 
        
        print("Logging to:", self.config["log_dir"])
        

        self.max_reward = -float('inf')
        self.eval_history = []

        # 保存配置文件
        self.save_config()
    
    def save_config(self, path: str= None):
        path = path if path is not None else f"{self.config['log_dir']}/config.json"
        import json
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)

    
    def explore_env(self, env_wrapper):

        #while episode_num < max_episodes:
        while True:
            self.step += 1

            # 与环境交互一步
            state_tensor = torch.as_tensor(self.state, dtype=torch.float32)
            action = self.pi(state_tensor.unsqueeze(0)).squeeze(0).detach() #.cpu().numpy()
            next_state, reward, done, info = env_wrapper.step(self.state, action)

            # mask: 1 - done，用于 GAE 中的 masks_2
            mask = 1.0 - done
            self.memory.push(state_tensor, action, reward, mask)

            self.state = next_state
            self.episode_reward += reward

            # episode 结束：统计一下回报，重新 reset
            if done:
                self.writer.add_scalar('Episode Reward', self.episode_reward, self.episode_num)
                self.train_return.append(self.episode_reward)
                self.episode_num += 1
                print(f"Episode {self.episode_num}/{self.max_episodes}, Reward: {self.episode_reward}")
                
                if (self.episode_num) >= self.max_episodes:
                    return None
                self.state = env_wrapper.reset()
                self.episode_reward = 0.0
            
            if self.step % self.horizon == 0:
                mem = self.memory.sample()
                batch = {
                    'states': torch.stack(mem.state).to(self.device),
                    'actions': torch.stack(mem.action).to(self.device),
                    'rewards': torch.stack(mem.reward).to(self.device),
                    'masks_2': torch.stack(mem.mask).to(self.device),
                }
                return batch

    def train_init(self, env_wrapper, num_episodes, model_path: str = None):
        """训练前的初始化工作"""
        if model_path is not None and os.path.exists(model_path):
            start_iter = self.load_model(model_path)
            resume = True
        else:
            resume = False
            start_iter = 0

        self._init_logging(resume=resume, start_iteration=start_iter)

        self.horizon = self.config['PPO']['horizon']
        self.max_episodes = num_episodes

        self.episode_num = start_iter
        self.step = 0
        self.episode_reward = 0.0
        self.train_return = []
        
        self.state = env_wrapper.reset()
        self.memory = Memory(fields=('state', 'action', 'reward', 'mask'))

    def eval_and_save(self, env_wrapper):
        """在训练循环中评估和保存模型"""
        if self.episode_num % self.config['PPO']['eval_interval'] == 0 and self.episode_num > 0:
            avg_reward_step = self.evaluate(env_wrapper, self.config['PPO']['eval_episodes'])
            self.writer.add_scalar('Eval/Average Return', avg_reward_step, self.episode_num)
            self.eval_history.append((self.episode_num, avg_reward_step))
            if avg_reward_step > self.max_reward:
                self.max_reward = avg_reward_step
                self.save_model(name='best', iteration=self.episode_num)
                self.writer.add_scalar('Eval/Best Return', self.max_reward, self.episode_num)
                print(f"New best model saved with average return: {self.max_reward}")

    def train_post(self, env_wrapper):
        """训练结束后的收尾工作"""
        self.save_model(name='final', iteration=self.max_episodes) 
        self.evaluate(env_wrapper, 10, for_best_model_eval=True,for_eval_data=self.config['collect_eval_data'])

    
    def train(self, env_wrapper, num_episodes, model_path: str = None):
        """PPO 独立训练的完整流程"""
        import copy
        self.train_env = env_wrapper
        self.eval_env = copy.deepcopy(env_wrapper)

        self.train_init(self.train_env, num_episodes, model_path)

        while self.episode_num < self.max_episodes:
            batch = self.explore_env(self.train_env)
            if batch is None:
                break

            last_value = self.value(torch.as_tensor(self.state, dtype=torch.float32).unsqueeze(0)).squeeze().to(self.device)
            self.PPO_learn(batch, last_value)
            
            self.eval_and_save(self.eval_env)

            self.memory.clear()
            
        self.train_post(self.eval_env)
                
    
    def evaluate(self, env_wrapper, num_episodes: int,for_best_model_eval=False,best_model_path=None,for_eval_data=False) -> float:
        if for_best_model_eval:
            best_model_path = best_model_path if best_model_path is not None else os.path.join(self.config["log_dir"] , 'model_best.pth')
            self.load_model(best_model_path)
            env_wrapper.reproducible = False  # 评估最好模型时不固定随机种子
        
        if for_eval_data:
            states = []
            actions = []
            env_wrapper.reproducible = False 
        
        total_reward = []
        total_step = []
        for episode in range(num_episodes):
            state = env_wrapper.reset()
            done = False
            episode_reward = 0.0
            episode_steps = 0
            
            while not done:

                state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                action = self.pi(state_tensor,return_mean=True).squeeze(0).detach()
                if for_eval_data:
                    states.append(state.cpu().numpy())  
                    actions.append(action.cpu().numpy())
                next_state, reward, done, info = env_wrapper.step(state, action)
                episode_reward += reward
                episode_steps += 1
                state = next_state

                
            total_reward.append(episode_reward.cpu().numpy())
            total_step.append(episode_steps)
        avg_reward_step = [reward/step for reward, step in zip(total_reward, total_step)]

        if for_eval_data:
            np.savez(os.path.join(self.config["log_dir"] , 'eval_data.npz'), states=np.array(states), actions=np.array(actions),index = np.array(total_step))

        if for_best_model_eval:
            episode_avg_rewards = sum(total_reward) / num_episodes
            np.savez(os.path.join(self.config["log_dir"] , 'eval_rewards.npz'), episode_rewards=total_reward, episode_avg_rewards=episode_avg_rewards)
            return episode_avg_rewards , total_reward  

        return np.mean(avg_reward_step)
    
if __name__ == "__main__":

    '''
    python -m GAIL_file.PPO2
    '''
    from config import PPO_COBFIG

    env = gym.make('Pendulum-v1')
    env_wrapper = gymEnvWrapper(env, PPO_COBFIG)

    ppo = PPO(PPO_COBFIG)
    ppo.train(env_wrapper, num_episodes=1000)

    #ppo.evaluate(env_wrapper, num_episodes=10, for_best_model_eval=True,best_model_path='logs/Pendulum-v1/GAIL/20251218-111642/model_best.pth')
    ###
