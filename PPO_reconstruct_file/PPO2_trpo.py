
'''
使用一种全新的PPO代码结构
总体逻辑一致，若效果有区别：检查cpu/gpu训练，adv_norm,环境状态归一化。

改一版trpo
'''


import torch
import torch.nn.functional as F
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from torch import nn
import copy
from PPO2_utils import  mlp, Memory, gymEnvWrapper # PolicyEnvWrapper, load_npz
from torch.distributions import Normal, Categorical
from torch.distributions import Independent, kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters

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
    

# 新增：离散动作用的 Actor（输出 logits）
class ActorDiscrete(nn.Module):
    def __init__(self, config, s_dim: int, hidden_dims, a_dim: int):
        super().__init__()
        self.config = config
        # 输出维度为 a_dim (logits)
        self.backbone = mlp(s_dim, hidden_dims, a_dim, act=getattr(nn, config['PPO']['activation'])(), out_act=None,
                            dropout=config['PPO']['dropout'], layernorm=config['PPO']['layernorm'])

    def forward(self, s: torch.Tensor):
        logits = self.backbone(s)  # no tanh
        return logits

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
        # 支持离散/连续分支
        if self.config.get('discrete', False):
            self.P = ActorDiscrete(config, config['s_dim'], 2 * [config['PPO']['policy_mlp_dim']], config['a_dim'])
        else:
            self.P = Actor(config, config['s_dim'], 2 * [config['PPO']['policy_mlp_dim']], config['a_dim'],
                           act=act, dropout=config['PPO']['dropout'], layernorm=config['PPO']['layernorm'])
        self.V = Critic(config, config['s_dim'], 2 * [config['PPO']['policy_mlp_dim']], 1,
                        act=act, dropout=config['PPO']['dropout'], layernorm=config['PPO']['layernorm'])

    def pi(self, s: torch.Tensor, return_dist: bool = False, return_mean: bool = False):
        ''' 策略网络前向传播，支持离散/连续 '''
        if self.config.get('discrete', False):
            logits = self.P(s)
            if return_mean:
                # --- 修改：如果是评估/确定性测试，直接返回概率最大的动作索引 ---
                return torch.argmax(logits, dim=-1)
            dist = Categorical(logits=logits)
            if return_dist:
                return dist
            action = dist.sample()  # shape: [batch] (LongTensor)
            return action
        else:
            mean, std = self.P(s)
            if return_mean:
                return mean
            dist = Normal(mean, std)
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
        if config['algo'] in ('TRPO', 'TRPO_minibatch'):
            self.P.eval()  # 固定 dropout，使采样、计算旧策略和线搜索使用同一策略。
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
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        ## 解析存储地址
        self.config['log_dir'] = os.path.dirname(path)

        self.P.load_state_dict(checkpoint['P_state_dict'])
        self.V.load_state_dict(checkpoint['V_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
        torch.set_rng_state(checkpoint['cpu_rng_state'])
        if torch.cuda.is_available() and checkpoint['cuda_rng_state'] is not None:
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
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state()
        }, path)
        del P_model

    
    def GAE(self,rewards, values, masks_1, masks_2, gamma, lam, last_value=None, MC=False,adv_norm=False):
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
            # 只有一个优势值时，默认的样本标准差没有定义，保留原优势。
            if adv_norm and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            del rewards, masks_1, masks_2, values
        
        return advantages, returns # [batch,1] ,[batch,1]
    
    def PPO_learn(self,batch,last_value=None):
        batch_states = batch['states'] # [batch, s_dim]
        batch_actions = batch['actions'] # [batch, a_dim] or [batch,1] for discrete
        rewards = batch['rewards']  # [batch, ]

        with torch.no_grad():
            value = self.value(batch_states).squeeze(-1)
            old_dist = self.pi(batch_states, return_dist=True)
            # 区分离散/连续 计算旧log_prob
            if self.config.get('discrete', False):
                # batch_actions 存的是 int 或 1D 张量，转换为 long
                acts = batch_actions.view(-1).long()
                p_log_probs = old_dist.log_prob(acts).unsqueeze(-1)  # [batch,1]
                #print(p_log_probs.shape)
            else:
                p_log_probs = old_dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)

        # 计算优势函数
        ## mask 处理
        # 如果batch 有masks_1 这个键
        
        mask_1 = batch['masks_1'] if 'masks_1' in batch else torch.ones_like(rewards) # 最好是使用mask_1 来处理 terminate
        mask_2 = batch['masks_2'] # [batch, ]

        advantages, returns = self.GAE(rewards, value, mask_1, mask_2, 
                                       gamma=self.config['PPO']['gamma'], 
                                       lam=self.config['PPO']['lam'],
                                       last_value=last_value,
                                       adv_norm = self.config['PPO'].get('adv_norm', False))        
        advantages = advantages.reshape(-1,1)
        returns = returns.reshape(-1,1)

        # 三个分支共用采样策略、优势和回报目标。
        if self.config['algo'] == 'TRPO':
            return self.TRPO_learn(batch_states, batch_actions, old_dist, p_log_probs, advantages, returns)
        if self.config['algo'] == 'TRPO_minibatch':
            return self.TRPO_minibatch_learn(batch_states, batch_actions, old_dist, p_log_probs, advantages, returns)
        

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
                if self.config.get('discrete', False):
                    acts_m = mini_batch_actions.reshape(-1).long()  # [batch,1] -> [batch,]
                    new_log_probs = p_dist.log_prob(acts_m).unsqueeze(-1)
                    dist_entropy = p_dist.entropy().unsqueeze(-1).mean()  # entropy per-sample -> mean
                else:
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

    
    def trpo_action_dist(self, states):
        action_dists = self.pi(states, return_dist=True)
        if self.config['discrete']:
            return action_dists
        # 多维连续动作视为一个事件，log_prob 和 KL 都会沿动作维度求和。
        return Independent(action_dists, 1)

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        # 计算黑塞矩阵 H 和一个向量的乘积。H 是平均 KL 在旧策略参数处的二阶导数。
        new_action_dists = self.trpo_action_dist(states)
        kl = kl_divergence(old_action_dists, new_action_dists).mean()  # D_KL(旧策略 || 新策略)
        params = tuple(self.P.parameters())
        kl_grad = torch.autograd.grad(kl, params, create_graph=True)  # 得到 KL 梯度
        kl_grad_vector = parameters_to_vector(kl_grad)
        # KL 梯度先和向量做点积，再对参数求导，得到 H * vector。
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, params)
        grad2_vector = parameters_to_vector(grad2)
        if self.config['discrete']:
            return grad2_vector  # 笔记本离散版：阻尼为 0
        damping = self.config['TRPO']['damping']
        return grad2_vector + damping * vector  # 笔记本连续版：(H + 0.1 I) * vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        # 共轭梯度法求解 Hx = g；连续版用加了阻尼的矩阵代替 H。
        x = torch.zeros_like(grad)   # x 的初始值
        r = grad.clone()             # 残差 r = g - Hx，初始时 x = 0
        p = r.clone()                # 搜索方向
        rdotr = torch.dot(r, r)      # r^T r
        for _ in range(self.config['TRPO']['cg_iters']):
            if rdotr < 1e-10:
                break
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            pHp = torch.dot(p, Hp)
            if not torch.isfinite(pHp) or pHp <= 0:
                break
            alpha = rdotr / pHp      # 沿 p 方向走多远
            x += alpha * p           # 更新解 x
            r -= alpha * Hp          # 更新残差 r
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p         # 更新搜索方向 p
            rdotr = new_rdotr
        return x

    @staticmethod
    def compute_surrogate_obj(action_dists, actions, advantage, old_log_probs):
        # 计算代理目标 J(θ)。每个样本只有一个概率比和一个优势值。
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    @torch.no_grad()
    def line_search(self, states, actions, advantage, old_log_probs, old_action_dists, max_vec):
        old_para = parameters_to_vector(self.P.parameters())
        # 多次更新时，以本次试步前的目标为基准，概率比分母和 KL 仍用采样策略。
        current_dists = self.trpo_action_dist(states)
        old_obj = self.compute_surrogate_obj(current_dists, actions, advantage, old_log_probs)
        alpha = self.config['TRPO']['backtrack_coeff']
        kl_constraint = self.config['TRPO']['max_kl']
        accepted = False
        try:
            for i in range(self.config['TRPO']['backtrack_iters']):  # 线性搜索主循环
                coef = alpha ** i
                new_para = old_para + coef * max_vec  # θ = θ_old + coef * max_vec
                vector_to_parameters(new_para, self.P.parameters())
                new_action_dists = self.trpo_action_dist(states)
                kl_div = kl_divergence(old_action_dists, new_action_dists).mean()
                new_obj = self.compute_surrogate_obj(new_action_dists, actions, advantage, old_log_probs)
                # J(θ) > J(θ_old)，且 D_KL(θ_old || θ) 在上限内，才接受这组参数。
                if (torch.isfinite(new_obj) and torch.isfinite(kl_div)
                        and new_obj > old_obj and kl_div <= kl_constraint):
                    accepted = True
                    return True, kl_div.item()
        finally:
            if not accepted:
                vector_to_parameters(old_para, self.P.parameters())  # 搜索失败，恢复旧策略
        return False, 0.0

    def trpo_max_step(self, states, actions, advantage, old_log_probs, current_action_dists):
        # 先计算 J(θ)，对策略网络求导得到 g，再把各层梯度展成一维。
        action_dists = self.trpo_action_dist(states)
        surrogate_obj = self.compute_surrogate_obj(action_dists, actions, advantage, old_log_probs)
        grads = torch.autograd.grad(surrogate_obj, self.P.parameters())
        obj_grad = parameters_to_vector(grads).detach()

        # 用共轭梯度法求 x ≈ H^(-1)g，并计算 Hd，其中 d 就是这个更新方向 x。
        descent_direction = self.conjugate_gradient(obj_grad, states, current_action_dists)
        Hd = self.hessian_matrix_vector_product(states, current_action_dists, descent_direction)
        curvature = torch.dot(descent_direction, Hd)
        if not torch.isfinite(curvature) or curvature <= 0:
            return None
        kl_constraint = self.config['TRPO']['max_kl']
        max_coef = torch.sqrt(2 * kl_constraint / curvature)  # 初始步长 β
        return descent_direction * max_coef                   # β * x

    def trpo_update_critic(self, states, returns):
        # 更新价值函数。Actor 的 grad 为 None，原有 Adam 只更新 Critic。
        self.optim.zero_grad(set_to_none=True)
        value_loss = F.mse_loss(self.value(states), returns)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.V.parameters(), 0.5)
        self.optim.step()

    def TRPO_learn(self, states, actions, old_action_dists, old_log_probs, advantage, returns):
        if self.config['discrete']:
            actions = actions.flatten().long()
        else:
            old_action_dists = Independent(old_action_dists, 1)
        advantage = advantage.flatten()
        old_log_probs = old_log_probs.flatten()

        max_vec = self.trpo_max_step(states, actions, advantage, old_log_probs, old_action_dists)
        accepted, actual_kl = False, 0.0
        if max_vec is not None:
            accepted, actual_kl = self.line_search(
                states, actions, advantage, old_log_probs, old_action_dists, max_vec)
        self.trpo_update_critic(states, returns)
        return accepted, actual_kl

    def TRPO_minibatch_learn(self, states, actions, old_action_dists, old_log_probs, advantage, returns):
        """像 PPO 一样多轮打乱、分小批；每次 Actor 试步都受整批轨迹的 KL 约束。"""
        if self.config['discrete']:
            actions = actions.flatten().long()
        else:
            old_action_dists = Independent(old_action_dists, 1)
        advantage = advantage.flatten()
        old_log_probs = old_log_probs.flatten()  # 所有轮次共用采样时的分母
        mini_batch_size = self.config['PPO']['mini_batch_size']
        attempts, accepted_steps, actual_kl = 0, 0, 0.0

        for _ in range(self.config['PPO']['K_epochs']):
            indices = torch.randperm(len(states), device=states.device)
            for idx in indices.split(mini_batch_size):
                # 小批梯度决定方向。Fisher 在当前参数处计算，每接受一步就会变化。
                mini_states = states[idx]
                with torch.no_grad():
                    current_dists = self.trpo_action_dist(mini_states)
                max_vec = self.trpo_max_step(
                    mini_states, actions[idx], advantage[idx], old_log_probs[idx], current_dists)
                attempts += 1
                if max_vec is not None:
                    # 在整批状态上比较目标和 KL，避免每个小批都重新获得一份 KL 预算。
                    accepted, kl = self.line_search(
                        states, actions, advantage, old_log_probs, old_action_dists, max_vec)
                    if accepted:
                        accepted_steps += 1
                        actual_kl = kl
                self.trpo_update_critic(mini_states, returns[idx])
        return {'actor_attempts': attempts, 'actor_accepted': accepted_steps,
                'critic_updates': attempts, 'rollout_kl': actual_kl}

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
            # 这里策略输出对离散/连续两类都兼容：
            action = self.pi(state_tensor.unsqueeze(0))
            action_save = action.squeeze(0).detach()
            action_to_env = action_save  # wrapper 负责把离散动作转为整数、连续动作缩放到 Box

            next_state, reward, done, info = env_wrapper.step(self.state, action_to_env)

            mask_1 = 1.0 - info['terminate']
            # mask: 1 - done，用于 GAE 中的 masks_2
            mask_2 = 1.0 - done
            learn_reward = reward
            if done and not info['terminate']:
                # 时间截断仍需 bootstrap。先计入截断末状态的价值，避免 GAE 使用 reset 后的状态。
                with torch.no_grad():
                    learn_reward = reward + self.config['PPO']['gamma'] * self.value(next_state).squeeze(-1)
                mask_1 = torch.zeros_like(mask_1)
            self.memory.push(state_tensor, action_save, learn_reward, mask_1, mask_2)

            self.state = next_state
            self.episode_reward += reward

            # episode 结束：统计一下回报，重新 reset
            if done:
                self.writer.add_scalar('Episode Reward', self.episode_reward, self.episode_num)
                self.train_return.append(self.episode_reward)
                self.episode_num += 1
                print(f"Episode {self.episode_num}/{self.max_episodes}, Reward: {self.episode_reward}")
                
                if (self.episode_num) >= self.max_episodes:
                    if self.config['algo'] not in ('TRPO', 'TRPO_minibatch'):
                        return None
                else:
                    self.state = env_wrapper.reset()
                self.episode_reward = 0.0
            
            if self.step % self.horizon == 0 or self.episode_num == self.max_episodes:
                mem = self.memory.sample()
                batch = {
                    'states': torch.stack(mem.state).to(self.device),
                    'actions': torch.stack(mem.action).to(self.device),
                    'rewards': torch.stack(mem.reward).to(self.device),
                    'masks_1': torch.stack(mem.mask_1).to(self.device),
                    'masks_2': torch.stack(mem.mask_2).to(self.device),
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
        self.memory = Memory(fields=('state', 'action', 'reward', 'mask_1', 'mask_2')) 

    def eval_and_save(self, env_wrapper):
        """在训练循环中评估和保存模型"""
        if self.config['PPO']['eval_interval'] > 0 and self.episode_num % self.config['PPO']['eval_interval'] == 0 and self.episode_num > 0:
            avg_reward = self.evaluate(env_wrapper, self.config['PPO']['eval_episodes'])
            self.writer.add_scalar('Eval/Average Return', avg_reward, self.episode_num)
            self.eval_history.append((self.episode_num, avg_reward))
            if avg_reward > self.max_reward:
                self.max_reward = avg_reward
                self.save_model(name='best', iteration=self.episode_num)
                self.writer.add_scalar('Eval/Best Return', self.max_reward, self.episode_num)
                print(f"New best model saved with average return: {self.max_reward}")

    def train_post(self, env_wrapper):
        """训练结束后的收尾工作"""
        self.save_model(name='final', iteration=self.max_episodes) 
        for_best_model_eval = self.config['PPO']['eval_interval'] > 0 and os.path.exists(
            os.path.join(self.config['log_dir'], 'model_best.pth'))
        self.evaluate(env_wrapper, 10, for_best_model_eval=for_best_model_eval,for_eval_data=self.config['collect_eval_data'])

    
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
            metrics = self.PPO_learn(batch, last_value)
            if self.config['algo'] == 'TRPO_minibatch':
                for key, value in metrics.items():
                    self.writer.add_scalar(f'TRPO_minibatch/{key}', value, self.step)
            
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
        #avg_reward_step = [reward/step for reward, step in zip(total_reward, total_step)]

        if for_eval_data:
            np.savez(os.path.join(self.config["log_dir"] , 'eval_data.npz'), states=np.array(states), actions=np.array(actions),index = np.array(total_step))

        if for_best_model_eval:
            episode_avg_rewards = sum(total_reward) / num_episodes
            np.savez(os.path.join(self.config["log_dir"] , 'eval_rewards.npz'), episode_rewards=total_reward, episode_avg_rewards=episode_avg_rewards)
            return episode_avg_rewards , total_reward  

        return np.mean(total_reward)
    
if __name__ == "__main__":

    '''
    直接专用终端运行
    FreeRL> python PPO_reconstruct_file/PPO2_trpo.py
    '''
    from config import PPO_COBFIG

    

    PPO_COBFIG['algo'] = 'TRPO'  # 可选 'PPO'、'TRPO'、'TRPO_minibatch'
    env = gym.make(PPO_COBFIG['id'])  # 环境在 config.py 的 run_id 里选择
    PPO_COBFIG['discrete'] = isinstance(env.action_space, gym.spaces.Discrete)
    
    env_wrapper = gymEnvWrapper(env, PPO_COBFIG)

    ppo = PPO(PPO_COBFIG)
    ppo.train(env_wrapper, num_episodes=500)

    #ppo.evaluate(env_wrapper, num_episodes=10, for_best_model_eval=True,best_model_path='logs/Pendulum-v1/GAIL/20251218-111642/model_best.pth')
    ###
