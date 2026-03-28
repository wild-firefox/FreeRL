import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from torch import nn
from PPO2_utils import  mlp, gymEnvWrapper # PolicyEnvWrapper, load_npz
from GAIL_utils import ExpertDataset, create_expert_loader
from PPO2 import PPO


'''
gp_coef >0 为wgan版本
参考代码：https://github.com/nv-tlabs/ASE/blob/main/ase/learning/amp_agent.py#L468

'''

class Discriminator(nn.Module):
    def __init__(self, config, s_dim: int, hidden_dims ,a_dim: int, act=None, out_act=None, dropout=0.,layernorm=True):
        super().__init__()
        self.config = config
        self.backbone = mlp(s_dim, hidden_dims, a_dim, act=act ,out_act=out_act, dropout=dropout,layernorm=layernorm)

    def forward(self, s: torch.Tensor):
        v_out = self.backbone(s)
        if self.config['D']['gp_coef'] > 0:
            return v_out
        
        return torch.sigmoid(v_out)
    

class GAIL:
    def __init__(self, config):
        self.global_seed(config['seed'])
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')


        act = getattr(nn, config['D']['activation'])()
        self.D_model = Discriminator(config, config['s_dim']+config['a_dim'], 2*[config['D']['d_mlp_dim']], 1, act=act, dropout=config['D']['dropout'],layernorm=config['D']['layernorm']).to(self.device)
        
        if self.config.get('D', False).get('gp_coef', 0) > 0:
            betas = [0.5, 0.9]
        else:
            betas = [0.9, 0.999]
        
        self.D_optimizer = torch.optim.Adam(self.D_model.parameters(), lr=config['D']['d_lr'],betas = betas)

        # 拥有一个 PPO 实例
        self.ppo = PPO(config)


    def global_seed(self, seed: int):
        ''' 设置随机种子 '''
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def compute_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.config['D']['gp_coef'] > 0:
                disc_logits = self.D_model(torch.cat([states, actions], dim=-1))
                prob = 1 / (1 + torch.exp(-disc_logits)) 
                disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
                disc_r *= 2 #self._disc_reward_scale
                rewards = disc_r.squeeze(-1)
            else:
                d_out = self.D_model(torch.cat([states, actions], dim=-1))
                rewards = -torch.log(1 - d_out + 1e-8).squeeze(-1)
        return rewards
    
    def trian_D(self, expert_states: torch.Tensor, expert_actions: torch.Tensor,
               policy_states: torch.Tensor, policy_actions: torch.Tensor) -> float:
        # 计算判别器输出
        expert_d_out = self.D_model(torch.cat([expert_states, expert_actions], dim=-1))
        policy_d_out = self.D_model(torch.cat([policy_states, policy_actions], dim=-1))

        # 计算损失
        if self.config['D']['gp_coef'] > 0:

            expert_loss = F.binary_cross_entropy_with_logits(expert_d_out, torch.ones_like(expert_d_out))
            policy_loss = F.binary_cross_entropy_with_logits(policy_d_out, torch.zeros_like(policy_d_out))
            d_loss = 0.5 * (expert_loss + policy_loss)

            #######
            obs_demo = torch.cat([expert_states, expert_actions], dim=-1).requires_grad_(True)
            d_out_demo = self.D_model(obs_demo)
            disc_demo_grad = torch.autograd.grad(d_out_demo, obs_demo, grad_outputs=torch.ones_like(d_out_demo),
                                             create_graph=True, retain_graph=True, only_inputs=True)
            disc_demo_grad = disc_demo_grad[0]
            disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
            disc_grad_penalty = torch.mean(disc_demo_grad)
            ######

            d_loss += 5 * disc_grad_penalty

            w_dis =  (torch.mean(expert_d_out) - torch.mean(policy_d_out))
            expert_prob = torch.sigmoid(expert_d_out).mean().item()
            policy_prob = torch.sigmoid(policy_d_out).mean().item()


        else:

            expert_loss = F.binary_cross_entropy(expert_d_out, torch.ones_like(expert_d_out))
            policy_loss = F.binary_cross_entropy(policy_d_out, torch.zeros_like(policy_d_out))
            d_loss = expert_loss + policy_loss

            expert_prob = expert_d_out.mean().item()
            policy_prob =  policy_d_out.mean().item()


        # 优化判别器
        self.D_optimizer.zero_grad()
        d_loss.backward()
        self.D_optimizer.step()

        return d_loss.item(), expert_prob, policy_prob, w_dis.item() if self.config['D']['gp_coef'] > 0 else 0.0
    
    def train(self, env_wrapper, expert_loader, num_episodes, model_path: str = None):
        """GAIL 的核心训练逻辑"""

        # 初始化 PPO (包括日志、加载模型等)
        self.ppo.train_init(env_wrapper, num_episodes, model_path)
        
        # 从 PPO 实例同步状态
        self.episode_num = self.ppo.episode_num


        d_step = 1

        if self.config['D']['gp_coef'] > 0:
            d_step = 1

        while self.ppo.episode_num < self.ppo.max_episodes:
            # 1. PPO 探索环境，收集一个 horizon 的数据
            for _ in range(d_step):
                batch = self.ppo.explore_env(env_wrapper)
                # 清空 PPO 的 buffer
                self.ppo.memory.clear()
                if batch is None:
                    break
                
                policy_states = batch['states']
                policy_actions = batch['actions']

                # 2. 训练判别器 D
                expert_states, expert_actions = next(expert_loader)
                expert_states = expert_states.to(self.device)
                expert_actions = expert_actions.to(self.device)
                d_loss, expert_prob, policy_prob, w_dis = self.trian_D(expert_states, expert_actions, policy_states, policy_actions)
            self.ppo.writer.add_scalar('Discriminator/Loss', d_loss, self.ppo.episode_num)
            self.ppo.writer.add_scalar('Discriminator/Expert', expert_prob, self.ppo.episode_num)
            self.ppo.writer.add_scalar('Discriminator/Policy', policy_prob, self.ppo.episode_num)
            if self.config['D']['gp_coef'] > 0:
                self.ppo.writer.add_scalar('Discriminator/Wasserstein_Dis', w_dis, self.ppo.episode_num)

            if batch is None:
                break

            # 3. 使用 D 计算新的奖励
            gail_rewards = self.compute_reward(policy_states, policy_actions)
            
            # 4. 用新的奖励替换 batch 中的环境奖励
            batch['rewards'] = gail_rewards

            # 5. PPO 使用这个修改后的 batch 进行学习
            last_value = self.ppo.value(torch.as_tensor(self.ppo.state, dtype=torch.float32).unsqueeze(0)).squeeze()
            self.ppo.PPO_learn(batch, last_value)

            # 6. PPO 负责评估和保存模型
            self.ppo.eval_and_save(env_wrapper)



        # 训练结束，PPO 保存最终模型
        self.ppo.train_post(env_wrapper)
    
  


if __name__ == "__main__":

    '''
    python -m GAIL_file.PPO2
    '''
    from config import GAIL_COBFIG

    gail = GAIL(GAIL_COBFIG)

    env = gym.make('Pendulum-v1')
    env_wrapper = gymEnvWrapper(env, GAIL_COBFIG)

    # 创建专家数据集和无限 loader
    expert_dataset = ExpertDataset(GAIL_COBFIG['expert_data_path'])
    expert_loader = create_expert_loader(expert_dataset, GAIL_COBFIG)

    

    gail.train(env_wrapper, expert_loader, 2000)
