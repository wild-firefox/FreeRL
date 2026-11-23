"""用三个随机种子训练 PPO 和 TRPO；默认运行三个环境的全部 18 组实验。"""

import argparse
from copy import deepcopy

import gymnasium as gym

from config import PPO_COBFIG
from PPO2_trpo import PPO
from PPO2_utils import gymEnvWrapper

EPISODES = {"CartPole-v1": 500, "Pendulum-v1": 500, "BipedalWalker-v3": 500}
SEEDS = (0, 1, 2)
ALGORITHMS = ("TRPO", "PPO")


def train_one(env_id, algo, seed, num_episodes):
    with gym.make(env_id) as env:
        config = deepcopy(PPO_COBFIG)
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        config.update(
            id=env_id,
            algo=algo,
            seed=seed,
            discrete=discrete,
            s_dim=env.observation_space.shape[0],
            a_dim=int(env.action_space.n) if discrete else env.action_space.shape[0],
            num_episodes=num_episodes,
            log_dir=f"logs/{env_id}/seed_{seed}",
            env_reproducible=False,
        )
        # 只在启动时给环境设种子。之后 wrapper 用 reset(seed=None)，
        # 各回合沿同一随机序列取初始状态；重跑同一 seed 可复现该序列。
        env.reset(seed=seed)
        agent = PPO(config)  # 原构造函数使用同一个 seed 初始化 PyTorch 和 NumPy。
        print(f"\n{env_id} | {algo} | seed={seed} | episodes={num_episodes}", flush=True)
        try:
            agent.train(gymEnvWrapper(env, config), num_episodes=num_episodes)
        finally:
            if hasattr(agent, "writer"):
                agent.writer.close()
            if hasattr(agent, "eval_env"):
                agent.eval_env.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=(*ALGORITHMS, "TRPO_minibatch"), help="仅训练指定算法")
    parser.add_argument("--env", choices=EPISODES, help="仅训练指定环境")
    args = parser.parse_args()

    for env_id in ([args.env] if args.env else EPISODES):
        for algo in ([args.algo] if args.algo else ALGORITHMS):
            for seed in SEEDS:
                train_one(env_id, algo, seed, EPISODES[env_id])


if __name__ == "__main__":
    main()
