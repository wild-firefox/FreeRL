import numpy as np
import torch

from DQN import DQN
from DQN import get_env 
from DQN import dis_to_con

import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
'''
pip install imageio[ffmpeg]

'''
'''
    评估模型 -- 和模型保存在同一个地方
    一条线是goal_return,一条线是evaluate_return,还一条为平滑曲线
    gym 环境目标分数 https://github.com/openai/gym/blob/master/gym/envs/__init__.py 若无此参数则越高越好
    reward_threshold = env.reward_threshold
'''
def plot_evaluate(evaluate_return,goal_return,model_dir,smooth_rate=0.9):
    if goal_return:
        plt.plot([goal_return]*len(evaluate_return))
    plt.plot(evaluate_return)
    smooth_return = []
    for i in range(len(evaluate_return)):
        if i == 0:
            smooth_return.append(evaluate_return[i])
        else:
            smooth_return.append(smooth_rate*smooth_return[-1]+(1-smooth_rate)*evaluate_return[i])
    plt.plot(smooth_return)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Evaluate')
    plt.legend(['goal_return','evaluate_return','smooth_return'])
    # 保存
    plt.savefig(os.path.join(model_dir,"evaluate.png"))

def render(env_name,policy,action_dim,dis_to_con_b,model_dir):
    env = gym.make(env_name,render_mode="rgb_array")
    episode = np.random.randint(args.max_episodes)
    obs,info = env.reset(seed = episode)
    frames = []
    done = False
    while not done:
        frame = env.render()
        frames.append(frame)
        action = policy.select_action(obs)
        action_ = action
        if dis_to_con_b and isinstance(env.action_space, gym.spaces.Box):
            action_ = dis_to_con(action, env, action_dim)
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        done = terminated or truncated
        obs = next_obs
    env.close()
    # 保存gif 
    imageio.mimsave(os.path.join(model_dir,"evaluate.gif"),frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 评估模型位置
    parser.add_argument("--results_dir", type=str, default="./DQN_file/results/")
    parser.add_argument("--env_name", type=str, default="FrozenLake-v1")
    parser.add_argument("--folder_name", type=str, default="DQN_3") #
    # 种子和评估次数设置
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max_episodes", type=int, default=100) #
    # 注意要和训练时一致
    parser.add_argument("--dis_to_con_b", type=bool, default=True) # dqn 默认为True
    args = parser.parse_args()

    env,dim_info, max_action, is_continuous = get_env(args.env_name,args.dis_to_con_b)
    action_dim = dim_info[1]
    # 结果文件夹
    results_dir =  args.results_dir if args.results_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)),'./results')
    model_dir = os.path.join(results_dir, args.env_name,args.folder_name) 

    policy = DQN.load(dim_info,model_dir=model_dir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 以不同seed评估100次
    evaluate_return = []
    for e in range(args.max_episodes):
        obs,info = env.reset(seed=e)
        episode_reward = 0
        done = False
        while not done:
            action = policy.select_action(obs)
            action_ = action
            if args.dis_to_con_b and isinstance(env.action_space, gym.spaces.Box):
                action_ = dis_to_con(action, env, action_dim)  
            next_obs, reward,terminated, truncated, infos = env.step(action_) 
            done = terminated or truncated
            episode_reward += reward
            obs = next_obs
        # episode finished
        evaluate_return.append(episode_reward)
        print(f"Episode {e+1} Reward: {episode_reward}")

    # plot
    env_spec = gym.spec(args.env_name)
    goal_return = env_spec.reward_threshold if env_spec.reward_threshold else None
    plot_evaluate(evaluate_return,goal_return,model_dir,smooth_rate=0.9)

    # 随机挑选一个episode进行展示动画 并保存gif
    render(args.env_name,policy,action_dim,args.dis_to_con_b,model_dir)

