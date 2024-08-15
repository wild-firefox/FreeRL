import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
"""
    绘制学习曲线 -- 从保存模型的位置读取 保存在learning_curves + env_name文件夹下
    实线为平均值，阴影部分为标准差，
    这里是在随机种子seed=0,10,100下的进行的实验。可以自设随机种子
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="LunarLander-v2")
    parser.add_argument("--results_dir", type=str, default="./DQN_file/results/")
    parser.add_argument("--learning_curves", type=str, default="./DQN_file/learning_curves/")
    parser.add_argument("--seed_num", type=int, default=3) #评估的随机种子数
    args = parser.parse_args()

    # 结果文件夹 -取出
    results_dir =  args.results_dir if args.results_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)),'./results')
    results_env_dir = os.path.join(results_dir, args.env_name) 

    # 学习曲线文件夹 -存入
    learning_curves = args.learning_curves if args.learning_curves else os.path.join(os.path.dirname(os.path.abspath(__file__)),'./learning_curves')
    learning_curves_env_dir = os.path.join(learning_curves, args.env_name)
    os.makedirs(learning_curves_env_dir) if not os.path.exists(learning_curves_env_dir) else None
    
    rewards = []
    for i in range(args.seed_num):
        rewards_dir = os.path.join(results_env_dir,f"DQN_{i+1}")
        re = np.load(rewards_dir,[f for f in os.listdir(rewards_dir) if f.endswith(".npy")][0]) #找到.npy文件并加载
        rewards.append(re)

    rewards = np.array(rewards) # ->(3,episodes)

    # 保存rewards
    np.save(os.path.join(learning_curves_env_dir, f"DQN_{rewards.shape[0]}_seed.npy"), rewards)

    # 画图
    # 平滑处理 , 通过卷积核进行平滑处理 
    window_size = 10
    smoothed_rewards = np.array([np.convolve(reward, np.ones(window_size)/window_size, mode='valid') for reward in rewards]) #  mode='valid' 保证输出和输入的维度一致

    # 计算平均值和标准差
    mean_rewards = np.mean(smoothed_rewards, axis=0)
    std_rewards = np.std(smoothed_rewards, axis=0)

    # 绘制图表
    plt.figure(figsize=(10,6))
    plt.title(args.env_name)
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    
    plt.plot(mean_rewards, label='DQN')
    plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
    
    plt.legend(loc='upper left')

    # 保存
    plt.savefig(os.path.join(learning_curves_env_dir, "DQN.png"))
