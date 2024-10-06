import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
"""
    绘制学习曲线 -- 从保存模型的位置读取 保存在learning_curves + env_name文件夹下
    实线为平均值，阴影部分为标准差，
    这里是在随机种子seed=0,10,100下的进行的实验。可以自设随机种子
"""
def policy_trick_name(policy_name, trick=None):
    if trick is None or not any(trick.values()):
        prefix = policy_name + '_'
    else:
        prefix = policy_name + '_'
        for key in trick.keys():
            if trick[key]:
                prefix += key + '_'
    return prefix[:-1]
'''
十种常见颜色                                                                    蓝绿色                    紫红色              
1.'b': 'blue',(0,0,1) ; 2.'g': 'green',(0,1,0) ; 3.'r': 'red',(1,0,0) ; 4.'c': 'cyan',(0,1,1) ; 5.'m': 'magenta',(1,0,1) ; 6.'y': 'yellow',(1,1,0) ; 7.'k': 'black',(0,0,0) ; 8.'w': 'white',(1,1,1) ; 9.'orange':(1,0.5,0) ; 10.'purple':(0.5,0,0.5)
默认matplotlib颜色 C0-C9 C0:蓝色 C1:橙色 C2:绿色 C3:红色 C4:紫色 C5:棕色 C6:粉红色 C7:灰色 C8:黄色 C9:青色 和上述的颜色在RGB颜色空间是不一样的,如C0: (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)  
'''
def plot_learning_curves(rewards,title,label,color ='C0' ,window_size=10):
    # rewards shape (seed_num,episodes)
    # 平滑处理 , np.convolve通过卷积核进行平滑处理 mode='valid' 保证输出和输入的维度一致
    smoothed_rewards = np.array([np.convolve(reward, np.ones(window_size)/window_size, mode='valid') for reward in rewards]) 
    # 计算平均值和标准差
    mean_rewards = np.mean(smoothed_rewards, axis=0)
    std_rewards = np.std(smoothed_rewards, axis=0)

    # 绘制图表
    plt.figure(figsize=(10,6))
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.plot(mean_rewards,label=label,color=color)
    plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, color=color,alpha=0.3)
    plt.grid() ## 显示网格
    plt.legend(loc='upper left')
    # 保存
    plt.savefig(os.path.join(learning_curves_env_dir, f"{label}.png"))

def plot_compare_learning_curves(rewards,title,labels,colors ,window_size=20):
    # rewards shape (len(compare_tricks),seed_num,episodes) 
    # 绘制图表
    plt.figure(figsize=(10,6))
    for i in range(rewards.shape[0]):
        # 平滑处理 , np.convolve通过卷积核进行平滑处理 mode='valid' 保证输出和输入的维度一致
        smoothed_rewards = np.array([np.convolve(reward, np.ones(window_size)/window_size, mode='valid') for reward in rewards[i]]) 
        # 计算平均值和标准差
        mean_rewards = np.mean(smoothed_rewards, axis=0)
        std_rewards = np.std(smoothed_rewards, axis=0)
        plt.plot(mean_rewards,label=labels[i],color = colors[i])
        plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, color=colors[i],alpha=0.1)

    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Return')

    plt.grid() ## 显示网格
    plt.legend(loc='upper left')
    # 保存
    plt.savefig(os.path.join(learning_curves_env_dir, f"compare_{rewards.shape[0]}.png"))

''' 
环境见：CartPole-v1,Pendulum-v1,MountainCar-v0,MountainCarContinuous-v0;LunarLander-v2,BipedalWalker-v3;FrozenLake-v1
https://github.com/openai/gym/blob/master/gym/envs/__init__.py
FrozenLake-v1 在5000episode下比较好
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="MountainCarContinuous-v0") # /results/env_name
    parser.add_argument("--results_dir", type=str, default=None) # None (其他算法下直接填None) /results
    parser.add_argument("--learning_curves", type=str, default=None) # None /learning_curves
    parser.add_argument("--seed_num", type=int, default=1) #评估的随机种子数
    ## 注意：是否比较其他算法
    parser.add_argument("--is_compare", type=str, default=False) # 多个算法比较
    # trick 评估 算法 or 算法 + trick
    parser.add_argument("--policy_name", type=str, default='DDPG_simple')  # 算法 # DDPG_simple
    parser.add_argument("--trick", type=dict, default=None) 
    args = parser.parse_args()

    # 结果文件夹 -取出
    results_dir =  args.results_dir if args.results_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)),'./results')
    results_env_dir = os.path.join(results_dir, args.env_name) 

    # 学习曲线文件夹 -存入
    learning_curves = args.learning_curves if args.learning_curves else os.path.join(os.path.dirname(os.path.abspath(__file__)),'./learning_curves')
    learning_curves_env_dir = os.path.join(learning_curves, args.env_name)
    os.makedirs(learning_curves_env_dir) if not os.path.exists(learning_curves_env_dir) else None
    
    # 读取文件夹名: 算法 + trick 
    policy_trick = policy_trick_name(args.policy_name, args.trick)

    '''读取rewards
    # 第一版读取：文件夹名和seed数可以根据自己的实际情况修改
    # reward_1 = np.load(os.path.join(results_env_dir, f"{policy_trick}_1", f"{args.policy_name}_seed_0.npy")) 
    # reward_2 = np.load(os.path.join(results_env_dir, f"{policy_trick}_2", f"{args.policy_name}_seed_10.npy"))
    # reward_3 = np.load(os.path.join(results_env_dir, f"{policy_trick}_3", f"{args.policy_name}_seed_100.npy"))
    # rewards = np.array([reward_1, reward_2, reward_3]) # ->(3,episodes)

    第二版本如下所示: 以较少的行数表示，缺点:f"{policy_trick}_{i+1}" 只能按顺序执行。
    '''
    
    rewards = []
    for i in range(args.seed_num):
        rewards_dir = os.path.join(results_env_dir,f"{policy_trick}_{i+1}")    
        re = np.load(os.path.join(rewards_dir,[f for f in os.listdir(rewards_dir) if f.endswith(".npy")][0])) #找到.npy文件并加载
        rewards.append(re)
    rewards = np.array(rewards) #->(args.seed_num,episodes)
    
    # 保存rewards
    np.save(os.path.join(learning_curves_env_dir, f"{policy_trick}_{rewards.shape[0]}_seed.npy"), rewards)

    # 画图 单个算法(mean-std 图) 并保存
    plot_learning_curves(rewards,title = args.env_name,label = f'{policy_trick}')

    # 画对比图 默认已有 无trick的DQN
    if args.is_compare:
        compare_tricks = ['DQN','DQN_Double','DQN_Dueling','DQN_PER','DQN_Noisy','DQN_N_Step','DQN_Categorical','DQN_Rainbow'] # 指定要对比的算法名称 (algorithm/results/env_name/文件夹名（algorithm_trick）)
        colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','b','g','r','c','m','y','k','w','orange','purple'] # 共20种颜色
        rewards_compare = []
        for i in range(len(compare_tricks)):
            rewards_c = np.load(os.path.join(learning_curves_env_dir, f"{compare_tricks[i]}_{args.seed_num}_seed.npy"))
            rewards_compare.append(rewards_c)
        rewards_compare = np.array(rewards_compare) # ->(len(compare_tricks),args.seed_num,episodes)
        plot_compare_learning_curves(rewards_compare,title = args.env_name,labels = compare_tricks,colors = colors)



