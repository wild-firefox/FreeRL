import numpy as np
import torch

from DDPG import DDPG
from DDPG import get_env 

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
    if goal_return:
        plt.legend(['goal_return','evaluate_return','smooth_return'])
    else:
        plt.legend(['evaluate_return','smooth_return'])
    # 保存
    plt.savefig(os.path.join(model_dir,"evaluate.png"))

def render(env_name,policy,model_dir):
    '''随机挑选一个episode并保存gif'''
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
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        done = terminated or truncated
        obs = next_obs
    env.close()
    # 保存gif 
    imageio.mimsave(os.path.join(model_dir,"evaluate.gif"),frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 评估模型位置 results/env_name/algorithm_trick_n
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--env_name", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--folder_name", type=str, default="DDPG_simple_1") # 模型文件夹名 model名 + trick名
    # 环境参数
    parser.add_argument("--max_action", type=float, default=None)
    # 种子和评估次数设置
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max_episodes", type=int, default=100) #
    ## 是否保存gif
    parser.add_argument("--save_gif", type=bool, default=False)
    # 注意要和训练时一致
    parser.add_argument("--is_dis_to_con", type=bool, default=False) # dqn 默认为True
    parser.add_argument("--policy_name", type=str, default='DDPG_simple')   
    ## trick和folder_name一致 (尽管有些trick在评估时不会用到,但不少是改变模型结构的会用到)                           
    parser.add_argument("--supplement", type=dict, default={'weight_decay':True,'OUNoise':True,'ObsNorm':False,'net_init':True,'Batch_ObsNorm':True})
    parser.add_argument("--trick", type=dict, default=None)  
    args = parser.parse_args()
    if args.policy_name == 'DDPG_simple' or args.supplement == {'weight_decay':False,'OUNoise':False,'ObsNorm':False,'net_init':False,'Batch_ObsNorm':False}:
        args.policy_name = 'DDPG_simple'
        args.supplement = {'weight_decay':False,'OUNoise':False,'ObsNorm':False,'net_init':False,'Batch_ObsNorm':False}
    print(args)
    print('Algorithm:',args.policy_name)

    
    ## 环境配置
    env,dim_info, max_action, is_continue = get_env(args.env_name,args.is_dis_to_con)
    max_action = max_action if max_action is not None else args.max_action
    action_dim = dim_info[1]
    print(f'Env:{args.env_name}  obs_dim:{dim_info[0]}  action_dim:{dim_info[1]}  max_action:{max_action}  max_episodes:{args.max_episodes}')
    
    ## 随机数种子(cpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ### cuda
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random Seed:',args.seed)

    # 模型文件夹 - 读取
    results_dir =  args.results_dir if args.results_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)),'./results')
    model_dir = os.path.join(results_dir, args.env_name,args.folder_name) 
    print(f'model_dir: {model_dir}')

    policy = DDPG.load(dim_info,is_continue = is_continue ,model_dir=model_dir,trick = args.trick,supplement = args.supplement)

    if args.supplement['ObsNorm']:
        mean ,std = np.load(os.path.join(model_dir, f"{args.policy_name}_running_mean_std.npy"))
    if args.supplement['Batch_ObsNorm']:
        mean ,std = np.load(os.path.join(model_dir, f"{args.policy_name}_running_mean_std_batch_size.npy"))
    # 以不同seed评估100次
    evaluate_return = []
    for e in range(args.max_episodes):
        obs,info = env.reset(seed=e)
        if args.supplement['ObsNorm'] or args.supplement['Batch_ObsNorm']:
            obs = (obs - mean) / (std + 1e-8)
        episode_reward = 0
        done = False
        while not done:
            action = policy.evaluate_action(obs)
            action_ = action 
            next_obs, reward,terminated, truncated, infos = env.step(action_) 
            if args.supplement['ObsNorm'] or args.supplement['Batch_ObsNorm']:
                next_obs = (next_obs - mean) / (std + 1e-8)
            done = terminated or truncated
            episode_reward += reward
            obs = next_obs
        # episode finished
        evaluate_return.append(episode_reward)
        print(f"Episode {e+1} Reward: {episode_reward}")

    # save plot
    env_spec = gym.spec(args.env_name)
    goal_return = env_spec.reward_threshold if env_spec.reward_threshold else None
    plot_evaluate(evaluate_return,goal_return,model_dir,smooth_rate=0.9)
    
    # save gif
    if args.save_gif:
        render(args.env_name,policy,action_dim,args.is_dis_to_con,model_dir)

