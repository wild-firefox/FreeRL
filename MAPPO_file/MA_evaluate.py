import numpy as np
import torch

from MAPPO import MAPPO
from MAPPO import get_env 

import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import imageio

# 其他
import pickle
import importlib
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
    ## 如果是多智能体：evaluate_return是一个字典,取所有智能体的平均值
    if isinstance(evaluate_return,dict):
        evaluate_return = list(np.sum([values for values in evaluate_return.values()],axis=0)/len(evaluate_return))
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

def render(env_name,policy,model_dir,env_agent_n,args):
    '''随机挑选一个episode并保存gif'''
    # 动态导入环境
    module = importlib.import_module(f'pettingzoo.mpe.{env_name}')
    print('env_agent_n or num_good:',env_agent_n) 
    if env_agent_n is None: #默认环境
        env = module.parallel_env(max_cycles=25, continuous_actions=True,render_mode="rgb_array")
    elif env_name == 'simple_spread_v3' or 'simple_adversary_v3': 
        env = module.parallel_env(max_cycles=25, continuous_actions=True, N = env_agent_n,render_mode="rgb_array")
    elif env_name == 'simple_tag_v3': 
        env = module.parallel_env(max_cycles=25, continuous_actions=True, num_good= env_agent_n, num_adversaries=3,render_mode="rgb_array")
    elif env_name == 'simple_world_comm_v3':
        env = module.parallel_env(max_cycles=25, continuous_actions=True, num_good= env_agent_n, num_adversaries=4,render_mode="rgb_array")
    env.reset()
    #env = gym.make(env_name,render_mode="rgb_array")

    if args.trick['ObsNorm']:
        obs_norm = pickle.load(open(model_dir + "/obs_norm.pkl", "rb"))

    episode = np.random.randint(args.max_episodes)
    obs,info = env.reset(seed = episode)
    if args.trick['ObsNorm'] :
        obs = {agent_id: (obs[agent_id] - obs_norm[agent_id][0]) / (obs_norm[agent_id][1] + 1e-8) for agent_id in env_agents}
    frames = []
    done = {agent_id: False for agent_id in env_agents}
    while not any(done.values()):
        frame = env.render()
        frames.append(frame)
        action = policy.evaluate_action(obs)
        action_ = action
        ## pettingzoo环境需要将动作范围[-1,1]转换为[0,1]
        action_ = {agent_id: (action_[agent_id] + 1) / 2 for agent_id in env_agents} # [-1,1] -> [0,1] 
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        if args.trick['ObsNorm'] :
            next_obs = {agent_id: (next_obs[agent_id] - obs_norm[agent_id][0]) / (obs_norm[agent_id][1] + 1e-8) for agent_id in env_agents}
        done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
        obs = next_obs
    env.close()
    for frame in frames:
        if frame is None:
            print("Found None in frames list")
    # 保存gif 
    imageio.mimsave(os.path.join(model_dir,"evaluate.gif"),frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 评估模型位置 results/env_name/algorithm_trick_n
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--env_name", type=str, default="simple_spread_v3")
    parser.add_argument("--folder_name", type=str, default="MAPPO_simple_1") # 模型文件夹名 model名 + trick名
    # 环境参数
    parser.add_argument("--N", type=int, default=5) # 环境中智能体数量 默认None 这里用来对比设置
    parser.add_argument("--continuous_actions", type=bool, default=True) #
    # 种子和评估次数设置
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max_episodes", type=int, default=100) #
    ## 是否保存gif
    parser.add_argument("--save_gif", type=bool, default=True)
    # 注意要和训练时一致
    parser.add_argument("--policy_name", type=str, default='MAPPO_simple')   
    ## trick和folder_name一致 (尽管有些trick在评估时不会用到,但不少是改变模型结构的会用到)                           
    parser.add_argument("--trick", type=dict, default={'adv_norm':False,
                                                    'ObsNorm':False,
                                                    'reward_norm':False,'reward_scaling':False,    # or
                                                    'orthogonal_init':False,'adam_eps':False,'lr_decay':False, # 原代码中设置为False
                                                    # 以上均在PPO_with_tricks.py中实现过
                                                    'ValueClip':False,'huber_loss':False,
                                                    'LayerNorm':False,'feature_norm':False,
                                                    })  
    args = parser.parse_args()

    # 检查 reward_norm 和 reward_scaling 的值
    if args.trick['reward_norm'] and args.trick['reward_scaling']:
        raise ValueError("reward_norm 和 reward_scaling 不能同时为 True")
    
    if  args.policy_name == 'MAPPO' or ((args.trick['lr_decay'] is False ) and all(value  for key, value in args.trick.items() if key not in ['reward_norm','lr_decay'])) :
        args.policy_name = 'MAPPO'
        for key in args.trick.keys():
            if key not in ['reward_norm','lr_decay']:
                args.trick[key] = True
            else:
                args.trick[key] = False
    
    if args.policy_name == 'MAPPO_simple' or (not any(args.trick.values())) : # if all(value is False for value in args.trick.values()):
        args.policy_name = 'MAPPO_simple'
        for key in args.trick.keys():
            args.trick[key] = False
    print(args)
    print('Algorithm:',args.policy_name)

    ## 环境配置
    env,dim_info,max_action,is_continue = get_env(args.env_name, env_agent_n = args.N, continuous_actions = args.continuous_actions)
    print(f'Env:{args.env_name}  dim_info:{dim_info}  max_action:{max_action}  max_episodes:{args.max_episodes}')
    
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

    policy = MAPPO.load(dim_info,is_continue = is_continue ,model_dir=model_dir,trick = args.trick,)

    if args.trick['ObsNorm']:
        obs_norm = pickle.load(open(model_dir + "/obs_norm.pkl", "rb"))

    # 以不同seed评估100次
    env_agents = [agent_id for agent_id in env.agents]
    evaluate_return = {agent_id: [] for agent_id in env_agents}
    for e in range(args.max_episodes):
        obs,info = env.reset(seed=e)
        if args.trick['ObsNorm'] :
            obs = {agent_id: (obs[agent_id] - obs_norm[agent_id][0]) / (obs_norm[agent_id][1] + 1e-8) for agent_id in env_agents}
        episode_reward = {agent_id: 0 for agent_id in env_agents}
        done = {agent_id: False for agent_id in env_agents}
        while not any(done.values()):
            action = policy.evaluate_action(obs)
            action_ = action
            ## pettingzoo环境需要将动作范围[-1,1]转换为[0,1]
            if is_continue:
                action_ = {agent_id: (action_[agent_id] + 1) / 2 for agent_id in env_agents} # [-1,1] -> [0,1] 
            next_obs, reward,terminated, truncated, infos = env.step(action_) 
            if args.trick['ObsNorm'] :
                next_obs = {agent_id: (next_obs[agent_id] - obs_norm[agent_id][0]) / (obs_norm[agent_id][1] + 1e-8) for agent_id in env_agents}
            done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
            episode_reward = {agent_id: episode_reward[agent_id] + reward[agent_id] for agent_id in env_agents}
            obs = next_obs
        # episode finished
        for agent_id in env_agents:
            evaluate_return[agent_id].append(episode_reward[agent_id])
        print(f"Episode {e+1} Reward: {episode_reward}")

    # save plot
    plot_evaluate(evaluate_return,goal_return= None,model_dir = model_dir,smooth_rate=0.9)
    
    # save gif
    if args.save_gif:
        render(args.env_name,policy,env_agent_n = args.N,model_dir = model_dir,args=args)

