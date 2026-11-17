import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将需要复用的值存入变量
run_id = 'BipedalWalker-v3' #'BipedalWalker-v3'#'Pendulum-v1' # 'CartPole-v1' ; 'Pendulum-v1'

s_dim ={
    'Pendulum-v1': 3,
    'CartPole-v1': 4,
    'BipedalWalker-v3': 24,
}

a_dim = {
    'Pendulum-v1': 1,
    'CartPole-v1': 2,
    'BipedalWalker-v3': 4,
}

GAIL_COBFIG = {
    # 训练id
    'id': run_id,
    'expert_data_path': rf'{current_dir}\logs\Pendulum-v1\PPO\20251218-214834\eval_data.npz',
    'log_dir': f'logs/{run_id}',

    'algo': 'GAIL',

    'seed': 42,
    'env_reproducible': True, # True 时 训练效果可复现
    'collect_eval_data': False,

    'num_workers': 1,
    'use_gpu': True,  # 确保启用GPU


    's_dim': 3,
    'a_dim': 1,

    
    'PPO': {
        ## liner
        'policy_mlp_dim': 128,
        'activation':'LeakyReLU', # ReLU, Leakyrelu, Elu, Swish
        'dropout':0.0, #0.01,
        'layernorm':False,

        ## actor log_std range
        'log_std_min': -10,
        'log_std_max': 2,

        'p_lr': 1e-4,#1e-3,
        'v_lr': 3e-4,#1e-3,
        'horizon': 2048,
        'gamma': 0.99,
        'lam': 0.95,
        'clip_epsilon': 0.2,
        'K_epochs': 10,
        'value_loss_coef': 1,
        'ent_weight': 0.01,
        'mini_batch_size': 64,
        'eval_interval': 10,
        'eval_episodes':1,
        
        'use_sapo': False,
        'sapo_tau_pos': 1.0,
        'sapo_tau_neg': 1.0,
    },

    'D': {
        'd_mlp_dim': 128,
        'activation':'LeakyReLU', 
        'dropout':0.0,
        'layernorm':False,

        'd_lr': 4e-4,#1e-3,
        'gp_coef': 10.0,  # 加gp时gp_coef:10 为wgan 、不加gp时gp_coef:0

    },
}


PPO_COBFIG = {
    # 训练id
    'id': run_id,
    'log_dir': f'logs/{run_id}',
    'algo': 'PPO',
    'seed': 100,#42,
    'env_reproducible': True, # True 时 训练效果可复现
    'collect_eval_data': True,

    'num_workers': 1,
    'use_gpu': True,  # 确保启用GPU


    'normalize_obs': False, # 是否归一化obs 到[-1,1]，如果环境obs空间有inf边界，建议设置为False
    # pendulum
    # 's_dim': 3,
    # 'a_dim': 1,

    # cartpole
    's_dim': s_dim[run_id],
    'a_dim': a_dim[run_id],
    'discrete': False, # 是否离散动作空间，离散动作空间需要修改ppo的损失函数


    
    'PPO': {
        ## liner
        'policy_mlp_dim': 128,
        'activation':'ReLU', # ReLU, Leakyrelu, Elu, Swish
        'dropout':0.0, #0.01,
        'layernorm':False,

        ## actor log_std range
        'log_std_min': -10,
        'log_std_max': 2,

        ## GAE
        'adv_norm': True, # Trick :advantage normalization

        'p_lr': 3e-4, #1e-3,
        'v_lr': 3e-4, #1e-3,
        'horizon': 2048,
        'gamma': 0.99,
        'lam': 0.95,
        'clip_epsilon': 0.2,
        'K_epochs': 16, #10,
        'value_loss_coef': 1,
        'ent_weight': 0.01,
        'mini_batch_size': 256,#64,
        'eval_interval': 5, #2,
        'eval_episodes':1,
        
        'use_sapo': False,
        'sapo_tau_pos': 15.0,
        'sapo_tau_neg': 15.05,
        
        'use_fipo': True,
        'fipo_decay_rate': 32,
        'fipo_clip_range': 0.2,
        'fipo_clip_high_only': True,
        'fipo_delta': 10.0,
        'fipo_safety_threshold': 4.0,
    },

}


'''
activation 可选参数：
[
    "Threshold",
    "ReLU",
    "RReLU",
    "Hardtanh",
    "ReLU6",
    "Sigmoid",
    "Hardsigmoid",
    "Tanh",
    "SiLU",
    "Mish",
    "Hardswish",
    "ELU",
    "CELU",
    "SELU",
    "GLU",
    "GELU",
    "Hardshrink",
    "LeakyReLU",
    "LogSigmoid",
    "Softplus",
    "Softshrink",
    "MultiheadAttention",
    "PReLU",
    "Softsign",
    "Tanhshrink",
    "Softmin",
    "Softmax",
    "Softmax2d",
    "LogSoftmax",
]
'''