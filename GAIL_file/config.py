import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将需要复用的值存入变量
run_id = 'Pendulum-v1' # 'CartPole-v1' ; 'Pendulum-v1'

s_dim ={
    'Pendulum-v1': 3,
    'CartPole-v1': 4,
}

a_dim = {
    'Pendulum-v1': 1,
    'CartPole-v1': 2,
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
    'seed': 42,
    'env_reproducible': True, # True 时 训练效果可复现
    'collect_eval_data': True,

    'num_workers': 1,
    'use_gpu': True,  # 确保启用GPU

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

        'p_lr': 1e-3,
        'v_lr': 1e-3,
        'horizon': 2048,
        'gamma': 0.99,
        'lam': 0.95,
        'clip_epsilon': 0.2,
        'K_epochs': 10,
        'value_loss_coef': 1,
        'ent_weight': 0.01,
        'mini_batch_size': 64,
        'eval_interval': 2,
        'eval_episodes':1,
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