## 参考
* [动手学强化学习](https://hrl.boyuai.com/)
* [elegentRL](https://github.com/AI4Finance-Foundation/ElegantRL)
* [DRL-code-pytorch](https://github.com/Lizhi-sjtu/DRL-code-pytorch)
* [easy-rl](https://github.com/datawhalechina/easy-rl/blob/master/notebooks/DQN.ipynb)
* [maddpg-pettingzoo-pytorch](https://github.com/Git-123-Hub/maddpg-pettingzoo-pytorch)
* [深度强化学习](https://github.com/DeepRLChinese/DeepRL-Chinese/blob/master/04_dqn.py)
* [reinforcement-learning-algorithm](https://github.com/Git-123-Hub/reinforcement-learning-algorithm)
* [DRL-Pytorch](https://github.com/XinJingHao/DRL-Pytorch/)
* [cleanRL](https://github.com/vwxyzjn/cleanrl)

目的是写出像TD3作者那样简单易懂的DRL代码,  
由于参考了ElegentRL和Easy的库,from easy to elegent 故起名为freeRL,  
free也是希望写出的代码可以随意的,自由的从此代码移植到自己的代码上。  
库的介绍：[【FreeRL】我的深度学习库构建思想](https://blog.csdn.net/weixin_56760882/article/details/142176797)


## python环境
```python
python 3.11.9
torch 2.3.1+cu121
gymnasium[all] 0.29.1
pygame 0.25.2 # 这个版本和gymnasium[all]0.29.1兼容
pettingzoo 1.24.3
```
## 效果
用DQN算法在LunarLander-v2环境下训练500个轮次的3个seed的效果：线为均值，阴影为方差
![alt text](DQN_file/learning_curves/LunarLander-v2/DQN.png)
用 seed = 0 训练的模型评估,评估100个不同的seed的结果
![alt text](DQN_file/results/LunarLander-v2/DQN_1/evaluate.png)
随机选择其中一个seed的结果，渲染环境
![alt text](DQN_file/results/LunarLander-v2/DQN_1/evaluate.gif)

## 显示训练
训练时，使用tensorboard来显示实时的学习曲率。

在DQN_file（算法）文件夹下，D:FreeRL/DQN_file 终端里输入：
tensorboard --logdir=results/env_name
在跳出的http://localhost:6008/ 按住ctrl点击进入就行。

tensorboard保存的文件events.out.tfevents.和模型的位置一致。

注：上述生成图的画图函数(in plot_learning_curves.py)已修改，由卷积滑动smooth造成的横坐标显示不全改成了可以显示全的指数平滑。

## 已实现
以下均实现了cuda和cpu切换。
* DQN 1.DQN   
2.DQN_Double  
3.DQN_Dueling  
4.DQN_PER  
5.DQN_Noisy  
6.DQN_N_Step  
7.DQN_Categorical  
8.DQN_Rainbow  

![alt text](image.png)  

* DDPG    
其中包括4细节trick实现：in DDPG ,   discrete的2种实现 in DDPG_simple_add_discrete.py 
注：lose为利用gumbel_softmax来实现DDPG_discrete留存的失败实现。   
1.weight_decay  
2.OUNoise  
3.net_init  
4.Batch_ObsNorm or ObsNorm  
* PPO  
其中包括7个trick实现 in PPO_with_tricks.pt,一个变体 in PPO_std_decay.py   
注：PPO_d,PPO_no_minibath为PPO复刻时的对比实验。  
1.adv_norm  
2.ObsNorm or Batch_ObsNorm   
3.reward_norm or reward_scaling  
4.lr_decay    
5.orthogonal_init  
6.adam_eps  
7.tanh    
* TD3
* SAC
包含discrete的一种实现 in SAC_add_discrete.py
* MADDPG  
注：
1.MADDPG_reproduction.py,主要实现论文中说明的两种方式。      
2.MADDPG_simple.py 为MADDPG论文中伪代码的实现,也为DDPG_simple.py的多智能体版本。
3.MADDPG.py 为DDPG.py的多智能体版本。  
其中包括1个trick实现 in MADDPG_simple_with_tricks.py
1.ATT  
* MAPPO 
注：MAPPO 为复刻原代码实现 其中有12个tricks MAPPO_simple 为无tricks的实现。
1.MAPPO_attention.py 加入了attention机制
2.HAPPO.py 为MAPPO基础之上的改进
3.MAT.py 以HAPPO为基础，引入了transformer
* MAAC
1.MASAC.py (自己写的版本,只有continue版本)
2.MAAC_discrete.py (根据论文复刻的版本)


未完善：
* CEM_GD3PG