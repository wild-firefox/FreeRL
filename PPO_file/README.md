常用代码：
```
PPO.py
PPO_with_tricks.py (加入了7个tricks)
PPO_std_decay.py (PPO变体 只训练mean,std衰减版)
```
对比代码：
```
PPO_d.py
PPO_no_minibatch.py
```
复现失败代码；
```
PPO_with_tricks(lose_centering).py(加入reward_centering后效果并不好，推测应该是复现失败)  
```
使用了Pendulum-v1和BipedalWalker-v3两个环境进行测试，效果均不好。