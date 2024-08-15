## 参考
* [动手学强化学习](https://hrl.boyuai.com/)
* [elegentRL](https://github.com/AI4Finance-Foundation/ElegantRL)
* [DRL-code-pytorch](https://github.com/Lizhi-sjtu/DRL-code-pytorch)
* [easy-rl](https://github.com/datawhalechina/easy-rl/blob/master/notebooks/DQN.ipynb)
* [maddpg-pettingzoo-pytorch](https://github.com/Git-123-Hub/maddpg-pettingzoo-pytorch)
* [深度强化学习](https://github.com/DeepRLChinese/DeepRL-Chinese/blob/master/04_dqn.py)

目的是写出像TD3作者那样简单易懂的DRL代码,  
由于参考了ElegentRL和Easy的库,from easy to elegent 故起名为freeRL,
free也是希望写出的代码可以随意的,自由的从此代码移植到自己的代码上。

## python环境
```python
python 3.11.5
torch 2.3.1+cu121
gymnasium[all] 0.29.1
pygame 0.25.2 # 这个版本和gymnasium[all]0.29.1兼容
```