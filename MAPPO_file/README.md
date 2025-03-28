常用代码:
MAPPO:
```
MAPPO.py（只支持continue）
MAPPO_discrete.py
MAPPO_attention.py
```
HAPPO:
```
HAPPO.py（只支持continue）
```
MAT:
```
MAT.py（只支持continue）
MAT_mod_buffer.py(简化buffer相关函数的代码，但是效率一致)
```

已弃用
```
MAPPO(discard).py(更新learn函数逻辑不对，但在env_step(固定种子)效果和MAPPO.py一致,由于之前的实验数据基本都在这个上面运行，所以保留)
HAPPO(discard).py(更新learn函数逻辑不对，但在env_step(固定种子)效果和HAPPO.py一致,由于之前的实验数据基本都在这个上面运行，所以保留)
```

---2025.1.4更新---
将 PPO update 的代码
![alt text](image_assist/image.png)
均改成类似如下形式（效果不变），以避免在中间箭头时在MAT中出现的 can't assign a numpy.ndarray to a torch.FloatTensor错误，错误原因：numpy.float64可以赋值给torch.FloatTensor，但是numpy.ndarray不行。


![alt text](image_assist/image-1.png)


---2025.3.20更新---

第一部分
增加了MAPPO_discrete的实现。  
关于env.reset(seed = args.seed)的问题，和MADDPG的情况一样。  
补充说明：  
1.MAPPO.py只支持continue环境的情况。  
2.MAPPO_discrete.py只支持离散动作空间的情况。
3.

具体参数只要将学习率1e-3 改为5e-4，max_episodes改为120000即可。(学习率改为1e-3不收敛，以及改为continuous_actions= False也不收敛,所以是只适用于连续动作空间的情况)
（详见：MAPPO_file\image_assist\MAPPO copy.py）

或者见：MAPPO_discrete.py 这里环境为env.reset() 这里学习率为1e-3

两者效果如下：（均能收敛,discrete的学习率较大所以收敛快点）
![alt text](image_assist/image-2.png)

第二部分
对于HAPPO同样进行逻辑的修改
HAPPO只实现了连续动作的实现，离散动作暂未实现，离散动作实现可参考MAPPO_discrete.py。
学习率则改为1e-4，才能收敛，5e-4不收敛（或收敛不明显）。

效果如下：HAPPO:1e-4 MAPPO_simple:5e-4
![alt text](image_assist/image-3.png)


更多算法，效果如下：  
MAT：1e-4    
![alt text](image_assist/image-4.png)