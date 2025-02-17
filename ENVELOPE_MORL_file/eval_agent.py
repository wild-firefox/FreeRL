### 得到当前agent的帕累托前沿和目标帕累托前沿
from ENVELOPE_DQN import ENVELOPE
from ENVELOPE_DQN import get_env
import numpy as np
import os

import matplotlib.pyplot as plt
## 中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

time = [-1, -3, -5, -7, -8, -9, -13, -14, -17, -19]

treasure = [0.7, 8.2, 11.5, 14., 15.1, 16.1, 19.6, 20.3, 22.4, 23.7]

np.random.seed(0)
env,dim_info, max_action, is_continue = get_env('deep-sea-treasure-v0')

num = 1 # 2展示训练不好的情况 2: max_epiosed:2000 1: max_epiosed:5000 
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'./results')
model_dir = f'{results_dir}\deep-sea-treasure-v0\ENVELOPE_DQN_{num}'

policy = ENVELOPE.load(dim_info= dim_info,is_continue=is_continue , model_dir = model_dir)

realc_x = []
realc_y = []
max_re = []
##
time_reward = []
treasure_reward = []

# 目标奖励
realc_x_dst = []
realc_y_dst = []
max_re_dst = []
## 
time_reward_dst = []
treasure_reward_dst = []


w1 = np.arange(0, 1, 0.01) # 测试共100个权重
w2 = 1 - w1
loss = 0
for i in range(len(w1)):
    # w = np.random.randn(2)
    # w = np.abs(w) / np.linalg.norm(w, ord=1)
    w = [w1[i], w2[i]]
    w = np.round(w,2)
    w_e = w / np.linalg.norm(w, ord=2) #单位向量

    id = np.argmax(w[0]*np.array(treasure) + w[1]*np.array(time))
    #print([time[id],treasure[id]])
    for _ in range(1):
        obs,info = env.reset(seed=0)
        episode_reward = 0
        done = False
        reward_vec = np.array([0,0])
        while not done:
            action = policy.evaluate_action(obs,w)
            action_ = action

            next_obs, reward,terminated, truncated, infos = env.step(action_) 
            done = terminated or truncated
            reward_vec = np.add(reward_vec,reward)
            episode_reward += np.dot(reward, w)
            obs = next_obs

        #print(reward_vec,[treasure[id],time[id]])

    # if w[0] >= 0.8:
    #     print(w[0] , episode_reward)
    realc = np.dot(w, reward_vec) * w_e
    #print(episode_reward,np.dot(w, reward_vec)) 这两一样
    max_re.append(episode_reward)
    realc_x.append(realc[0])
    realc_y.append(realc[1])

    treasure_reward.append(reward_vec[0])
    time_reward.append(reward_vec[1])

    ## 目标奖励
    realc_dst = np.dot(w,[treasure[id],time[id]]) * w_e
    realc_x_dst.append(realc_dst[0])
    realc_y_dst.append(realc_dst[1])
    max_re_dst.append(np.dot(w,[treasure[id],time[id]]))

    treasure_reward_dst.append(treasure[id])
    time_reward_dst.append(time[id])


    ## AE: 适应性误差 # 多乘w 和w_e       
    base = np.linalg.norm(realc_dst, ord=2)
    loss += np.linalg.norm( realc_dst - realc , ord=2)/base


##将每个w 对应的w_e(单位向量) 与real_sol相乘，得到x,y坐标，并标记相应值，画出帕累托前沿

'''
论文中的定义 CR
在这里 
模型的解（两个目标）为 reward_vec
目标的解 time = [-1, -3, -5, -7, -8, -9, -13, -14, -17, -19]
        treasure = [0.7, 8.2, 11.5, 14., 15.1, 16.1, 19.6, 20.3, 22.4, 23.7]
        这里使用向量化的方式realc_x_dst, realc_x替代
precision = 模型（在帕累托上)预测对的解/模型的解
recall = 模型（在帕累托上)预测对的解/目标的解
F1 = 2 * precision * recall / (precision + recall)
'''

def find_in(A, B, base=2):
    # 此函数copy自原代码
    # base = 0: tolerance w.r.t. A
    # base = 1: tolerance w.r.t. B
    # base = 2: no tolerance
    cnt = 0.0
    for a in A:
        for b in B:
            if base == 0:
              if np.linalg.norm(a - b, ord=1) < 0.20*np.linalg.norm(a):
                  cnt += 1.0
                  break
            elif base == 1:
              if np.linalg.norm(a - b, ord=1) < 0.20*np.linalg.norm(b):
                  cnt += 1.0
                  break
            elif base == 2:
              if np.linalg.norm(a - b, ord=1) < 0.3:
                  cnt += 1.0
                  break
    return cnt / len(A)

act = np.vstack((realc_x,realc_y)).transpose() # ->[[x1,y1],[x2,y2],...] # 模型的解
obj = np.vstack((realc_x_dst,realc_y_dst)).transpose() # ->[[x1,y1],[x2,y2],...] # 目标的解
precision = find_in(act, obj, base=2)
recall = find_in(act, obj, base=2)
print("precision: ", precision, "recall: ", recall)
CR = 2 * precision * recall / (precision + recall)
print("CR: ", CR)
print("AE_loss: ", loss/len(w1))


# 保存reward
np.save(f"{model_dir}/reward.npy",max_re)

# 指定图的大小 画出论文里定义的帕累托前沿
plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.title('帕累托前沿')
plt.plot(realc_x_dst, realc_y_dst, 'bo', markersize=2, label='目标帕累托前沿')
plt.plot(realc_x, realc_y, 'ro', markersize=2, label='模型帕累托前沿')
plt.legend(loc='upper left')
plt.xlabel('w_treasure_vector')
plt.ylabel('w_time_vector')
plt.grid()

# ---------------------
plt.subplot(212)
plt.title('不同权重下的奖励')
plt.plot(w1, max_re_dst, 'bo', markersize=2, label='目标奖励')
plt.plot(w1, max_re, 'ro', markersize=2, label='模型奖励')
plt.legend(loc='upper left')
plt.xlabel('w_treasure')
plt.ylabel('max_reward (ex gamma)')

## 栅格化
plt.grid()

plt.savefig(f"{model_dir}/pareto_reward.png")
plt.show()

# ---------------------
## 画出大部分定义的帕累托的图

plt.plot(time_reward, treasure_reward, 'go', markersize=2, label='模型实际帕累托')
plt.plot(time_reward_dst, treasure_reward_dst, 'bo', markersize=2, label='目标帕累托')

# plt.xlim(np.min(x)-1e-4, np.max(x)+1e-4)
# plt.ylim(np.min(y)-1e-8, np.max(y)+1e-8)

plt.legend(loc='upper left')
plt.xlabel('w_1_value')
plt.ylabel('w_2_value')
plt.grid()
plt.savefig(f"{model_dir}/pareto_reward_common.png")

plt.show()
