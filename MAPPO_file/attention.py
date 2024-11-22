import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## trick
from MAPPO import net_init

'''
!!!此attention机制原理是为师兄所写,使用此代码请引用论文：!!!
基于深度强化学习的有源配电网协同调压控制方法研究_毕刚

这里原理和师兄一致，代码稍微有些不同，增加了多头注意力机制
'''



'''随机种子问题
当你在类的默认参数中传递一个对象时，该对象是在类定义时创建的，
而不是在实例化类时创建的。因此，如果在类外部设置随机种子，它不会影响在类定义时已经创建的对象。

具体来说：使用如下 会造成随机初始化问题
import torch
import torch.nn as nn

class A:
    def __init__(self):
        self.x = torch.nn.Linear(1, 1)
        print(f'Weight: {self.x.weight[0][0]}')

class B:
    def __init__(self, a=A()):
        self.a = a

# 设置随机种子
torch.manual_seed(0)

b = B()

## 或者 or

import random

class A:
    def __init__(self):
        self.x = random.random()
        print(f'x: {self.x}')

class B:
    def __init__(self, a=A()):
        self.a = a

# 设置随机种子
random.seed(0)

b = B()

解决方法：
1.在类A中多增加一个随机种子的设置
2.在主代码中，先实例化类A，然后将实例作为参数传递给类B。

解1：以上述第一个问题为例子
将classA修改如下
class A:
    def __init__(self):
        torch.manual_seed(0) # 增加此行
        self.x = torch.nn.Linear(1, 1)
        print(f'Weight: {self.x.weight[0][0]}')

解2：以上述第一个问题为例子
修改
class B:
    def __init__(self, a=None): # 改
        self.a = a

# 设置随机种子
torch.manual_seed(0)
a_ = A() # 改
b = B(a=a_) # 改

总结：之前使用1来解决，但是使用2更方便理解
这里使用2 
'''

class Attention(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=128, num_heads=4):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim should be divisible by num_heads"
        #torch.manual_seed(0) ## 解决随机初始化问题
        self.query = nn.Linear(in_dim, hidden_dim,bias=False) 
        #print(self.query.weight[0][0])
        self.key = nn.Linear(in_dim, hidden_dim,bias=False) 
        self.value = nn.Sequential(nn.Linear(in_dim,hidden_dim),nn.LeakyReLU()) # 输出经过激活函数处理

        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, e_q, e_k):  
        '''
        公式为：Attention(Q, K, V) = softmax(Q*K^T/sqrt(d_k)) * V 输出为当前智能体的注意力值
        Q: 查询         K: 键        V: 值        d_k: 键的维度   【注】n为其余智能体数目
        e_q: 为batch_size * 1 * in_dim  #e_q 为当前状态编码 
        e_k: 为batch_size * n * in_dim  #e_k 为其余智能体状态动作编码
        本质：在其余智能体中找到与当前智能体最相关的智能体
        '''
        Q = self.query(e_q)  #查询当前智能体价值 Q: batch_size * 1 * hidden_dim
        K = self.key(e_k)    #其余智能体的键 K: batch_size * n * hidden_dim
        V = self.value(e_k)  #其余智能体的值 V: batch_size * n * hidden_dim
        #d_k = K[0].shape[1] #键的维度 也就是hidden_dim

        # Split the keys, queries and values in num_heads
        Q = Q.reshape(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim) #Q: batch_size * 1 * num_heads * head_dim
        K = K.reshape(K.shape[0], K.shape[1], self.num_heads, self.head_dim) #K: batch_size * n * num_heads * head_dim
        V = V.reshape(V.shape[0], V.shape[1], self.num_heads, self.head_dim) #V: batch_size * n * num_heads * head_dim

        Q = Q.permute(2, 0, 1, 3)  #Q: num_heads * batch_size * 1 * head_dim
        K = K.permute(2, 0, 1, 3)  #K: num_heads * batch_size * n * head_dim
        V = V.permute(2, 0, 1, 3)  #V: num_heads * batch_size * n * head_dim
        d_k = self.head_dim
        
        # 主要公式
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(d_k)                     #Q*K^T/sqrt(d_k)           #scores:       num_heads * batch_size * 1 * n
        attn_weights = torch.softmax(scores, dim=-1)                                       #softmax(Q*K^T/sqrt(d_k))  #attn_weights: num_heads * batch_size * 1 * n
        attn_values = torch.matmul(attn_weights, V)                                        #attn_weights * V          #attn_values:  num_heads * batch_size * 1 * head_dim
        
        attn_values = attn_values.permute(1, 2, 0, 3)                                      #batch_size * 1 * num_heads * head_dim
        attn_values = attn_values.reshape(attn_values.shape[0],  attn_values.shape[1], -1) #batch_size * 1 * hidden_dim

        out = self.fc_out(attn_values.squeeze(1))  # batch_size * hidden_dim

        return out #当前智能体的注意力值
    
class Attention_Critic(nn.Module):
    id_num : int = 0
    def __init__(self, dim_info:dict[str,list] ,hidden_1=128, hidden_2 = 128 ,is_continue = False,attention_dim=128, attention_block=None,trick = None):
        '''
        dim_info: dict(id:[s,a],id:[s,a],...)  #id为智能体编号
        注意力机制作用：改善了MADDPG中critic输入随智能体数目增大而指数增加的扩展性问题
        测试发现：(测试位置：代码打上##的地方)
        1.encoder_fc 后不加leaky_relu激活层更好
        2.attention机制输出后，只使用两层全连接层效果更好
        '''
        super(Attention_Critic, self).__init__()
        self.is_continue = is_continue
        # 智能体编号
        self.id = Attention_Critic.id_num
        Attention_Critic.id_num += 1

        if attention_block is None:
            attention_block = Attention(in_dim=attention_dim, hidden_dim=hidden_1, num_heads=4)
        self.attention = attention_block

        # 状态维度序号 [8,10,10] ->[0,8,18,28]  # 海象运算符:= 既计算某个值，也将其赋给一个变量 
        s_d = [dim_info[id][0] for id in dim_info.keys()]
        sum_ = 0  
        self.s_d = [0] + [sum_:= sum_ + i for i in s_d] # 意思等同 s_d = [0] + [sum(s_d[:i+1]) for i in range(len(s_d))] 

        # 智能体个数
        self.agent_n = len(dim_info)
        self.agent_id = list(dim_info.keys())[self.id]

        ## 输入准备
        self.encoder_fc_s = nn.Linear(s_d[self.id], attention_dim)

        # q_value 输出 
        self.l1 = torch.nn.Linear(2*attention_dim, hidden_1)
        #self.l2 = torch.nn.Linear(hidden_1,hidden_2) 
        self.l3 = torch.nn.Linear(hidden_1,1)  

        self.trick = trick
        # 使用 orthogonal_init
        if trick['orthogonal_init']:
            net_init(self.l1)
            ##net_init(self.l2)
            net_init(self.l3)  
        

    def forward(self, s, agents):
        s = torch.cat(list(s), dim=1) # batch_size * state_dim(state 表示全局状态)
        if self.trick['feature_norm']:
            s = F.layer_norm(s, s.size()[1:])

        s_i_c = s[:,self.s_d[self.id]:self.s_d[self.id+1]] #当前智能体状态 batch_size * obs_dim
        s_i = self.encoder_fc_s(s_i_c) #当前智能体的embedding batch_size * attention_dim
        ##s_i = F.leaky_relu(s_i)     # 加一层激活
        e_q = s_i.unsqueeze(dim = 1) #当前智能体的查询 batch_size * 1 * attention_dim

        e_k = []
        for i in range(self.agent_n):
            if i!=self.id:
                s_other_i = s[:,self.s_d[i]:self.s_d[i+1]]
                s_other_i = agents[self.agent_id].critic.encoder_fc_s(s_other_i)  
                ##s_other_i = F.leaky_relu(s_other_i)
                e_k.append(s_other_i.unsqueeze(dim = 1)) # batch_size * 1 * attention_dim
        e_k = torch.cat(e_k,dim=1) # batch_size * (agent_n-1) * attention_dim
        x = self.attention(e_q, e_k) # batch_size * attention_dim
        x = torch.cat([s_i,x],dim=1) # batch_size * 2*attention_dim
        q = F.relu(self.l1(x))
        if self.trick['LayerNorm']:
            q = F.layer_norm(q, q.size()[1:])
        ##q = F.relu(self.l2(q))
        q = self.l3(q)
        return q 







        
    

