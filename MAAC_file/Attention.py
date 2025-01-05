import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
关于随机种子问题：在MAPPO_file/attention.py中已经解释清楚了，这里
解1（使用了解决方法1）：这里的Attention_Critic 对应MAAC_discrete.py中的attention_critic
解2（使用了解决方法2）：这里的Attention_Critic_ 对应MAAC_discrete中的attention_critic 

解2 中删去 Attention中的 torch.manual_seed(0) 也可以使得两次运行一致,解1 则不行

但是两者结果不一致的原因：
解1的seed初始化两次 

例子：下述的结果为：
Weight1: -0.007486820220947266
Weight2: -0.8230451345443726

## 1
import torch
import torch.nn as nn

class A:
    def __init__(self):
        torch.manual_seed(0) # 增加此行 解1
        self.x = torch.nn.Linear(1, 1)
        #print(f'Weight: {self.x.weight[0][0]}') # 一样

class B:
    def __init__(self, a=A()):
        self.a = a
        self.b = torch.nn.Linear(1, 1)
        print(f'Weight1: {self.b.weight[0][0]}') #不一样

# 设置随机种子
torch.manual_seed(0)
b = B()

## 2
class A:
    def __init__(self):
        #torch.manual_seed(0) # 增加此行 解1
        self.x = torch.nn.Linear(1, 1)
        #print(f'Weight: {self.x.weight[0][0]}') # 一样

class B:
    def __init__(self, a=None): # 解2 
        self.a = a
        self.b = torch.nn.Linear(1, 1)
        print(f'Weight2: {self.b.weight[0][0]}') #不一样

# 设置随机种子
torch.manual_seed(0)
a_ = A() # 改
b = B(a=a_) # 改
'''

'''
基于MAAC代码修改 改成连续域的版本：
https://github.com/shariqiqbal2810/MAAC/blob/master/utils/critics.py#L8
'''
class Attention(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=1):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim should be divisible by num_heads"
        torch.manual_seed(0) ## 解决随机初始化问题
        self.query = nn.Linear(in_dim, hidden_dim,bias=False) 
        print(self.query.weight[0][0])
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
        #print(Q[0][0][0] )
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
        
        #print(Q[0][0][0][0] )
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
    def __init__(self, dim_info:dict ,hidden_1=128, norm_in=False, is_continue = False,attention_dim=128, attention_block=Attention(in_dim=128, hidden_dim=128,num_heads=4)):
        '''
        dim_info: dict(id:[s,a],id:[s,a],...)  #id为智能体编号
        注意力机制作用：改善了MADDPG中critic输入随智能体数目增大而指数增加的扩展性问题
        '''
        super(Attention_Critic, self).__init__()
        
        self.is_continue = is_continue
        # 智能体编号
        self.id = Attention_Critic.id_num
        Attention_Critic.id_num += 1

        # 状态维度序号 [8,10,10] ->[0,8,18,28]  # 海象运算符:= 既计算某个值，也将其赋给一个变量 
        s_d = [dim_info[id][0] for id in dim_info.keys()]
        sum_ = 0  
        self.s_d = [0] + [sum_:= sum_ + i for i in s_d] # 意思等同 s_d = [0] + [sum(s_d[:i+1]) for i in range(len(s_d))] 
        # 动作维度序号
        a_d = [dim_info[id][1] for id in dim_info.keys()]
        sum_ = 0
        self.a_d = [0] + [sum_:= sum_ + i for i in a_d]  #self.a_d 和 self.s_d 用于forward中切片
        # 状态+动作维度
        sa_d =  [i+j for i,j in zip(s_d,a_d)] 
        # 智能体个数
        self.agent_n = len(dim_info)

        # 注意力相关
        self.attention = attention_block
        ## 输入准备 encoder_fc -> leaky_relu 
        if norm_in: # affine=False 不使用可学习参数 只做归一化
            self.encoder_fc_s = nn.Sequential(nn.BatchNorm1d(s_d[self.id],affine=False),nn.Linear(s_d[self.id], attention_dim),)
            self.encoder_fc_sa = nn.Sequential(nn.BatchNorm1d(sa_d[self.id],affine=False),nn.Linear(sa_d[self.id], attention_dim),)
            self.encoder_fc_a = nn.Sequential(nn.BatchNorm1d(a_d[self.id],affine=False),nn.Linear(a_d[self.id], attention_dim),)
        else:
            self.encoder_fc_s = nn.Linear(s_d[self.id], attention_dim)
            self.encoder_fc_sa = nn.Linear(sa_d[self.id], attention_dim)
            self.encoder_fc_a = nn.Linear(a_d[self.id], attention_dim)
        #print(self.encoder_fc_s.weight[0])
        # q_value 输出 
        self.fc1 = torch.nn.Linear(2*attention_dim, hidden_1)
        self.fc2 = torch.nn.Linear(hidden_1, a_d[self.id])  # 1 为action_dim

        #### bias 连续域用 作为baseline
        self.bias_fc1 = nn.Linear(attention_dim, attention_dim)
        self.bias_fc2 = nn.Linear(attention_dim, 1) 
        
    def forward(self, s,a, agents,require_v  = False): #agents[agent_id]=Agent() 
        s = torch.cat(list(s), dim=1) # batch_size * state_dim(state 表示全局状态)
        a = torch.cat(list(a), dim=1)  
        agent_id_list = list(agents.keys())
        # 准备工作
        s_i_c = s[:,self.s_d[self.id]:self.s_d[self.id+1]] #当前智能体状态 batch_size * obs_dim
        s_i = self.encoder_fc_s(s_i_c) #当前智能体的embedding batch_size * attention_dim
        s_i = F.leaky_relu(s_i)     # 加一层激活
        e_q = s_i.unsqueeze(dim = 1) #当前智能体的查询 batch_size * 1 * attention_dim

        ## 补充 用于continue 
        if self.is_continue:
            '''参考https://github.com/Future-Power-Networks/MAPDN/blob/main/critics/maac_critic.py#L141'''
            a_i = a[:,self.a_d[self.id]:self.a_d[self.id+1]] #当前智能体动作 batch_size * action_dim
            sa_i = torch.cat((s_i_c,a_i),dim = 1)
            sa_i = self.encoder_fc_sa(sa_i)
            sa_i = F.leaky_relu(sa_i)
            # '''自创'''
            # # a_i = self.encoder_fc_a(a_i)
            # # a_i = F.leaky_relu(a_i)

        e_k = []
        for i in range(self.agent_n):
            if i!=self.id:
                s_other_i = s[:,self.s_d[i]:self.s_d[i+1]]
                a_other_i = a[:,self.a_d[i]:self.a_d[i+1]] # batch_size * action_dim
                sa_other_i = torch.cat((s_other_i,a_other_i),1)
                sa_other_i = agents[agent_id_list[i]].critic.encoder_fc_sa(sa_other_i)  ### 本页测试时，去掉.critic
                sa_other_i = F.leaky_relu(sa_other_i)
                e_k.append(sa_other_i.unsqueeze(dim = 1)) # batch_size * 1 * attention_dim
        e_k = torch.cat(e_k,dim=1) # batch_size * (agent_n-1) * attention_dim
        # print(e_q[0][0][0])
        # print(e_k[0][0][0])
        # 注意力机制
        '''当前智能体的注意力值 即其他智能体对当前智能体的贡献值'''
        other_values = self.attention(e_q, e_k) # batch_size * attention_dim
        #print(other_values[0][0])
        # q_value 输出 
        if self.is_continue :
            ''' 参考'''
            X_in=torch.cat([sa_i, other_values], dim=1)
            X_in = F.leaky_relu(self.fc1(X_in))
            q = self.fc2(X_in) # batch_size * 1  #当前动作的q值
            
            if require_v:
                bias = self.bias_fc1(s_i)
                bias = F.leaky_relu(bias)
                b = self.bias_fc2(bias)
                adv = q - b 
            # ''' 自创'''
            # X_in_ = torch.cat([s_i, other_values], dim=1)
            # X_in_ = F.leaky_relu(self.fc1(X_in_))
            # all_q = self.fc2(X_in_) # batch_size * action_dim 所有动作的q值
            # v = q - all_q

        else:
            X_in=torch.cat([s_i, other_values], dim=1)
            X_in = F.leaky_relu(self.fc1(X_in))
            all_q = self.fc2(X_in) # batch_size * action_dim 所有动作的q值
            #print(all_q)
            int_acs = torch.argmax(a[:,self.a_d[self.id]:self.a_d[self.id+1]], dim=1, keepdim=True)
            q = all_q.gather(dim = 1, index =int_acs) # batch_size * 1 当前动作的q值

        if require_v and self.is_continue:
            return q,adv
        elif self.is_continue:
            return q
        else:
            return q ,all_q #输出 batch_size * 1


class Attention_Critic_(nn.Module):
    id_num : int = 0
    def __init__(self, dim_info:dict ,hidden_1=128, norm_in=False, is_continue = False,attention_dim=128, attention_block=None):
        '''
        dim_info: dict(id:[s,a],id:[s,a],...)  #id为智能体编号
        注意力机制作用：改善了MADDPG中critic输入随智能体数目增大而指数增加的扩展性问题
        '''
        super(Attention_Critic_, self).__init__()
        
        self.is_continue = is_continue
        # 智能体编号
        self.id = Attention_Critic.id_num
        Attention_Critic.id_num += 1

        if attention_block is None:
            attention_block = Attention(in_dim=attention_dim, hidden_dim=hidden_1, num_heads=4)
        #self.attention = attention_block

        # 状态维度序号 [8,10,10] ->[0,8,18,28]  # 海象运算符:= 既计算某个值，也将其赋给一个变量 
        s_d = [dim_info[id][0] for id in dim_info.keys()]
        sum_ = 0  
        self.s_d = [0] + [sum_:= sum_ + i for i in s_d] # 意思等同 s_d = [0] + [sum(s_d[:i+1]) for i in range(len(s_d))] 
        # 动作维度序号
        a_d = [dim_info[id][1] for id in dim_info.keys()]
        sum_ = 0
        self.a_d = [0] + [sum_:= sum_ + i for i in a_d]  #self.a_d 和 self.s_d 用于forward中切片
        # 状态+动作维度
        sa_d =  [i+j for i,j in zip(s_d,a_d)] 
        # 智能体个数
        self.agent_n = len(dim_info)

        # 注意力相关
        self.attention = attention_block
        ## 输入准备 encoder_fc -> leaky_relu 
        if norm_in: # affine=False 不使用可学习参数 只做归一化
            self.encoder_fc_s = nn.Sequential(nn.BatchNorm1d(s_d[self.id],affine=False),nn.Linear(s_d[self.id], attention_dim),)
            self.encoder_fc_sa = nn.Sequential(nn.BatchNorm1d(sa_d[self.id],affine=False),nn.Linear(sa_d[self.id], attention_dim),)
            self.encoder_fc_a = nn.Sequential(nn.BatchNorm1d(a_d[self.id],affine=False),nn.Linear(a_d[self.id], attention_dim),)
        else:
            self.encoder_fc_s = nn.Linear(s_d[self.id], attention_dim)
            self.encoder_fc_sa = nn.Linear(sa_d[self.id], attention_dim)
            self.encoder_fc_a = nn.Linear(a_d[self.id], attention_dim)
        #print(self.encoder_fc_s.weight[0])
        # q_value 输出 
        self.fc1 = torch.nn.Linear(2*attention_dim, hidden_1)
        self.fc2 = torch.nn.Linear(hidden_1, a_d[self.id])  # 1 为action_dim

        #### bias 连续域用 作为baseline
        self.bias_fc1 = nn.Linear(attention_dim, attention_dim)
        self.bias_fc2 = nn.Linear(attention_dim, 1) 
        
    def forward(self, s,a, agents,require_v  = False): #agents[agent_id]=Agent() 
        s = torch.cat(list(s), dim=1) # batch_size * state_dim(state 表示全局状态)
        a = torch.cat(list(a), dim=1)  
        agent_id_list = list(agents.keys())
        # 准备工作
        s_i_c = s[:,self.s_d[self.id]:self.s_d[self.id+1]] #当前智能体状态 batch_size * obs_dim
        s_i = self.encoder_fc_s(s_i_c) #当前智能体的embedding batch_size * attention_dim
        s_i = F.leaky_relu(s_i)     # 加一层激活
        e_q = s_i.unsqueeze(dim = 1) #当前智能体的查询 batch_size * 1 * attention_dim

        ## 补充 用于continue 
        if self.is_continue:
            '''参考https://github.com/Future-Power-Networks/MAPDN/blob/main/critics/maac_critic.py#L141'''
            a_i = a[:,self.a_d[self.id]:self.a_d[self.id+1]] #当前智能体动作 batch_size * action_dim
            sa_i = torch.cat((s_i_c,a_i),dim = 1)
            sa_i = self.encoder_fc_sa(sa_i)
            sa_i = F.leaky_relu(sa_i)
            # '''自创'''
            # # a_i = self.encoder_fc_a(a_i)
            # # a_i = F.leaky_relu(a_i)

        e_k = []
        for i in range(self.agent_n):
            if i!=self.id:
                s_other_i = s[:,self.s_d[i]:self.s_d[i+1]]
                a_other_i = a[:,self.a_d[i]:self.a_d[i+1]] # batch_size * action_dim
                sa_other_i = torch.cat((s_other_i,a_other_i),1)
                sa_other_i = agents[agent_id_list[i]].critic.encoder_fc_sa(sa_other_i)  ### 本页测试时，去掉.critic
                sa_other_i = F.leaky_relu(sa_other_i)
                e_k.append(sa_other_i.unsqueeze(dim = 1)) # batch_size * 1 * attention_dim
        e_k = torch.cat(e_k,dim=1) # batch_size * (agent_n-1) * attention_dim

        # 注意力机制
        '''当前智能体的注意力值 即其他智能体对当前智能体的贡献值'''
        other_values = self.attention(e_q, e_k) # batch_size * attention_dim
        # q_value 输出 
        if self.is_continue :
            ''' 参考'''
            X_in=torch.cat([sa_i, other_values], dim=1)
            X_in = F.leaky_relu(self.fc1(X_in))
            q = self.fc2(X_in) # batch_size * 1  #当前动作的q值
            
            if require_v:
                bias = self.bias_fc1(s_i)
                bias = F.leaky_relu(bias)
                b = self.bias_fc2(bias)
                adv = q - b 
            # ''' 自创'''
            # X_in_ = torch.cat([s_i, other_values], dim=1)
            # X_in_ = F.leaky_relu(self.fc1(X_in_))
            # all_q = self.fc2(X_in_) # batch_size * action_dim 所有动作的q值
            # v = q - all_q

        else:
            X_in=torch.cat([s_i, other_values], dim=1)
            X_in = F.leaky_relu(self.fc1(X_in))
            all_q = self.fc2(X_in) # batch_size * action_dim 所有动作的q值
            #print(all_q)
            int_acs = torch.argmax(a[:,self.a_d[self.id]:self.a_d[self.id+1]], dim=1, keepdim=True)
            q = all_q.gather(dim = 1, index =int_acs) # batch_size * 1 当前动作的q值

        if require_v and self.is_continue:
            return q,adv
        elif self.is_continue:
            return q
        else:
            return q ,all_q #输出 batch_size * 1

''' 
# 测试代码
if __name__ == '__main__':
    import random
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(42)  # 设置一个固定的种子

    # 示例：使用 Attention_Critic
    dim_info = {0: [8, 10], 1: [8, 10]}  # 假设两个智能体
    attention_critic_1 = Attention_Critic(dim_info)
    attention_critic_2 = Attention_Critic(dim_info)
    

    # 准备输入数据
    s = [torch.rand(32, 8), torch.rand(32, 8)]  # 32 是 batch size
    a = [torch.rand(32, 10), torch.rand(32, 10)]
    
    print('s',s[0][0])
    print('a',a[0][0])
    # 执行前向传播
    q1, all_q1 = attention_critic_1(s, a, agents={0: attention_critic_1, 1: attention_critic_2})
    q2, all_q2 = attention_critic_2(s, a, agents={0: attention_critic_1, 1: attention_critic_2})

    # 打印结果
    # print("Result 1:", q1[0])
    # print("Result 2:", q2[0])

'''

        
        
        


