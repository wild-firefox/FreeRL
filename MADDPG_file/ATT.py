import torch
import torch.nn as nn
import torch.nn.functional as F

## 注意力机制 --Modelling the Dynamic Joint Policy of Teammates with Attention Multi-agent DDPG 2018 论文版
'''
https://github.com/maohangyu/marl_demo/blob/main/C_models.py#L179
这里的多头仅对encoder进行多头处理,关于论文代码里的critic更新,是所有loos相加之后再更新 #还用到了类似于MAAC(2020)的联合损失技术
注：输出的Q不是MADDPG中的Q = Q_i^u(s,a|a_i=u_i(o_i)) u为actor的策略
而是论文中的单个智能体的Q = Q_i^u_i|u_-i(s,a_i)  =  Σa-i∈A-i [u_-i(a_-i|s)] * Q_i^u_i(s,a_i,a_-i)  u_-i为其余智能体的actor策略
其中u_-i(a_-i|s)近似为atten中的权重
Q_i^u_i(s,a_i,a_-i)近似为atten中的值V (使用多头（多个不同动作）和隐藏向量近似)
'''
class Attention_ATT(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, hidden_dim, head_count):
        super(Attention_ATT, self).__init__()
        self.fc_encoder_input = nn.Linear(encoder_input_dim, hidden_dim)
        self.fc_encoder_heads = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(head_count)])
        self.fc_decoder_input = nn.Linear(decoder_input_dim, hidden_dim)

    def forward(self, encoder_input, decoder_input):
        ''' encoder_input 由所有智能体的状态和当前智能体动作组成，decoder_input 由其余智能体的动作组成'''
        # encoder_input shape: (batch_size, input_dim)
        encoder_h = F.relu(self.fc_encoder_input(encoder_input))
        # encoder_h shape: (batch_size, hidden_dim)

        encoder_heads = torch.stack([F.relu(head(encoder_h)) for head in self.fc_encoder_heads], dim=0)
        # encoder_heads shape: (head_count, batch_size, hidden_dim)

        # decoder_input shape: (batch_size, input_dim)
        decoder_H = F.relu(self.fc_decoder_input(decoder_input))
        # decoder_H shape: (batch_size, hidden_dim)

        ''' enocde_heads 用作键值对 decoder_H 用作查询 '''
        scores = torch.sum(torch.mul(encoder_heads, decoder_H), dim=2)
        # scores shape: (head_count, batch_size)

        attention_weights = F.softmax(scores.permute(1, 0), dim=1).unsqueeze(2)
        # attention_weights shape: (batch_size, head_count, 1)

        contextual_vector = torch.matmul(encoder_heads.permute(1, 2, 0), attention_weights).squeeze()
        # contextual_vector shape: (batch_size, hidden_dim)

        return contextual_vector

class MLPNetworkWithAttention(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim = 128 ,head_count = 8 ):
        super(MLPNetworkWithAttention, self).__init__()
        #self.args = args # 3为智能体个数 12为状态维度 1为动作维度 
        self.fc_obs = nn.Linear(12, hidden_dim) 
        self.fc_action = nn.Linear(1, hidden_dim)
        self.attention_modules = Attention_ATT(hidden_dim * (3 + 1), hidden_dim * (3 - 1),hidden_dim, head_count) 
        self.fc_qvalue = nn.Linear(hidden_dim, out_dim) 

    def forward(self, x, agent_id, agents):
        agent_id_list = list(agents.keys())
        agent_id_index = agent_id_list.index(agent_id) #获取agent_id在agents中的索引 按照顺序排
        agent_n = len(agent_id_list) #智能体数量3 #12为state_dim #3*12=36
        
        out_obs_list = [F.relu(self.fc_obs(x[:,:12])) , F.relu(self.fc_obs(x[:,12:24])) , F.relu(self.fc_obs(x[:,24:36]))]               
        # out_obs_list shape: [(batch_size, hidden_dim), ...] #即 batch_size * hidden_dim * agent_count

        out_action_list = [F.relu(self.fc_action(x[:,36:37])) , F.relu(self.fc_action(x[:,37:38])) , F.relu(self.fc_action(x[:,38:39]))]
        # out_action_list shape: [(batch_size, hidden_dim), ...]

        encoder_input = torch.cat(out_obs_list + [out_action_list[agent_id_index]], dim=1)
        # encoder_input shape: (batch_size, hidden_dim * (agent_count + 1))

        decoder_input = torch.cat(out_action_list[:agent_id_index] + out_action_list[agent_id_index+1:], dim=1)
        # decoder_input shape: (batch_size, hidden_dim * (agent_count - 1))

        contextual_vector = self.attention_modules(encoder_input, decoder_input)
        # contextual_vector shape: (batch_size, hidden_dim)

        qvalue = self.fc_qvalue(contextual_vector)
        # qvalue shape: (batch_size, 1)

        return qvalue
    
## 注意力机制改 --Modelling the Dynamic Joint Policy of Teammates with Attention Multi-agent DDPG 论文 改版
class Attention_ATT_2(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, hidden_dim, head_count):
        super(Attention_ATT_2, self).__init__()
        self.fc_encoder_input = nn.Linear(encoder_input_dim, hidden_dim)
        self.fc_encoder_heads = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(head_count)]) ##
        self.fc_decoder_input = nn.Linear(decoder_input_dim, hidden_dim)

    def forward(self, encoder_input, decoder_input):
        ''' encoder_input 由所有智能体的状态和当前智能体动作组成，decoder_input 由其余智能体的动作组成'''
        encoder_h = F.relu(self.fc_encoder_input(encoder_input)) # batch_size * hidden_dim

        encoder_heads = torch.stack([F.relu(head(encoder_h)) for head in self.fc_encoder_heads], dim=0)  # head_count * batch_size * hidden_dim

        decoder_H = F.relu(self.fc_decoder_input(decoder_input)) # batch_size * hidden_dim

        ''' enocde_heads 用作键值对 decoder_H 用作查询 '''
        scores = torch.sum(torch.mul(encoder_heads, decoder_H), dim=2)  # scores shape: (head_count, batch_size) <- before sum (head_count, batch_size, hidden_dim) 
       
        attention_weights = F.softmax(scores.permute(1, 0), dim=1).unsqueeze(2) # batch_size x head_count x 1

        contextual_vector = torch.matmul(encoder_heads.permute(1, 2, 0), attention_weights).squeeze() # batch_size x hidden_dim

        return contextual_vector
    
class MLPNetworkWithAttention_2(nn.Module):
    def __init__(self, in_dim, out_dim,dim_info,agent_id,hidden_dim = 128 ,head_count = 8 ):
        '''
        在Attention2中 hidden_dim = 128 ,head_count = 8  效果最好 在3v3的环境中
        测试 256
        '''
        super(MLPNetworkWithAttention_2, self).__init__()
        '''
        #self.args = args # 3为智能体个数 12为状态维度 1为动作维度 
        self.fc_obs = nn.Linear(12, hidden_dim) 
        self.fc_action = nn.Linear(1, hidden_dim)
        '''
        self.attention_modules = Attention_ATT_2(hidden_dim , hidden_dim ,hidden_dim, head_count) 
        self.fc_qvalue = nn.Linear(hidden_dim, out_dim) 

        self.dim_info = dim_info #dim_info = {'agent_id':[obs_dim, act_dim],}
        #所有智能体的状态和当前智能体动作 维度
        self.encoder_input_dim = sum([dim_info[agent_id_][0] for agent_id_ in dim_info.keys()]) + dim_info[agent_id][1]
        self.fc1 = torch.nn.Linear(self.encoder_input_dim, hidden_dim)
        #其余智能体的动作 维度
        self.decoder_input_dim = sum([dim_info[agent_id_][1] for agent_id_ in dim_info.keys() if agent_id_ != agent_id])
        self.fc2 = torch.nn.Linear(self.decoder_input_dim, hidden_dim)

        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)

        # 所有智能体的状态的维度
        self.all_obs_dim = sum([dim_info[agent_id_][0] for agent_id_ in dim_info.keys()])
        # 所有智能体的状态+动作维度
        self.all_obs_act_dim = in_dim

        # 动作维度的列号
        d = [dim_info[agent_id_][1] for agent_id_ in dim_info.keys()] # [1,2,1] # [1,1,1]
        c_num = 0
        self.d = [0]+[c_num := c_num + i for i in d] # [0,1,3,4] # [0,1,2,3]

        

    def forward(self, x,agent_id,agents):
        agent_id_list = list(agents.keys())
        agent_id_index = agent_id_list.index(agent_id) #获取agent_id在agents中的索引 按照顺序排
        agent_n = len(agent_id_list) #智能体数量 #12为state_dim #3*12=36

        '''改
        out_obs_list = [F.relu(self.fc_obs(x[:,:12])) , F.relu(self.fc_obs(x[:,12:24])) , F.relu(self.fc_obs(x[:,24:36]))]               
        # out_obs_list shape: [(batch_size, hidden_dim), ...] #即 batch_size * hidden_dim * agent_count
        out_action_list = [F.relu(self.fc_action(x[:,36:37])) , F.relu(self.fc_action(x[:,37:38])) , F.relu(self.fc_action(x[:,38:39]))]
        # out_action_list shape: [(batch_size, hidden_dim), ...]
        encoder_input = torch.cat(out_obs_list + [out_action_list[agent_id_index]], dim=1)
        # encoder_input shape: (batch_size, hidden_dim * (agent_count + 1))
        decoder_input = torch.cat(out_action_list[:agent_id_index] + out_action_list[agent_id_index+1:], dim=1)
        # decoder_input shape: (batch_size, hidden_dim * (agent_count - 1))
        '''

        #encoder_input = self.fc1(x[:,:37]) 
        #decoder_input = self.fc2(x[:,37:39]) ##??搞错了 效果还挺好？不如下面好
        
        #action_list = [x[:,36:37],x[:,37:38],x[:,38:39]]
        # 所有智能体的动作对应列
        action_list = x[:,self.all_obs_dim:self.all_obs_act_dim]
        action_list = [action_list[:,self.d[i]:self.d[i+1]] for i in range(len(self.d)-1)]
        encoder_input = self.fc1(torch.cat((x[:,:self.all_obs_dim],action_list[agent_id_index]),1)) #batch_size * 37 -> batch_size * hidden_dim
        decoder_input = self.fc2(torch.cat((action_list[:agent_id_index]+action_list[agent_id_index+1:]),1)) #batch_size * 2 -> batch_size * hidden_dim

        # 要满足 encoder_input shape: (batch_size, hidden_dim) decoder_input shape: (batch_size, hidden_dim) 
        contextual_vector = self.attention_modules(encoder_input, decoder_input)
        # contextual_vector shape: (batch_size, hidden_dim)
        t1 = F.relu(self.fc3(contextual_vector))
        #t = F.relu(self.fc4(t1))

        qvalue = self.fc_qvalue(t1)
        # qvalue shape: (batch_size, 1)

        return qvalue

''' 对上述MLPNetworkWithAttention_2 简化一下代码 '''
class ATT_critic(nn.Module):
    id_num : int = 0
    def __init__(self, dim_info ,hidden_dim = 128 ,head_count = 4 ):
        '''
        hidden_dim = 128 ,head_count = 4  效果最好 
        '''
        super(ATT_critic, self).__init__()
        # 智能体编号
        self.id = ATT_critic.id_num
        ATT_critic.id_num += 1
        
        # 动作维度序号 # [1,2,1] -> [0,1,3,4] # 海象运算符:= 既计算某个值，也将其赋给一个变量 
        a_d = [dim_info[id][1] for id in dim_info.keys()]
        sum_ = 0
        self.a_d = [0] + [sum_:= sum_ + i for i in a_d]    #self.a_d 和 self.s_d 用于forward中切片
        # 智能体个数
        self.agent_n = len(dim_info)
        agent_id = list(dim_info.keys())[self.id]

        # 注意力相关 
        ''' 这里是每个智能体给一个attention模块'''
        self.attention_modules = Attention_ATT_2(hidden_dim , hidden_dim ,hidden_dim, head_count)  
        # 输入准备
        ## 所有智能体的状态和当前智能体动作 维度
        self.encoder_input_dim = sum([dim_info[agent_id_][0] for agent_id_ in dim_info.keys()]) + dim_info[agent_id][1]
        self.encoder_input_fc = torch.nn.Linear(self.encoder_input_dim, hidden_dim)
        ## 其余智能体的动作 维度
        self.decoder_input_dim = sum([dim_info[agent_id_][1] for agent_id_ in dim_info.keys() if agent_id_ != agent_id])
        self.decoder_input_fc = torch.nn.Linear(self.decoder_input_dim, hidden_dim)
        
        # q_value 输出 
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)   

    def forward(self, s,a):
        s = torch.cat(list(s), dim=1) # batch_size * state_dim(state 表示全局状态)
        a = torch.cat(list(a), dim=1)  

        # 准备工作
        a_i = a[:,self.a_d[self.id]:self.a_d[self.id+1]] #当前智能体动作 batch_size * action_dim
        encoder_input = self.encoder_input_fc(torch.cat((s,a_i), dim=1)) # batch_size * (state_dim + action_dim)
        decoder_input = self.decoder_input_fc(torch.cat([a[:,self.a_d[i]:self.a_d[i+1]] for i in range(self.agent_n) if i != self.id], dim=1)) # 其余智能体动作 # 如action_dim 一致 则batch_size * (action_dim * (agent_count - 1))
        
        contextual_vector = self.attention_modules(encoder_input, decoder_input) # batch_size * hidden_dim

        x = F.relu(self.fc1(contextual_vector))
        q = self.fc2(x)

        return q #(batch_size, 1)

'''对上述MLPNetworkWithAttention简化一下代码'''
class ATT_critic_raw(nn.Module):
    id_num : int = 0
    def __init__(self, dim_info ,hidden_dim = 128 ,head_count = 4 ):
        '''
        hidden_dim = 128 ,head_count = 8  效果最好 4 好
        '''
        super(ATT_critic_raw, self).__init__()
        # 智能体编号
        self.id = ATT_critic_raw.id_num
        ATT_critic_raw.id_num += 1
        
        # 状态维度序号 # [8,10,10] ->[0,8,18,28]  # 海象运算符:= 既计算某个值，也将其赋给一个变量
        s_d = [dim_info[id][0] for id in dim_info.keys()]
        sum_ = 0
        self.s_d = [0] + [sum_:= sum_ + i for i in s_d]   
        
        # 动作维度序号 # [1,2,1] -> [0,1,3,4] # 海象运算符:= 既计算某个值，也将其赋给一个变量 
        a_d = [dim_info[id][1] for id in dim_info.keys()]
        sum_ = 0
        self.a_d = [0] + [sum_:= sum_ + i for i in a_d]    #self.a_d 和 self.s_d 用于forward中切片
        # 智能体个数
        self.agent_n = len(dim_info)
        agent_id = list(dim_info.keys())[self.id]

        # 注意力相关 
        ''' 这里是每个智能体给一个attention模块'''
        self.attention_modules = Attention_ATT(hidden_dim * (self.agent_n+1), hidden_dim *(self.agent_n-1) ,hidden_dim, head_count)  
        self.fc_obs = {i : nn.Linear(obs_act[0], hidden_dim) for i ,obs_act in zip(range(self.agent_n),dim_info.values())}
        self.fc_action = {i : nn.Linear(obs_act[1], hidden_dim) for i ,obs_act in zip(range(self.agent_n),dim_info.values())}
        
        # q_value 输出 
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)   

    def forward(self, s,a):
        s = torch.cat(list(s), dim=1) # batch_size * state_dim(state 表示全局状态)
        a = torch.cat(list(a), dim=1)  

        # 准备工作

        out_obs_list = [F.relu(self.fc_obs[i](s[:,self.s_d[i]:self.s_d[i+1]])) for i in range(self.agent_n)]           
        # out_obs_list shape: [(batch_size, hidden_dim), ...] #即 batch_size * hidden_dim * agent_count

        out_action_list = [F.relu(self.fc_action[i](a[:,self.a_d[i]:self.a_d[i+1]])) for i in range(self.agent_n)]
        # out_action_list shape: [(batch_size, hidden_dim), ...]

        encoder_input = torch.cat(out_obs_list + [out_action_list[self.id]], dim=1)
        # encoder_input shape: (batch_size, hidden_dim * (agent_count + 1))

        decoder_input = torch.cat(out_action_list[:self.id] + out_action_list[self.id+1:], dim=1)
        
        contextual_vector = self.attention_modules(encoder_input, decoder_input) # batch_size * hidden_dim

        x = F.relu(self.fc1(contextual_vector))

        q = self.fc2(x)

        return q #(batch_size, 1)