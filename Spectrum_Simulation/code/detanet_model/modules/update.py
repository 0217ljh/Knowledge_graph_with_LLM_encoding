from e3nn import o3
from .acts import activations
from torch_scatter import scatter
from torch import nn
import torch

class Tensorproduct_Attention(nn.Module):
    def __init__(self,num_features,irreps_T,act,):
        super(Tensorproduct_Attention, self).__init__()
        self.feature=num_features
        self.lq=o3.Linear(irreps_T,irreps_T, internal_weights=True,shared_weights=True)
        self.lk=o3.Linear(irreps_T,irreps_T, internal_weights=True,shared_weights=True)
        self.lv=o3.Linear(irreps_T,irreps_T, internal_weights=True,shared_weights=True)
        self.ls=nn.Linear(num_features,2*num_features)
        self.lvs=nn.Linear(num_features,num_features)
        irreps_scalar = o3.Irreps([(num_features, (0, 1))])
        intp1=[]
        intp2=[]
        # 'uuu' is the feature-wise tensor product
        # That is, a tensor product is performed for each irrep tensor feature in each set of irrep tensor features.
        # For example, two 128-dimensional '1o' features are multiplied 'uuuu'
        # each of the 128 features are multiplied to produce a set of '0e+1o+2e', resulting in '128x0e+128x1o+128x2e'
        for i, (_, _) in enumerate(o3.Irreps(irreps_T)):
                intp1.append((i,i,0,'uuu',False))
                intp2.append((0,i,i,'uuu',True))
        self.tp1=o3.TensorProduct(irreps_T,irreps_T,irreps_scalar,instructions=intp1)
        self.tp2=o3.TensorProduct(irreps_scalar,irreps_T,irreps_T,instructions=intp2)
        self.softmax=activations('softmax')
        self.actlvs=activations(act, num_features=num_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ls.weight)
        self.ls.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lvs.weight)
        self.lvs.bias.data.fill_(0)

    def forward(self,T,S):
        # First, the feature-wise tensor product is performed on the query and key of the irrep tensor feature,
        # and the attention feature is generated
        s=self.tp1(self.lq(T),self.lk(T))
        # The generated attention feature is passed through a linear layer and a softmax function
        # and is divided into 2 parts

        su,sd=torch.split(self.softmax(self.ls(s)),split_size_or_sections=[self.feature,self.feature],dim=-1)

        #The value of final scalar and irrep tensor features are multiplied by attention feature to generate the result
        tu=self.tp2(sd,self.lv(T))
        return tu,su*self.actlvs(self.lvs(S))

class Update(nn.Module):
    def __init__(self,num_features,act,irreps_mout,irreps_T,dropout=0.0):
        super(Update, self).__init__()
        self.actu=activations(act,num_features=num_features)
        self.drop=nn.Dropout(dropout)
        self.outt = o3.Linear(irreps_in=irreps_mout, irreps_out=irreps_T, internal_weights=True,
                                shared_weights=True)
        self.outs=nn.Linear(num_features,num_features)

        self.uattn=Tensorproduct_Attention(num_features=num_features,irreps_T=irreps_T,act=act)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.outs.weight)
        self.outs.bias.data.fill_(0)
    def forward(self,T,S,mijt,mijs,index):
        # Update by resnet_style, adding first the results of message
        # and then the results of the tensor product attention module.
        mijt = torch.softmax(mijt, dim=0)  # 对 mijt 进行 softmax 归一化
        mijs = torch.softmax(mijs, dim=0)  # 对 mijs 进行 softmax 归一化
        mijt = torch.nan_to_num(mijt, nan=0.0)
        j=index[1]
        ut=self.outt(scatter(src=mijt,index=j,dim=0))
        us=self.actu(self.outs(scatter(src=mijs,index=j,dim=0)))
        T=T+ut
        S=S+self.drop(us)
        ut2,us2=self.uattn(T=T,S=S)
        # print("ut2:", ut2)
        # print("us2:", us2)
        T=T+ut2
        S=S+self.drop(us2)
        return T,S

# import torch
# from e3nn import o3
# from .acts import activations
# from torch_scatter import scatter
# from torch import nn
# from .LLM_Edge import LLM_Edge_Module  # 假设 LLM_Edge_Module 在这个文件里定义

# class Update(nn.Module):
#     def __init__(self, num_features, act, irreps_mout, irreps_T, dropout=0.0, llm_edge_module=None):
#         super(Update, self).__init__()
#         self.actu = activations(act, num_features=num_features)
#         self.drop = nn.Dropout(dropout)
#         self.outt = o3.Linear(irreps_in=irreps_mout, irreps_out=irreps_T, internal_weights=True, shared_weights=True)
#         self.outs = nn.Linear(num_features, num_features)

#         # 使用传入的 LLM Edge 模块
#         if llm_edge_module is not None:
#             self.llm_edge = llm_edge_module  # 使用提供的 LLM_Edge_Module 实例
#         else:
#             self.llm_edge = LLM_Edge_Module(num_features=num_features)  # 默认初始化 LLM_Edge_Module

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.outs.weight)
#         self.outs.bias.data.fill_(0)


#     def forward(self, T, S, mijt, mijs, index):

#         # print("irreps_in:", self.outt.irreps_in)
#         # print("irreps_out:", self.outt.irreps_out)
#         # print("outt weight:", self.outt.weight)
#         # print("outt bias:", self.outt.bias)

#         # 进行 softmax 归一化
#         mijt = torch.softmax(mijt, dim=0)  # 对 mijt 进行 softmax 归一化
#         mijs = torch.softmax(mijs, dim=0)  # 对 mijs 进行 softmax 归一化
#         mijt = torch.nan_to_num(mijt, nan=0.0)
#         # print(torch.isnan(mijt).any(), torch.isinf(mijt).any())
#         # print(torch.isnan(mijs).any(), torch.isinf(mijs).any())
#         # print("mijt after softmax:",mijt)
#         # print("mijs after softmax:",mijs)

#         j = index[0]
#         ut1=scatter(src=mijt, index=j, dim=0)
#         # print('outs+scatter ut:',ut1)
#         ut = self.outt(scatter(src=mijt, index=j, dim=0))
#         us = self.actu(self.outs(scatter(src=mijs, index=j, dim=0)))
        
#         # print("ut:", ut)
#         # print("us:", us)
        
#         T = T + ut
#         S = S + self.drop(us)

#         # 由于使用了 LLM，这里可以去掉原有的 tensor product attention 逻辑
#         return T, S


    






