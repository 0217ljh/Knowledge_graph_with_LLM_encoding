# from torch import nn
# from .update import Update
# #from .message import Message
# from .message import Message
# class Interaction_Block(nn.Module):
#     '''Interaction layers, each consisting of a message layer and an update layer.'''
#     def __init__(self,
#                  num_features,
#                  act,
#                  head,
#                  num_radial,
#                  irreps_sh,
#                  irreps_T,
#                  dropout
#                  ):
#         super(Interaction_Block,self).__init__()
#         self.message=Message(head=head,num_radial=num_radial,act=act,
#                              num_features=num_features,irreps_sh=irreps_sh)
#         self.update=Update(num_features=num_features,act=act,irreps_mout=self.message.tp.irreps_out,
#                            irreps_T=irreps_T,dropout=dropout)

#     def forward(self,S,T,rbf,sh,index,data):
#         mijt,mijs=self.message(S=S,rbf=rbf,sh=sh,index=index)
#         T,S=self.update(T=T,S=S,mijt=mijt,mijs=mijs,index=index)
#         return S,T

import torch
from torch import nn
from .update import Update
# from .message import Message
from .message_LLM import Message

class Interaction_Block(nn.Module):
    '''Interaction 层，使用 LLM 驱动的 Message 模块'''
    def __init__(self, num_features, act, head, num_radial, irreps_sh, irreps_T, dropout, device='cuda'):
        super(Interaction_Block, self).__init__()
        self.device = device  # 设置设备
        
        # 使用带有 LLM 的 Message 模块
        self.message = Message(
            llm_model_name="distilbert/distilbert-base-cased-distilled-squad",
            num_features=num_features,
            irreps_sh=irreps_sh,
            act=act,
            device=device
        )
        
        # 其他更新层保持不变
        self.update = Update(
            # llm_model_name="bert-base-uncased",
            num_features=num_features,
            act=act,
            irreps_mout=self.message.tp.irreps_out,
            irreps_T=irreps_T,
            dropout=dropout
        )

    
    def forward(self, S, T, rbf, sh, index):
        # 确保所有输入张量在同一设备上
        S = S.to(self.device)
        T = T.to(self.device)
        rbf = rbf.to(self.device)
        sh = sh.to(self.device)
        index = index.to(self.device)
        
        # 调用 message 和 update 模块
        mijt, mijs = self.message(S=S, rbf=rbf, sh=sh, index=index)
        # 确保 mijt 和 mijs 也在同一设备上
        mijt = mijt.to(self.device)
        mijs = mijs.to(self.device)
        # print("mijt:",mijt)
        # print("mijs",mijs)
        # 更新 T 和 S
        #print("S origin:", S)
        #print("T origin:", T)
        T, S = self.update(T=T, S=S, mijt=mijt, mijs=mijs, index=index)
        # S = torch.nan_to_num(S, nan=0.0)  # 将 NaN 值替换为 0
        # T = torch.nan_to_num(T, nan=0.0)
        # print("S update:", S)
        # print("T update:", T)

        # print(torch.isnan(S).any(), torch.isinf(S).any())
        # print(torch.isnan(T).any(), torch.isinf(T).any())
        # print(torch.isnan(rbf).any(), torch.isinf(rbf).any())
        # print(torch.isnan(sh).any(), torch.isinf(sh).any())

        return S, T







