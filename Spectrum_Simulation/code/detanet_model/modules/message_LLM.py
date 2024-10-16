from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
from .LLM_Edge import LLM_Edge_Module  # Assuming LLM_Edge is saved here
from e3nn import o3

class Message(nn.Module):
    '''使用 LLM 代替 Edge_Attention 的 Message 模块'''
    def __init__(self, llm_model_name="bert-base-uncased", num_features=40, irreps_sh=None, act='swish', device='cuda'):
        super(Message, self).__init__()
        self.feature = num_features
        self.device = device  # 确定设备
        
        # 使用 LLM_Edge_Module 作为新的注意力机制
        self.LLM = LLM_Edge_Module(llm_model_name=llm_model_name, num_features=num_features, device=device)
        
        # TensorProduct 的初始化
        irreps_mout = []
        instructions = []
        for i, (_, ir_sh) in enumerate(irreps_sh):
            for ir_out in o3.Irrep('0e') * ir_sh:
                k = len(irreps_mout)
                irreps_mout.append((num_features, ir_out))
                instructions.append((0, i, k, "uvu", True))
        
        self.tp = o3.TensorProduct(
            irreps_in1=o3.Irreps([(num_features, (0, 1))]), 
            irreps_in2=irreps_sh,
            irreps_out=irreps_mout, 
            instructions=instructions, 
            shared_weights=True,
            internal_weights=True
        )
    
    def forward(self, S, rbf, sh, index):
        # 确保所有输入张量在同一设备上
        S = S.to(self.device)
        rbf = rbf.to(self.device)
        sh = sh.to(self.device)
        index = index.to(self.device)
        
        # 使用 LLM 模块计算边特征
        mijs2, mijs = self.LLM(S=S, rbf=rbf, index=index)
        
        # 确保 mijs2 和 sh 在相同设备上
        mijs2 = mijs2.to(self.device)
        sh = sh.to(self.device)
        # 使用 TensorProduct 计算 equivariant 特征
        mijt = self.tp(mijs2, sh)
        
        return mijt, mijs



