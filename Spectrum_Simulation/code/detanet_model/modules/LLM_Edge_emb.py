import torch
import torch.nn as nn

class LLM_Edge_Module(nn.Module):
    def __init__(self,num_feature, embedding_dim, device='cpu'):
        super(LLM_Edge_Module, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_feature = num_feature
        
        # 定义原子特征和边特征的统一嵌入层，输入都是 num_feature 维度
        self.feature_embedding = nn.Linear(num_feature, embedding_dim).to(self.device)
        
        # 你可以根据需要定义后续的LLM模块，这里以全连接层为例
        self.fc = nn.Linear(embedding_dim * 3, 128).to(self.device)  # 将嵌入后的特征拼接后送入全连接层
        self.output_layer = nn.Linear(128, 1).to(self.device)  # 输出层，用于生成最终的预测

    def format_input_for_llm(self, S, rbf, index):
        """
        使用嵌入层处理原子特征和边特征，并返回组合后的向量输入。
        Args:
            S: 节点（原子）的特征矩阵 (num_atoms, num_feature)
            rbf: 边的特征矩阵 (num_edges, num_feature)
            index: 边的索引 (2, num_edges)，包含连接的原子对的索引
        Returns:
            combined_inputs: 拼接后的特征向量，(num_edges, embedding_dim * 3)
        """
        i, j = index  # i和j表示连接的两个原子的索引

        input_vectors = []
        for idx_i, idx_j in zip(i, j):
            # 获取原子i和原子j的特征，以及它们之间的边特征
            atom_i_features = S[idx_i]
            atom_j_features = S[idx_j]
            edge_features = rbf[idx_i]

            # 嵌入层处理特征（统一使用 feature_embedding 处理原子和边特征）
            atom_i_embedded = self.feature_embedding(atom_i_features)
            atom_j_embedded = self.feature_embedding(atom_j_features)
            edge_embedded = self.feature_embedding(edge_features)

            # 将嵌入的特征拼接成一个大向量
            combined_features = torch.cat([atom_i_embedded, atom_j_embedded, edge_embedded], dim=-1)
            input_vectors.append(combined_features)

        # 拼接所有的特征向量，形成一个张量
        combined_inputs = torch.stack(input_vectors)
        return combined_inputs

    def forward(self, S, rbf, edge_index):
        """
        前向传播，处理节点和边特征，并返回预测结果。
        Args:
            S: 节点（原子）的特征矩阵
            rbf: 边的特征矩阵
            edge_index: 边的索引
        Returns:
            output: 最终的输出预测结果
        """
        # 处理输入特征
        combined_inputs = self.format_input_for_llm(S, rbf, edge_index)

        # 送入全连接层处理
        hidden = self.fc(combined_inputs)
        hidden = torch.relu(hidden)

        # 最终的输出
        output = self.output_layer(hidden)
        return output

