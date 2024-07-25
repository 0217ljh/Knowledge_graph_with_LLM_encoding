import os
import sys
import tqdm
import torch
import pickle
import numpy as np
import pandas as pd
import torch_geometric as pyg
from task_data.base_task_dataset import BasePygDataset
from Data.base_collect import MSAPygDataset
from Data.data_collect.Merge_collect import Merge_PygDataset


def get_local_text(dataset:list[MSAPygDataset,],
                   ) -> tuple[
                            list[MSAPygDataset,], 
                            list[str,]
                            ]:
    """
    获取文本信息与光谱信息的复合图
    """
    print("get graph")
    graph_list = []   # 列表：图
    # name_list = []    # 列表：图对应的样本名
    # label_list = []   # 列表：图对应的标签

    FG_graph = dataset[0]
    Spectrum_graph = dataset[1]
    
    for name,spe_graph in tqdm.tqdm(Spectrum_graph.items()):
        test_class = Merge_PygDataset(FG_graph,spe_graph,Name=name)
        connected_dictionary = test_class.preprocess()      # 官能团信息：正相关
        all_Table,isexist,score_dict = test_class.Negative_relevant_information(is_print=False)   # 官能团信息：负相关
        OP = test_class.construct_graph(connected_dictionary,all_Table,is_print=False)        # 生成混合图谱
        fin = test_class.subgraph_extraction(hop=2)    # 生成邻近子图
        graph_list.append(test_class)
        # name_list.append(name)
        # label_list.append(test_class.Label)
    
    return graph_list
    #return graph_list, label_list
    
class Spec_Edge_Dataset(BasePygDataset):
    def gen_data(self):
        task_index = self.task_index
        dataset_name = self.name.split('-')[0]

        with open(f'./Datasets/Data_for_different_Task/{task_index}/{dataset_name}/FG_graph.pkl', 'rb') as f: # 加载处理后的化学知识数据
            FG_graph = pickle.load(f)
        with open(f'./Datasets/Data_for_different_Task/{task_index}/{dataset_name}/Spectrum_graph.pkl', 'rb') as f: # 加载处理后的光谱数据
            Load_spec = pickle.load(f)
        datasets = (FG_graph, Load_spec)
        # 根据任务标签，数据集标签，获取用于进一步处理的函数    
        now_path = os.path.dirname(__file__)
        task_path = next((os.path.join(now_path, i) for i in os.listdir(now_path) if task_index in i), None)
        if task_path:
            module_path = next((os.path.join(task_path, i) for i in os.listdir(task_path) if dataset_name in i), None)
        sys.path.append(module_path)
        from gen_sample import get_local_text, gen_graph
        pyg_graph, texts, split = gen_graph(get_local_text(datasets))
        return [d for d in pyg_graph], texts, split

    def add_raw_texts(self, data_list, texts):
        data, slices = self.collate(data_list)
        data.node_embs = np.array(texts[0])
        data.edge_embs = np.array(texts[1])
        data.class_node_text_feat = np.array(texts[2])
        data.prompt_edge_text_feat = np.array(texts[3])
        data.noi_node_text_feat = np.array(texts[4])
        return data, slices

    def add_text_emb(self, data_list, text_emb):
        """
        notably that we only store unique one node/edge text embeddings and keep the indices
        """
        data, slices = self.collate(data_list)      #  拼接所有样本 
        data.node_embs = text_emb[0]                # 节点文本
        data.edge_embs = text_emb[1]                # 边文本
        data.class_node_text_feat = text_emb[2]     # class文本
        data.prompt_edge_text_feat = text_emb[3]    # prompt,class的边线文本
        data.noi_node_text_feat = text_emb[4]       # prompt文本
        return data, slices

    def text2feature(self,texts,encoder):
        try: 
            judge = isinstance(texts[0], str)
            if judge:
                return self.data2vec(encoder,texts)
        except: 
            judge = isinstance(texts, dict)
            if judge:
                output_data = {key:{} for key,value in texts.edge_attributes.items()}
                collect_edge_name = []
                collect_edge_data = []
                for key, value in texts.items():
                    if value:
                        for name,attr in value.items():
                            collect_edge_name.append(f'{key}' + '_' + name)
                            collect_edge_data.append(attr)
                            #collect_data[name] = data2vec(encoder,attr)
                encode_data = self.data2vec(encoder,collect_edge_data)
                #print(np.array(collect_edge_name).shape)
                for i in range(len(collect_edge_name)):
                        output_data[eval(collect_edge_name[i].split('_')[0])][collect_edge_name[i].split('_')[1]] = encode_data[i]
                        #output_data[key] = collect_data
                return output_data
        return [self.text2feature(t,encoder) for t in texts]
    
    def data2vec(self,encoder, data: list[str]) -> torch.Tensor:
        r"""
        Encode a list of string to a len(data)-by-d matrix, where d is the output dimension of the LLM.
        """
        if encoder is None:
            raise NotImplementedError("LLM encoder is not defined")
        if data is None:
            return None
        embeddings = encoder.encode(data).cpu().numpy()
        return embeddings
    
    def get(self, index):
        data = super().get(index)
        #node_feat = self.node_embs[data.x.numpy()]
        #edge_feat = self.edge_embs[data.xe.numpy()]
        # 获取ID

        # 将data.node_ID中的所有列表拼接为1个完整列表
        node_feat = self.node_embs[data.node_ID]   # 从堆叠的编码文本中抽出对应的节点信息
        edge_feat = []
        for edge_dict in data.edge_ID:             # 从堆叠的编码文本中抽出对应的边信息
            for name,attr in edge_dict.items():
                for attr_name,edge_label in attr.items():
                    edge_feat.append(self.edge_embs[edge_label][0])
        edge_feat = np.array(edge_feat)

        data.node_text_feat = node_feat
        data.edge_text_feat = edge_feat
        data.y = data.y_label.view(-1)   # 这个和y_label一般情况下都一致，保留这个部分是为了一些可能需要进一步处理的标签
        return data

    def get_idx_split(self):
        """获取数据集划分索引"""
        return self.side_data[0]

    def get_task_map(self):
        """获取任务映射"""
        return self.side_data[1]

    def get_edge_list(self, mode="e2e_graph"):
        if mode == "e2e_graph":
            return {"f2n": [1, [0]], "n2f": [3, [0]], "n2c": [2, [0]]}
            #return {"f2n": [1, [0]], "n2f": [2, [0]], "n2c": [3, [0]]}

        elif mode == "lr_graph":
            return {"f2n": [1, [0]], "n2f": [3, [0]]}
