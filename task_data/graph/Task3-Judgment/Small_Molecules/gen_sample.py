import os
import tqdm
import torch
import numpy as np
import pandas as pd
import torch_geometric as pyg
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

def gen_graph(graphs: list[MSAPygDataset,], 
              #labels_features: list[str,]
              ):
        """
        获取文本信息与光谱信息的复合图。graph_list中包含了划分train/valid/test的掩膜,以及对应的数字标签
        """
        print("get text")
        data = []               # 列表：图
        # labels_features = [] 
        prompt_edge_text = []   # 列表：连接prompt和图的边文本
        prompt_text = []        # 列表：连接prompt和图的节点文本
        split = {"train": [], "valid": [], "test": []}   # 存放数据集中train，valid，test部分的索引

        # 读取class_node的文本描述 ****************************************************
        # 注：labels_features中的数字为查找class_node的索引
        now_path = os.path.dirname(__file__)
        category_desc = pd.read_csv(
                    os.path.join(now_path, "categories_Yes_or_No.csv"), 
                                sep=","
                ).values
        ordered_desc = []
    
        label_name = ['No','Yes']   # index used to get label 
        for i, label in enumerate(label_name):
            true_ind = label == category_desc[:, 0]  # 定位[False,...,True,...,False]
            ordered_desc.append((label, category_desc[:, 1][true_ind]))
        
        Labels_text = {desc[0]:
                "prompt node. Judgment: " 
                + desc[0]
                + ". "
                + desc[1][0]
                for desc in ordered_desc
        } # 类别分类标签文本

        # 提取所有图中的节点，边，标签文本信息，一起存放，方便做统一的编码处理 ****************************************************
        node_texts = []      # 存放所有节点信息
        edge_texts = []      # 存放所有边信息
        y_texts_index = []   # 存放所有标签信息

        for g in graphs:
            sample_graph = g.GP.from_networkx_2_pyg(g.GP.finnal_graph)    # 转换为pyg格式，并整理所有相关信息
            node_attribute = sample_graph.node_attributes.copy()          # 获取节点属性
            edge_attribute = sample_graph.edge_attributes.copy()          # 获取边属性
            y_texts_index.append(Labels_text[g.Label])                    # 获取标签信息,之后还是改成根据字典进行获取好一点。0对应Label_name的第一个类别，1对应第二个类别
            node_text = [node_attribute[i]['raw_text'] for i in node_attribute]  # 节点信息统一以属性'raw_text'进行保存，要求所有节点具有该属性。
            for i in node_text:
                node_texts.append(i)
            for key,value in edge_attribute.items():
                if value:
                    for name,edge_attr in value.items():
                        edge_texts.append(edge_attr)    # 部分边属性可能为空，需要进行判断。边信息统一以属性'relation'进行保存。
                    
        # 去除列表中的重复部分，方便后续编码处理。
        unique_node_texts = set(node_texts)
        unique_edge_texts = set(edge_texts)
        unique_y_texts_index = set(y_texts_index)

        u_node_texts_lst = list(unique_node_texts)       # 列表：所有图中的节点信息，唯一存在
        u_edge_texts_lst = list(unique_edge_texts)       # 列表：所有图中的边信息，唯一存在
        u_edge_texts_lst.append('feature edge. general connection') # blank edge information filled with this
        u_y_texts_index_lst = list(unique_y_texts_index) # 列表：所有图中的标签信息，唯一存在
        u_y_texts_index_lst.sort(key = list(Labels_text.values()).index)

        node_texts2id = {v: i for i, v in enumerate(u_node_texts_lst)}         # 该字典将节点信息转换为ID
        edge_texts2id = {v: i for i, v in enumerate(u_edge_texts_lst)}         # 该字典将边信息转换为ID
        y_texts_index2id = {v: i for i, v in enumerate(u_y_texts_index_lst)}   # 该字典将标签信息转换为ID，这里其实是和原来的g.Label是一样的
        

        # 将所有图中的相关信息用ID进行保存 ****************************************************
        for num, g in enumerate(graphs):
            sample_graph = g.GP.from_networkx_2_pyg(g.GP.finnal_graph)
            node_attribute = sample_graph.node_attributes.copy()
            edge_attribute = sample_graph.edge_attributes.copy()

            node_information_ID = [node_texts2id[node_attribute[i]['raw_text']] for i in node_attribute] # 将节点信息转换为ID

            edge_information_ID = [] # 将边信息转换为ID
            for key,value in edge_attribute.items():
                sample_edge_dict = {}
                if value: # 如果边属性存在，则通过字典转换为ID
                    for name,edge_attr in value.items():
                        sample_edge_dict[key] = {name :[edge_texts2id[value[name]]]}
                else: # 如果边属性不存在，则用特殊字符进行填充
                    #sample_edge_dict[key] = {'without' : [-9999]}
                    sample_edge_dict[key] = {name :[edge_texts2id['feature edge. general connection']]}  # 对于无信息的边线，指明这条连线属于一般的连线，采用同一的格式
                edge_information_ID.append(sample_edge_dict)
        
            #nx_z = pyg.utils.to_networkx(sample_graph,to_undirected=False,to_multi=True)  # 非无向图，可存在多连接
            edge_index = sample_graph.edge_index  # 获取连接的边索引
            #edge_index = torch.tensor(list(nx_z.edges())).T  # 获取连接的边索引
            data_dict = sample_graph.to_dict()  # 将原始数据转换为字典,键值包括：
            # if data_dict['edge_index'] == edge_index:
            # print('same')
            data_dict['edge_index'] == edge_index
            new_data = pyg.data.data.Data(
                x=data_dict['x'],
                edge_index=data_dict['edge_index'],
                node_name = [data_dict['node_name']],
                node_attributes = [data_dict['node_attributes']],
                edge_attributes = [data_dict['edge_attributes']],
                node_ID = node_information_ID,
                edge_ID = edge_information_ID,
                y_label = y_texts_index2id[Labels_text[g.Label]],
                # 如果g中存在name属性，则从g中获取为new_data添加name属性
                Name = g.Name if getattr(g, 'Name', None) else None
                )

            data.append(new_data)
            
            split[g.Mask].append(num)  # 根据train/valid/test分配该样本

        prompt_edge_text = [
                        "prompt edge.", 
                        "prompt edge. edge for query graph that is our target",
                        "prompt edge. edge for support graph that is an example", 
                        ] # 边标签文本

        prompt_text = [
                        "prompt node. graph judgement about the existence of 'Phenyl' in the sample's molecular structure is based on the spectrum information and the knowledge graph",
                        "prompt node. few shot task node for graph classification that decides whether the query molecule belongs to "
                        "the class of support molecules.", 
                        ]    # 提问prompt文本
            

        prompt_text_map = {
                        "e2e_graph": {
                            "noi_node_text_feat": ["noi_node_text_feat", [0]],
                            "class_node_text_feat": ["class_node_text_feat",
                                                                torch.arange(len(u_y_texts_index_lst))],
                            "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},

                        "lr_graph": {"noi_node_text_feat": ["noi_node_text_feat", [1]],
                                    "class_node_text_feat": ["class_node_text_feat",
                                                                torch.arange(len(u_y_texts_index_lst))],
                                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]]}}

        ret = (
                data, 
                [u_node_texts_lst, u_edge_texts_lst, u_y_texts_index_lst, prompt_edge_text, prompt_text, ],
                [split, prompt_text_map],
                )
        return ret