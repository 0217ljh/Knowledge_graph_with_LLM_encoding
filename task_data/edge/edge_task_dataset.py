import os
import sys
import tqdm
import torch
import pickle
import numpy as np
import pandas as pd
import torch_geometric as pyg
from task_data.base_task_dataset import BasePygDataset
from Data_process.base_collect import MSAPygDataset
from Data_process.data_collect.Merge_collect import Merge_PygDataset
import re

def get_related_fp_from_strength(node_list,sub_graph):
    strength_dict = {}
    for i in node_list:
        index = re.sub(u"([^\u0030-\u0039])","",i[0])
        link_fp = [i for i in sub_graph.successors(f'wavelength{index}') if 'Range' in i]
        interval = []
        for j in link_fp:
            interval.append([k for k in sub_graph.predecessors(j) if 'wave' not in k])
        interval = [k[0] for k in interval]
        strength_dict[i[0]] = interval
    return strength_dict

def get_local_text(dataset: list[MSAPygDataset,]) -> tuple[
        list[MSAPygDataset,], 
        list[str,]
    ]:
    """
    获取文本信息与光谱信息的复合图
    """
    print("get graph")
    graph_list = []
    total_pairs = 0  # 用于统计所有子图中的节点对数量

    FG_graph = dataset[0]
    Spectrum_graph = dataset[1]

    #融合图谱，添加配置
    for name, spe_graph in tqdm.tqdm(Spectrum_graph.items()):
        test_class = Merge_PygDataset(FG_graph, spe_graph, Name=name)
        connected_dictionary = test_class.preprocess()      # 官能团信息：正相关
        all_Table, isexist, score_dict = test_class.Negative_relevant_information(is_print=False)   # 官能团信息：负相关
        OP = test_class.construct_graph(connected_dictionary, all_Table, is_print=False)        # 生成混合图谱
        fin = test_class.subgraph_extraction(hop=2)    # 生成邻近子图

        # generate edge sub_graph
        # Step1: get strength node
        pair_dict = {}
        molecule_name = name
        strength_node = [(node_name, attr) for node_name, attr in Spectrum_graph[molecule_name]['graph'].nodes(data=True) if 'strength' in node_name]
        
        # 从strength_node中获取两两的分组
        for i in range(len(strength_node)):
            for j in range(i + 1, len(strength_node)):
                pair_dict[(strength_node[i][0], strength_node[j][0])] = (strength_node[i][1], strength_node[j][1])
        
        # Step2: generate sample config
        sample_list = []
        for index, key in enumerate(pair_dict):
            sample_config = {'Molecule': molecule_name, 'Combination_Index': index, 'Peak': [key[0], key[1]], 'Functional_Group': [], 'Label': [], 'node_index': []}
            sample_list.append(sample_config)

        # Step3: get the functional group associated with peak combination
        strength_dict = get_related_fp_from_strength(strength_node, OP)

        # Step4: compare the peak pair and get the label of sample
        for i in sample_list:
            for sub_peak in i['Peak']:
                i['Functional_Group'].append(strength_dict[sub_peak])
            # compare the two elements in i['Functional_Group'] whether they have the same element
            i['Label'] = 'Yes' if len(set(i['Functional_Group'][0]) & set(i['Functional_Group'][1])) > 0 else 'No'

        test_class.sample_list = sample_list
        test_class.Mask = ''     # clear wrong Mask
        test_class.Label = ''    # clear wrong Label

        # Step5: generate mask for all the samples in sample_list, the ratio should be 6:2:2 of train:valid:test
        # delete all the sample just with one peak, such as: pair_dict = {}
        if test_class.sample_list:
            graph_list.append(test_class)

        # 统计当前子图的节点对数量
        total_pairs += len(pair_dict)  # 累加当前子图的节点对数

    # print(f"Total number of node pairs: {total_pairs}")  # 输出总节点对数
    # print(f"Total number of subgraphs: {len(graph_list)}")  # 输出子图的总数量
    return graph_list

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
                os.path.join('task_data\edge\Task2-Edge_relation\Small_Molecules\categories_Yes_or_No.csv'), 
                            sep=","
            ).values
    ordered_desc = []
    label_name = ['No','Yes'] # Index order of label
    for i, label in enumerate(label_name):
        true_ind = label == category_desc[:, 0]  # 定位[False,...,True,...,False]
        ordered_desc.append((label, category_desc[:, 1][true_ind]))
    
    Labels_text = {desc[0]:
            "prompt node. " 
            + desc[0]
            + ". "
            + desc[1][0]
            for desc in ordered_desc
    } # 类别分类标签文本

    # 提取所有图中的节点，边，标签文本信息，一起存放，方便做统一的编码处理 ****************************************************
    node_texts = []      # 存放所有节点信息
    edge_texts = []      # 存放所有边信息
    y_texts_index = list(Labels_text.values())   # 存放所有标签信息
    cal_node2=0
    cal_edge2=0
    global_map2=None
    for g in graphs:
        sample_graph,cal_node2,cal_edge2,global_map2 = g.GP.from_networkx_2_pyg_global(g.GP.finnal_graph,cal_node2,cal_edge2,global_map2)    # 转换为pyg格式，并整理所有相关信息
        node_attribute = sample_graph.node_attributes.copy()          # 获取节点属性
        edge_attribute = sample_graph.edge_attributes.copy()          # 获取边属性
        #y_texts_index.append(Labels_text[g.Label])                    # 获取标签信息,之后还是改成根据字典进行获取好一点。0对应Label_name的第一个类别，1对应第二个类别
        node_text = [node_attribute[i]['raw_text'] for i in node_attribute]  # 节点信息统一以属性'raw_text'进行保存，要求所有节点具有该属性。
        for i in node_text: # all node text
            node_texts.append(i)
        for key,value in edge_attribute.items():
            if value:
                for name,edge_attr in value.items():
                    edge_texts.append(edge_attr)    # 部分边属性可能为空，需要进行判断。边信息统一以属性'relation'进行保存。
                
    # 去除列表中的重复部分，方便后续编码处理。
    # unique_node_texts = set(node_texts)
    # unique_edge_texts = set(edge_texts)
    unique_y_texts_index = set(y_texts_index)

    unique_node_texts = node_texts
    unique_edge_texts = edge_texts
    #unique_y_texts_index = y_texts_index
    

    u_node_texts_lst = list(unique_node_texts)       # 列表：所有图中的节点信息，唯一存在

    u_edge_texts_lst = list(unique_edge_texts)       # 列表：所有图中的边信息，唯一存在
    u_edge_texts_lst.append('feature edge. general connection')
    u_y_texts_index_lst = list(unique_y_texts_index) # 列表：所有图中的标签信息，唯一存在
    u_y_texts_index_lst.sort(key = list(Labels_text.values()).index)
    node_texts2id = {v: i for i, v in enumerate(u_node_texts_lst)}         # 该字典将节点信息转换为ID
    edge_texts2id = {v: i for i, v in enumerate(u_edge_texts_lst)}         # 该字典将边信息转换为ID
    y_texts_index2id = {v: i for i, v in enumerate(u_y_texts_index_lst)}   # 该字典将标签信息转换为ID，这里其实是和原来的g.Label是一样的
    
    cal_node1=0
    cal_edge1=0
    global_map1=None
    # 将所有图中的相关信息用ID进行保存 ****************************************************
    for num, g in enumerate(graphs):
        sample_graph,cal_node1,cal_edge1,global_map1 = g.GP.from_networkx_2_pyg_global(g.GP.finnal_graph,cal_node1,cal_edge1,global_map1)
        node_attribute = sample_graph.node_attributes.copy()
        edge_attribute = sample_graph.edge_attributes.copy()
    
        node_information_ID = [node_texts2id[node_attribute[i]['raw_text']] for i in node_attribute] # trasnsormer the node information to ID

        edge_information_ID = [] # 将边信息转换为ID
        for key,value in edge_attribute.items():
            sample_edge_dict = {}
            if value: # 如果边属性存在，则通过字典转换为ID
                for name,edge_attr in value.items():
                    sample_edge_dict[key] = {name :[edge_texts2id[value[name]]]}
            else: # Using the same format for the edge without information
                sample_edge_dict[key] = {name :[edge_texts2id['feature edge. general connection']]}  
            edge_information_ID.append(sample_edge_dict)
    
        #nx_z = pyg.utils.to_networkx(sample_graph,to_undirected=False,to_multi=True)  # 非无向图，可存在多连接

        edge_index = sample_graph.edge_index  
        sub_graph_config = g.sample_list # get the edge index of correpsonding strength node
        for i in sub_graph_config:
            Peak_pair = i['Peak']
            node_index = []
            for j in Peak_pair:
                node_index.append([k for k, v in sample_graph.node_name.items() if v== j][0])
            i['node_index'] = node_index
            desc_text = Labels_text[i['Label']]
            i['Label_index'] = y_texts_index2id[desc_text]

        data_dict = sample_graph.to_dict()  # 将原始数据转换为字典,键值包括：
        # if data_dict['edge_index'] == edge_index:
        # print('same')
        data_dict['edge_index'] == edge_index
        new_data = pyg.data.data.Data(
            x=data_dict['x'],
            edge_index=data_dict['edge_index'],
            node_name=[data_dict['node_name']],
            node_attributes=[data_dict['node_attributes']],
            edge_attributes=[data_dict['edge_attributes']],
            node_ID=node_information_ID,
            edge_ID=edge_information_ID,
            y_fake=sub_graph_config,
            Name=g.Name if getattr(g, 'Name', None) else None
            )

        data.append(new_data)

    prompt_edge_text = [
                    "prompt edge.", 
                    "prompt edge. edge for query graph that is our target",
                    "prompt edge. edge for support graph that is an example", 
                    ] # 边标签文本

    prompt_text = [
                    "prompt node. link prediction task to decide if two spectral peak nodes in a graph represent the same functional group based on their spectral features and the knowledge graph.",
                    "prompt node. few-shot task for link prediction that determines if two spectral peak nodes belong to the same functional group, using spectral information and edge relationships in the graph.", 
                    ]    # 提问prompt文本
    
    prompt_text = [
    "prompt node. The task involves link prediction for a graph where the nodes represent spectral peaks, and the edges represent relationships between these peaks based on their spectral features. The goal is to determine if two spectral peak nodes in the graph belong to the same functional group. This decision is based on analyzing the spectral features of the nodes and leveraging a knowledge graph that provides additional contextual information about these peaks. The task aims to predict whether the edge between two given spectral peaks indicates that they share the same chemical functional group or if they represent different groups, by analyzing their characteristics and relationships within the graph.",
    
    "prompt node. This prompt outlines a link prediction task, focusing on the determination of whether two spectral peak nodes, represented in a graph structure, belong to the same functional group. The model should use both the spectral features of the peaks (such as peak intensity, position, and shape) and the relationships between them as indicated by the graph edges. The task is particularly challenging because it requires integrating complex spectral data with knowledge graph information, which could include external databases or contextual information about known functional groups. The model will need to learn to identify patterns in the spectral data that indicate similarity or dissimilarity between peaks, and infer the likelihood that they share a functional group. The output will be a binary classification indicating whether the two nodes (spectral peaks) should be connected by an edge representing the same functional group.",
]       # Detailed prompt


    prompt_text_map = {
                    "e2e_edge": {"noi_node_text_feat": ["noi_node_text_feat", [1]],
                      "class_node_text_feat": ["class_node_text_feat",
                                                            torch.arange(len(u_y_texts_index_lst))],
                      "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},

                    "lr_edge": {
                        "noi_node_text_feat": ["noi_node_text_feat", [1]],
                        "class_node_text_feat": ["class_node_text_feat",
                                                            torch.arange(len(u_y_texts_index_lst))],
                        "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]]}}

    ret = (
            data, 
            [
                u_node_texts_lst, 
                u_edge_texts_lst, 
                u_y_texts_index_lst, 
                prompt_edge_text, 
                prompt_text, 
                ],
            [
                split, 
                prompt_text_map
                ],
            )
    return ret

class Spec_Edge_Dataset(BasePygDataset):
    def gen_data(self):
        #task_index = self.task_index
        #dataset_name = self.name.split('-')[0]

        with open(f'Datasets\Base_data\FG_graph.pkl', 'rb') as f: # 加载处理后的化学知识数据
            FG_graph = pickle.load(f)
        with open(f'Datasets\Base_data\Spectrum_graph.pkl', 'rb') as f: # 加载处理后的光谱数据
            Load_spec = pickle.load(f)
        datasets = (FG_graph, Load_spec)
        # # 根据任务标签，数据集标签，获取用于进一步处理的函数    
        # now_path = os.path.dirname(__file__)
        # task_path = next((os.path.join(now_path, i) for i in os.listdir(now_path) if task_index in i), None)
        # if task_path:
        #     module_path = next((os.path.join(task_path, i) for i in os.listdir(task_path) if dataset_name in i), None)
        # sys.path.append(module_path)
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
    
    # def data2vec(self,encoder, data: list[str]) -> torch.Tensor:
    #     r"""
    #     Encode a list of string to a len(data)-by-d matrix, where d is the output dimension of the LLM.
    #     """
    #     if encoder is None:
    #         raise NotImplementedError("LLM encoder is not defined")
    #     if data is None:
    #         return None
    #     embeddings = encoder.encode(data).cpu().numpy()
    #     return embeddings
    

    def data2vec(self, encoder, data: list[str]) -> torch.Tensor:
        r"""
        Encode a list of strings to a len(data)-by-d matrix, where d is the output dimension of the LLM.
        """
        try:
            if encoder is None:
                raise NotImplementedError("LLM encoder is not defined")
            
            if data is None:
                print("Warning: data is None.")
                return None

            # 检查每条数据是否为字符串，记录不合法的数据和替换后的数据
            processed_data = []
            invalid_data_indices = []

            for idx, item in enumerate(data):
                if not isinstance(item, str):
                    try:
                        # 尝试将非字符串数据转换为字符串
                        item = str(item)
                        print('turn to str:',idx,item)
                    except Exception as e:
                        invalid_data_indices.append(idx)  # 记录不合法数据的索引
                        item = 'Unknown'  # 替换为 'Unknown'
                    
                    processed_data.append(item)
                else:
                    processed_data.append(item)
            
            # 如果有不合法的数据，输出它们的索引
            if invalid_data_indices:
                print("Error: The following data entries could not be converted to strings and have been replaced with 'Unknown':")
                for idx in invalid_data_indices:
                    print(f"Index {idx}: {data[idx]} (type: {type(data[idx])})")
            
            # 进行编码
            embeddings = encoder.encode(processed_data).cpu().numpy()  # 一次性编码
            return embeddings

        except Exception as e:
            print(f"General error in data2vec: {e}")
            return None
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
        #data.y = data.y_label.view(-1)   # 这个和y_label一般情况下都一致，保留这个部分是为了一些可能需要进一步处理的标签
        return data

    def get_idx_split(self):
        """获取数据集划分索引"""
        return self.side_data[0]

    def get_task_map(self):
        """获取任务映射"""
        return self.side_data[1]

    def get_edge_list(self, mode="e2e_edge"):
        if mode == "e2e_edge":
            return {"f2n": [1, [0]], "n2f": [3, [0]], "n2c": [2, [0]]}
            #return {"f2n": [1, [0]], "n2f": [2, [0]], "n2c": [3, [0]]}

        elif mode == "lr_edge":
            return {"f2n": [1, [0]], "n2f": [3, [0]]}
