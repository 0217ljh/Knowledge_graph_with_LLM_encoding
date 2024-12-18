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
import random
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

def gen_graph(graphs: list[MSAPygDataset,], 
              #labels_features: list[str,]
              ):
    """
    获取文本信息与光谱信息的复合图。graph_list中包含了划分train/valid/test的掩膜,以及对应的数字标签
    """
    print("get text")
    data = [] 
    # labels_features = [] 
    prompt_edge_text = [] 
    prompt_text = []
    split = {"train": [], "valid": [], "test": []}

    # 读取class_node的文本描述 ****************************************************
    # 注：labels_features中的数字为查找class_node的索引
    now_path = os.path.dirname(__file__)
    category_desc = pd.read_csv(
                os.path.join("task_data\\node\\Task1_Classification\\categories_node.csv"), 
                            sep=","
            ).values
    ordered_desc = []
    label_name = {'Alkyl', 'Alkenyl', 'Alkynyl', 'Ether', 'Hydroxyl', 'Carboalkoxy', 'Ketone', 'Aldehyde', 'Carboxyl', 'furan', 'Carbonate Ester', 'Pyridyl', 'Nitro', 'Nitrile', 'Isocyanate', 'Nitrosooxy', 'Sulfonyl', 'Disulfide', 'Sulfide', 'Isothiocyanate', 'Sulfhydryl', 'Sulfonate', 'Sulfoxide', 'chloro', 'iodo', 'fluoro', 'bromo','Unknown'}# Index order of label
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

    wavelength_ranges={
        "Hydroxyl": [[3650, 3600], [3500, 3200], [1200, 1000], [769, 659]],
        "Carboxyl": [[3000, 2500], [1800, 1650], [1300, 1200], [940, 900]],
        "Ketone": [[1755, 1635]],
        "Aldehyde": [[2820, 2720], [1750, 1680]],
        "Nitro": [[1650, 1500], [1380, 1250], [920, 830]],
        "Carbonate Ester": [[1860, 1740]],
        "Nitrile": [[2260, 2215]],
        "Isocyanate": [[2275, 2250]],
        "Carboalkoxy": [[1750, 1670], [1275, 1163]],
        "Alkynyl": [[3340, 3300], [2250, 2100]],
        #"Phenyl": [[3100, 3000], [2000, 1600], [1600, 1450], [800, 680]],
        "Pyridyl": [[3070, 3020], [1640, 1440], [900, 700]],
        "Ether": [[1300, 1000]],
        #"Amines": [[3550, 3220], [1350, 1000], [1640, 1490], [900, 650]]
        "Sulfonate": [[1370, 1355], [1180, 1140]],
        "Sulfoxide": [[1215, 1040]],
        "Sulfonyl": [[1350, 1100]],
        "Isothiocyanate": [[2140, 2040]],
        "Alkenyl": [[3100, 2970], [1800, 1500], [1000, 675]],
        "furan": [[1200, 1120], [1100, 1050]],
        "Disulfide": [[1450, 1420]],
        "Sulphide": [[700, 590]],
        "Sulfhydryl": [[2700, 2630], [2600, 2500]],
        "fluoro": [[1400, 730]],
        "chloro": [[850, 550]],
        "bromo": [[690, 515]],
        "iodo": [[600, 500]]
        }
        
    # 读取分子表
    df = pd.read_excel('Database/spect-分子式+学名 (Eng).xlsx')
    molecule_functional_groups = dict(zip(df.iloc[:, 0], df.iloc[:, 2]))

    #初始化全局参数
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

        data_dict = sample_graph.to_dict()  # 将原始数据转换为字典,键值包括：
        # if data_dict['edge_index'] == edge_index:
        # print('same')
        edge_index = sample_graph.edge_index 
        data_dict['edge_index'] == edge_index


        # 获取当前化学分子对应的官能团列表
        functional_groups = molecule_functional_groups.get(g.Name, "").split(',')
        #print('functional_groups:',functional_groups)
        # 为strength节点生成标签
        strength_nodes = [(node_name, attr) for node_name, attr in node_attribute.items()if attr.get('raw_text') and 'strength' in attr['raw_text']]
        
        node_label_list = []  # 用于存储strength节点的node_name和标签字典
        for node_name, node_attr in strength_nodes:
            # 从节点文本中提取波长范围
            raw_text = node_attr.get('raw_text')
            strength_match = re.findall(r"Feature.*strength: <(.*?)> of spectral peak ranged from <(.*?)> to <(.*?)>", raw_text)
            if strength_match:
                value, start, end = strength_match[0]  # 提取的值
                start, end = float(start), float(end)  # 将波长范围转换为浮动数值

                # 使用原来的逻辑判断该波峰属于哪个官能团
                matching_groups = []
                for group in functional_groups:
                    group = group.strip()  # 移除空格
                    if group in wavelength_ranges:
                        # 对于每个波长范围，检查是否匹配
                        for range_pair in wavelength_ranges[group]:
                            upper, lower = range_pair
                            if lower <= start <= upper or lower <= end <= upper:
                                matching_groups.append(group)

                # # 如果有多个匹配的官能团，拼接它们的标签
                # if matching_groups:
                #     node_attr['label'] = ','.join(matching_groups)
                # else:
                #     node_attr['label'] = 'Unknown'
    

                # 如果有多个匹配的官能团，拼接它们的标签
                if matching_groups:
                    node_attr['label'] = matching_groups[0]
                else:
                    node_attr['label'] = 'Unknown'

                node_label_id = y_texts_index2id[Labels_text[node_attr['label']]]
                node_label = {node_name: node_label_id}
                node_label_list.append(node_label)

        print('node_label_list:',node_label_list)

        new_data = pyg.data.data.Data(
            x=data_dict['x'],
            edge_index=data_dict['edge_index'],
            node_name=[data_dict['node_name']],
            node_attributes=[data_dict['node_attributes']],
            edge_attributes=[data_dict['edge_attributes']],
            node_ID=node_information_ID,
            edge_ID=edge_information_ID,
            y=node_label_list,
            Name=g.Name if getattr(g, 'Name', None) else None,
            )

        data.append(new_data)  

    prompt_edge_text = [
                    "prompt edge.", 
                    "prompt edge. edge for query graph that is our target",
                    "prompt edge. edge for support graph that is an example", 
                    ] # 边标签文本

    prompt_text = [
                    "prompt node. node classification on the sample's category based on the spectral information and the knowledge graph",
                    "prompt node. few shot task node for node classification that decides whether the query molecule belongs to "
                    "the fuctional group of support molecules.", 
                    ]    # Simple prompt


    # prompt_text = [
    #                  "prompt node. Based on the knowledge graph and spectral data, classify the functional group of the moleculor node. Use the knowledge graph to identify functional group associations and validate the classification decision with relevant spectral evidence. Classify the molecule into categories based on the detected functional group and provide justification using spectral peak information and knowledge graph relationships.",
    #                 "prompt node. few shot task node for node classification that decides whether the query molecule belongs to. Use the knowledge graph to identify functional group associations and validate the classification decision with relevant spectral evidence. Classify the molecule into categories based on the detected functional group and provide justification using spectral peak information and knowledge graph relationships."
    #                 "the fuctional group of support molecules.", 
    #                 ]    # Detailed prompt


    prompt_text_map = {
                    "e2e_node": {"noi_node_text_feat": ["noi_node_text_feat", [1]],
                      "class_node_text_feat": ["class_node_text_feat",
                                                            torch.arange(len(u_y_texts_index_lst))],
                      "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},

                    "lr_node": {
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

class Spec_Node_Dataset(BasePygDataset):
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
                        #print('turn to str:',idx,item)
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
        #data.y = data.y.view(-1)   # 这个和y_label一般情况下都一致，保留这个部分是为了一些可能需要进一步处理的标签
        return data

    def get_idx_split(self):
        """获取数据集划分索引"""
        return self.side_data[0]

    def get_task_map(self):
        """获取任务映射"""
        return self.side_data[1]

    def get_edge_list(self, mode="e2e_node"):
        if mode == "e2e_node":
            return {"f2n": [1, [0]], "n2f": [3, [0]], "n2c": [2, [0]]}
            #return {"f2n": [1, [0]], "n2f": [2, [0]], "n2c": [3, [0]]}

        elif mode == "lr_node":
            return {"f2n": [1, [0]], "n2f": [3, [0]]}