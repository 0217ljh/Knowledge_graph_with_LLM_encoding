import os
import tqdm
import torch
import pickle
import numpy as np
import pandas as pd
import torch_geometric as pyg
from task_data.base_task_dataset import BasePygDataset
from Data_process.base_collect import MSAPygDataset
from Data_process.data_collect.Merge_collect import Merge_PygDataset


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
        data = []               # 列表：图
        # labels_features = [] 
        prompt_edge_text = []   # 列表：连接prompt和图的边文本
        prompt_text = []        # 列表：连接prompt和图的节点文本
        split = {"train": [], "valid": [], "test": []}   # 存放数据集中train，valid，test部分的索引

        # 读取class_node的文本描述 ****************************************************
        # 注：labels_features中的数字为查找class_node的索引
        now_path = os.path.dirname(__file__)
        category_desc = pd.read_csv(
                    os.path.join('task_data\graph\Task4-Classification\Small_Molecules\FS_label.csv'), 
                                sep=","
                ).values
        ordered_desc = []
        #label_name = ['Hydrocarbons','GroupsContainingBoron','GroupsContainingMetals','GroupsContainingPhosphorus','GroupsContainingHalogen','GroupsContainingNitrogen','GroupsContainingOxygen','GroupsContainingSulfur']
        label_name = ['Halogen compounds','Non-halogen compounds'] # Index order of label
        for i, label in enumerate(label_name):
            true_ind = label == category_desc[:, 0]  # 定位[False,...,True,...,False]
            ordered_desc.append((label, category_desc[:, 1][true_ind]))
        # Labels_text = [
        #         "prompt node. Type of compound and description: " #literature category and description: "
        #         + desc[0]
        #         + " . "
        #         + desc[1][0]
        #         for desc in ordered_desc
        #     ] # 类别分类标签文本
        
        Labels_text = {desc[0]:
                "prompt node. Type of compound and description: " 
                + desc[0]
                + " . "
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
        u_edge_texts_lst.append('feature edge. general connection')
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

        # prompt_text = [
        #                 "prompt node. graph classification on the sample's category based on the spectral information and the knowledge graph",
        #                 "prompt node. few shot task node for graph classification that decides whether the query molecule belongs to "
        #                 "the class of support molecules.", 
        #                 ]    # 提问prompt文本
        

         # 详细版prompt（增加了具体的peak范围，以及结合引导任务）、
        # prompt_text = [
        #                 "prompt node. Based on the molecular graph and spectral data, classify whether the molecule contains the 'Phenyl' group. Analyze the spectral peaks in the range of [1600, 1500] nm and classify the molecule based on the presence of these peaks. Additionally, use the knowledge graph to identify functional group associations and validate the classification decision with relevant spectral evidence. Classify the molecule into categories based on the detected functional group and provide justification using spectral peak information and knowledge graph relationships."
        #                 ]
        prompt_text = [
                "prompt node. few shot task node for graph classification that decides whether the query molecule belongs to."
                "the class of support molecules."
                ]


        prompt_text_map = {
                        "e2e_graph": {
                            "noi_node_text_feat": ["noi_node_text_feat", [0]],
                            "class_node_text_feat": ["class_node_text_feat",
                                                                torch.arange(len(u_y_texts_index_lst))],
                            "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},

                        "lr_graph": {
                            "noi_node_text_feat": ["noi_node_text_feat", [1]],
                            "class_node_text_feat": ["class_node_text_feat",
                                                                torch.arange(len(u_y_texts_index_lst))],
                            "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]]}}

        ret = (
                data, 
                [u_node_texts_lst, u_edge_texts_lst, u_y_texts_index_lst, prompt_edge_text, prompt_text, ],
                [split, prompt_text_map],
                )
        return ret

class Spec_Graphcls_Dataset_FS(BasePygDataset):
    def gen_data(self):
        with open('./Datasets/FS_data/FG_graph.pkl', 'rb') as f: # 加载处理后的化学知识数据
            FG_graph = pickle.load(f)
        with open('./Datasets/FS_data/Spectrum_graph.pkl', 'rb') as f: # 加载处理后的光谱数据
            Load_spec = pickle.load(f)
        datasets = (FG_graph, Load_spec)
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
        Since the majority of node/edge text embeddings are repeated, we only store unique
        ones, and keep the indices.
        """
        data, slices = self.collate(data_list)      #  拼接所有样本 
        data.node_embs = text_emb[0]                # 节点文本
        data.edge_embs = text_emb[1]                # 边文本
        data.class_node_text_feat = text_emb[2]     # class文本
        data.prompt_edge_text_feat = text_emb[3]    # prompt,class的边线文本
        data.noi_node_text_feat = text_emb[4]       # prompt文本
        return data, slices

    def text2feature(self, texts, encoder):
        try: 
            # 检查输入的第一个元素是否为字符串
            judge = isinstance(texts[0], str)
            if judge:
                print("Processing text input.")
                return self.data2vec(encoder, texts)  # 调用 data2vec 处理文本
        except Exception as e: 
            print(f"Error in processing string input: {e}")
            
        try:
            # 检查输入是否为字典类型
            judge = isinstance(texts, dict)
            if judge:
                print("Processing dict input.")
                output_data = {key: {} for key, value in texts.edge_attributes.items()}
                collect_edge_name = []
                collect_edge_data = []
                
                for key, value in texts.items():
                    if value:
                        for name, attr in value.items():
                            collect_edge_name.append(f'{key}' + '_' + name)
                            collect_edge_data.append(attr)
                
                if collect_edge_data:
                    print(f"Collected edge data: {len(collect_edge_data)} items.")
                    encode_data = self.data2vec(encoder, collect_edge_data)  # 调用 data2vec 处理边数据
                    
                    if encode_data is not None:
                        print(f"Encoded data shape: {encode_data.shape}")
                        for i in range(len(collect_edge_name)):
                            try:
                                key_part = eval(collect_edge_name[i].split('_')[0])
                                name_part = collect_edge_name[i].split('_')[1]
                                output_data[key_part][name_part] = encode_data[i]
                            except Exception as e:
                                print(f"Error in processing edge {collect_edge_name[i]}: {e}")
                    else:
                        print("Error: Encoded data is None.")
                else:
                    print("No edge data to encode.")
                return output_data
        except Exception as e:
            print(f"Error in processing dict input: {e}")
        
        # 递归调用 text2feature 处理文本列表
        try:
            return [self.text2feature(t, encoder) for t in texts]
        except Exception as e:
            print(f"Error in recursive text2feature call: {e}")
            return None
    
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
                    invalid_data_indices.append(idx)  # 记录不合法数据的索引
                    processed_data.append('Unknown')  # 替换为 'Unknown'
                else:
                    processed_data.append(item)
            
            # 如果有不合法的数据，输出它们的索引
            if invalid_data_indices:
                print("Error: The following data entries were not strings and have been replaced with 'Unknown':")
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