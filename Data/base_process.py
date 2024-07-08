from abc import ABC, abstractmethod
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
import matplotlib.pyplot as plt

class GraphTextDataset():
    """
    对数据集中进行图处理的基类
    """

    def __init__(self, 
                 #graph, 
                 #process_label_func: Callable, 
                 **kwargs):
        """
        Args:
            graph: Main graph objects, one single graph for single graph dataset, and list of graphs
                        for list of graphs.
            process_label_func: a Callable function that process the labels from original datasets to accommodate
                                    different tasks.
            **kwargs: additional arguments.
        """
        self.graph = None
        self.graph_dataset = {}
        #self.prompt_edge_emb = None
        #self.g = graph
        #self.process_label_func = process_label_func
        #self.kwargs = kwargs
        #self.llm_tokenizer = None
        #self.llm_max_length = None
        ##self.edge_mode = 1
        #if "prompt_edge_list" in kwargs:
        #    self.prompt_edge_list = kwargs["prompt_edge_list"]
        #else:
        #    self.prompt_edge_list = {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0], "c2n": [4, 0]},
        #if "no_class_node" in kwargs and kwargs["no_class_node"]:
        #    self.no_class_node = True
        #else:
        #    self.no_class_node = False
    

    def __getitem__(self, index):
        pass
        #feature_graph = self.make_feature_graph(index)
        #prompt_graph = self.make_prompted_graph(feature_graph)
        #ret_data = self.to_pyg(feature_graph, prompt_graph)
        #if "walk_length" in self.kwargs and self.kwargs["walk_length"] is not None:
        #    ret_data.rwpe = scipy_rwpe(ret_data, self.kwargs["walk_length"])
        #return ret_data
    
    @abstractmethod
    def construct_graph(self):
        pass
    

    def from_networkx_2_pyg(self,
                             graph=None
                             ):
        """
        从networkx转换为pytorch_geometric
        """
        
        # 将G保存为统一的pyg格式
        # 创建节点特征矩阵
        if graph is None:
            graph = self.graph
        x = torch.ones((graph.number_of_nodes(),1), dtype=torch.float)

        # 获取图G邻接矩阵的稀疏表示
        adj = nx.to_scipy_sparse_array(graph).tocoo()

        # 获取非零元素行索引
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        # 获取非零元素列索引
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)

        # 将行和列进行拼接，shape变为[2, num_edges], 包含两个列表，第一个是row, 第二个是col
        edge_index = torch.stack([row, col], dim=0)

        data = Data(x=x, edge_index=edge_index)

        node_name = {}
        node_attr = {}
        edge_attr = {}
        
        # 保存节点属性
        cal_node = 0
        node_attr = {}
        for node,attr in graph.nodes(data=True):
            node_name[cal_node] = node
            node_attr[cal_node] = attr
            cal_node += 1
        data.node_name = node_name
        data.node_attributes = node_attr

        # 保存边属性
        cal_edge = 0
        edge_relation = {}
        for edge1,edge2,attr in graph.edges(data=True):
            edge_relation[(list(node_name.keys())[list(node_name.values()).index(edge1)],list(node_name.keys())[list(node_name.values()).index(edge2)])] = attr
        data.edge_attributes = edge_relation

        # node_name = {}
        # colors = {}
        # raw_text = {}

        # cal = 0
        # for n in graph.nodes():
        #     node_name[cal] = n
        #     colors[cal] = graph.nodes[n]['color']
        #     raw_text[cal] = graph.nodes[n]['raw_text']
        #     cal = cal + 1

        # data.node_name = node_name
        # data.colors= colors
        # data.raw_text = raw_text

        
        # x = torch.ones((graph.number_of_nodes(),1), dtype=torch.float)

        # # 获取图G邻接矩阵的稀疏表示
        # adj = nx.to_scipy_sparse_array(graph).tocoo()

        # # 获取非零元素行索引
        # row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        # # 获取非零元素列索引
        # col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)

        # # 将行和列进行拼接，shape变为[2, num_edges], 包含两个列表，第一个是row, 第二个是col
        # edge_index = torch.stack([row, col], dim=0)

        # data = Data(x=x, edge_index=edge_index)
        # data.raw_text = list(graph.nodes)
        # # 根据data.raw_text创建字典，以自然数为键
        # data.raw_text_dict = {i:data.raw_text[i] for i in range(len(data.raw_text))}
        return data
    
    def from_pyg_2_networkx(self, graph=None):
        """
        从pyg转换为networkx
        """
        if graph is None:
            graph = self.graph
        networkx_graph = nx.MultiDiGraph()
        # 使用 add_nodes_from 批处理的效率比 add_node 高
        networkx_graph.add_nodes_from([i for i in range(graph.x.shape[0])])

        # 使用 add_edges_from 批处理的效率比 add_edge 高
        edges = np.array(graph.edge_index.T, dtype=int)
        networkx_graph.add_edges_from(edges)
        ## 遍历pyg_data中的node_attrs，给每一个节点添加对应的属性
        for node_index, attr_dict in graph.node_attributes.items():
            for key,value in attr_dict.items():
                networkx_graph.nodes()[node_index][key] = value

        # 根据pyg_data.edge_attributes中的信息，给每一个边线添加对应的属性
        for edge_index, attr_dict in graph.edge_attributes.items():
            for key,value in attr_dict.items():
                networkx_graph[edge_index[0]][edge_index[1]][0][key] = value

        networkx_graph = nx.relabel_nodes(networkx_graph, graph.node_name)

        # data = nx.Graph()
        # # 使用 add_nodes_from 批处理的效率比 add_node 高
        # data.add_nodes_from([i for i in range(graph.x.shape[0])])
        # # 使用 add_edges_from 批处理的效率比 add_edge 高
        # edges = np.array(graph.edge_index.T, dtype=int)
        # data.add_edges_from(edges)
        # ## 根据data中的color属性，给图中的节点添加这个属性
        # nx.set_node_attributes(data, graph.colors, 'color')
        # ## 根据data中的raw_text属性，给图中的节点添加这个属性
        # nx.set_node_attributes(data, graph.raw_text, 'raw_text')
        # ## 根据data中的node_name属性，将节点重新命名
        # data = nx.relabel_nodes(data, graph.node_name)

       

        return networkx_graph
    
    def draw_graph(self, 
                    graph,
                    print_node_attr = False,
                    node_attr = None,
                    print_edge_attr = False,
                    edge_attr = None,
                    colors = None,
                   ):
        """
        画出图
        """
        if graph is None:
            graph = self.graph
        plt.rcParams['font.sans-serif'] = ['Simhei']  #显示中文
        plt.rcParams['axes.unicode_minus'] = False    #显示负号
        pos = nx.spring_layout(graph)
        
        colors = []
        for n in graph.nodes():
            try:colors.append(graph.nodes[n]['color'])
            except:
                colors.append('brown')

        plt.figure(figsize=(12, 8))
        nx.draw(graph, pos=pos, 
                with_labels=True, 
                node_color=colors
                )

        #raw_text = nx.get_node_attributes(graph,'raw_text')
        
        if print_node_attr:
            node_labels = {}
            # 对与subgraph中的节点，检查是否存在node_attr属性，如果存在，则添加入node_label中，如果不存在，用'-'替代
            for node,attr in graph.nodes(data=True):
                if node_attr in attr:
                    node_labels[node] = attr[node_attr]
                # else:
                #     node_labels[node] = '-'
            nx.draw_networkx_labels(graph, pos, labels=node_labels,font_size=10,verticalalignment='bottom')

        
        if print_edge_attr:
            edge_labels = {}
            # 对与subgraph中的边，检查是否存在node_attr属性，如果存在，则添加入edge_labels中，如果不存在，用'-'替代
            for edge,attr in graph.edges(data=True):
                if edge_attr in attr:
                    edge_labels[(edge[0],edge[1])] = attr[edge_attr]
                # else:
                #     edge_labels[(edge[0],edge[1])] = '-'
            nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels)
        
        plt.show()