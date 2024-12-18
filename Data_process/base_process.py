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
        # 确保从 node_name 获取节点的原始编号
        node_ids = [graph.node_name[node_index] for node_index in range(graph.x.shape[0])]
        networkx_graph.add_nodes_from(node_ids)

        # 使用 add_edges_from 批处理的效率比 add_edge 高
        edges = np.array(graph.edge_index.T, dtype=int)
        # 使用 node_name 进行边的连接
        edges = [(graph.node_name[edge[0]], graph.node_name[edge[1]]) for edge in edges]
        networkx_graph.add_edges_from(edges)

        # 遍历pyg_data中的node_attrs，给每一个节点添加对应的属性
        for node_index, attr_dict in graph.node_attributes.items():
            node = graph.node_name[node_index]  # 获取原始节点
            for key, value in attr_dict.items():
                networkx_graph.nodes[node][key] = value

        # 根据pyg_data.edge_attributes中的信息，给每一条边添加对应的属性
        for edge_index, attr_dict in graph.edge_attributes.items():
            node1, node2 = edge_index
            networkx_graph[node1][node2][0].update(attr_dict)

        return networkx_graph
    

        #修改后，全局编码node和edge
    def from_networkx_2_pyg_global(self, graph=None, cal_node=0, cal_edge=0, global_node_map=None):
        """
        从networkx转换为pytorch_geometric，支持遍历所有子图并更新邻接矩阵的全局编号
        """
        if graph is None:
            graph = self.graph

        # 创建节点特征矩阵
        x = torch.ones((graph.number_of_nodes(), 1), dtype=torch.float)

        # 获取图G邻接矩阵的稀疏表示
        adj = nx.to_scipy_sparse_array(graph).tocoo()

        # 获取非零元素行索引
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        # 获取非零元素列索引
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)


       # 保存节点属性
        node_name = {}
        node_attr = {}
        edge_attr = {}
        # 如果global_node_map是None，初始化为空字典
        if global_node_map is None:
            global_node_map = {}

        # 为当前子图分配全局唯一编号，并更新global_node_map
        local_node_map = {}  # 当前子图的节点映射
        for node, attr in graph.nodes(data=True):
            local_node_map[node] = cal_node  # 将当前子图节点映射到全局节点编号
            global_node_map[cal_node] = node  # 记录全局编号对应的节点
            node_name[cal_node] = node  # 使用全局唯一编号作为键
            node_attr[cal_node] = attr
            cal_node += 1  # 递增全局计数器
        # 创建全局的edge_index
        global_row = []
        global_col = []
        for local_row, local_col in zip(row, col):
            # 在这里，local_row和local_col是稀疏矩阵中的索引，应该直接使用它们作为索引来访问图的节点
            node1 = list(graph.nodes)[local_row]  # 获取全局节点编号
            node2 = list(graph.nodes)[local_col]  # 获取全局节点编号
            
            # 使用local_node_map将本地节点映射到全局编号
            global_row.append(local_node_map[node1])
            global_col.append(local_node_map[node2])

        # 将行和列进行拼接，shape变为[2, num_edges], 包含两个列表，第一个是row, 第二个是col
        edge_index = torch.stack([torch.tensor(global_row), torch.tensor(global_col)], dim=0)
        #print('edge_index:',edge_index)


        # 创建Data对象
        data = Data(x=x, edge_index=edge_index)

        data.node_name = node_name
        data.node_attributes = node_attr
        #print('node_name:',node_name)

        # 保存边属性
        for edge1, edge2, attr in graph.edges(data=True):
            node1_idx = local_node_map[edge1]
            node2_idx = local_node_map[edge2]
            edge_attr[(node1_idx, node2_idx)] = attr
            cal_edge += 1  # 递增全局边计数器

        data.edge_attributes = edge_attr

        # 返回更新后的数据以及新的全局计数器状态和global_node_map
        return data, cal_node, cal_edge, global_node_map

    def from_pyg_2_networkx_global(self, graph=None):
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
        return networkx_graph


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