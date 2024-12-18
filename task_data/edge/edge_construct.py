import torch
import numpy as np
import torch_geometric as pyg
from abc import ABC, abstractmethod
from util.utils import set_mask,scipy_rwpe
from task_data.base_construct import DatasetWithCollate
from typing import Optional, Callable, Any, Tuple, Union
from scipy.sparse import csr_array

def process_binary_label(embs, label):
    binary_rep = torch.zeros((1, len(embs)))  # shape: [1, num_classes]
    binary_rep[0, label] = 1  # [0,...,1...,0] 直接使用 label
    embs = embs[[0, 1]]
    return torch.tensor(label).view(1, -1).to(torch.long), embs, binary_rep  # 将 label 转换为 tensor 并调整形状


def sample_fixed_hop_size_neighbor(adj_mat: object, root: object, hop: object, max_nodes_per_hop: object = 10) -> object:
    visited = np.array(root)
    fringe = np.array(root)
    nodes = np.array([])
    for h in range(1, hop + 1):
        u = adj_mat[fringe].nonzero()[1]
        fringe = np.setdiff1d(u, visited)
        visited = np.union1d(visited, fringe)
        if len(fringe) > max_nodes_per_hop:
            fringe = np.random.choice(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = np.concatenate([nodes, fringe])
        # dist_list+=[dist+1]*len(fringe)
    nodes = nodes.astype(int)
    return nodes


class OFA_collater:
    """
    Collater is used for merge a batch of OFA data. It supports two modes:
    1. If llm_tokenzier is None, collater assumes edge and node features are fixed size numpy array and convert it to torch tensor.
    2. If llm_tokenzier is not None, collater assumes edge and node features are raw texts and use tokenzier to convert it to text ids.
    All other attributes will be merged by default PyG collater.
    """
    def __init__(self, llm_tokenizer, llm_max_length):
        self.llm_tokenizer = llm_tokenizer
        self.llm_max_length = llm_max_length
        self.pyg_collater = pyg.loader.dataloader.Collater(None, None)

    def return_unique_text_mapping(self, texts):
        """
        return unique text list with a mapping back to original list.
        """
        sorted_position = np.argsort(texts)
        sorted_texts = texts[sorted_position]
        keys = np.unique(sorted_texts)
        lower = np.searchsorted(sorted_texts, keys)
        higher = np.append(lower[1:], len(sorted_texts))
        unique_texts = []
        mappings = np.zeros(len(texts)).astype(int)
        for i, (key, lower_i, higher_i) in enumerate(zip(keys, lower, higher)):
            unique_texts.append(key)
            mappings[sorted_position[lower_i: higher_i]] = i

        return np.array(unique_texts), mappings

    def tokenize(self, text_inputs):
        text_tokens = self.llm_tokenizer(text_inputs,
                                         return_tensors="pt",
                                         padding="longest",
                                         truncation=True,
                                         max_length=self.llm_max_length)
        return text_tokens
    def __call__(self, batch):
        g = self.pyg_collater(batch)
        if self.llm_tokenizer is None:
            g.x = torch.from_numpy(np.concatenate(g.x, axis=0))
            g.edge_attr = torch.from_numpy(np.concatenate(g.edge_attr, axis=0))
        else:
            text_inputs = np.concatenate(g.x + g.edge_attr, axis=0)
            unique_text_inputs, text_mapping = self.return_unique_text_mapping(text_inputs)
            text_tokens = self.tokenize(unique_text_inputs.tolist())
            g.text_tokens = text_tokens
            g.text_mapping = torch.from_numpy(text_mapping)
        return g


class GraphTextDataset(DatasetWithCollate, ABC):
    """
    Base class for all OFA runtime datasets, responsible for loading graphs from OFAPygDataset, subgraphing,
    and prompt graph construction.
    """

    def __init__(self, graph: Union[pyg.data.Data, list[pyg.data.Data]], process_label_func: Callable, **kwargs):
        """
        Args:
            graph: Main graph objects, one single graph for single graph dataset, and list of graphs
                        for list of graphs.
            process_label_func: a Callable function that process the labels from original datasets to accommodate
                                    different tasks.
            **kwargs: additional arguments.
        """
        self.prompt_edge_emb = None
        self.g = graph
        self.process_label_func = process_label_func
        self.kwargs = kwargs
        self.llm_tokenizer = None
        self.llm_max_length = None
        #self.edge_mode = 1
        if "prompt_edge_list" in kwargs:
            self.prompt_edge_list = kwargs["prompt_edge_list"]
        else:
            self.prompt_edge_list = {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0], "c2n": [4, 0]},
        if "no_class_node" in kwargs and kwargs["no_class_node"]:
            self.no_class_node = True
        else:
            self.no_class_node = False

    def __getitem__(self, index):
        feature_graph = self.make_feature_graph(index)
        prompt_graph = self.make_prompted_graph(feature_graph)
        ret_data = self.to_pyg(feature_graph, prompt_graph)
        if "walk_length" in self.kwargs and self.kwargs["walk_length"] is not None:
            ret_data.rwpe = scipy_rwpe(ret_data, self.kwargs["walk_length"])
        return ret_data

    @abstractmethod
    def make_feature_graph(self, index) -> list:
        """
        Create feature subgraph based on index

        Args:
            index: int

        Returns:
            feat: torch.Tensor, node vector representations
            edge_feat: torch.Tensor, edge vector representations
            edge_index: torch.Tensor, feature edge indices
            e_type: torch.Tensor, feature edge types most likely 0-vector
            target_node_id: torch.Tensor, the indices of NOI
            class_emb: class node vector representations
            binary_rep: one-/multi-hot label representations
        """
        pass

    @abstractmethod
    def make_prompt_node(self, feat, class_emb):
        """
        Create prompt node features
        Args:
            feat: feature graph node features
            class_emb: class node features

        Returns:
            prompt_graph_node_features: prompt graph node features
        """
        pass

    def make_prompted_graph(self, feature_graph):
        """
        Create prompted graph based on feature graphs, prompt edge construction is based on self.prompt_edge_list.
        self.prompt_edge_list defines connection types, edge_type indices, and indices in to self.prompt_edge_emb.
        Refer to data.ofa_data.OFAPygDataset.get_edge_list for details.
        Args:
            feature_graph: output from self.make_feature_graph

        Returns:

        """
        (feat, edge_feat, edge_index, e_type, target_node_id, class_emb, label, binary_rep,) = feature_graph
        n_feat_node = len(feat)
        feat = self.make_prompt_node(feat, class_emb)
        prompt_edge_lst = []
        prompt_edge_type_lst = []
        prompt_edge_feat_lst = []
        for prompt_edge_str in self.prompt_edge_list:
            prompt_e_index = getattr(self, "make_" + prompt_edge_str + "_edge")(target_node_id, class_emb, n_feat_node)
            prompt_edge_types = torch.zeros(len(prompt_e_index[0]), dtype=torch.long) + \
                                self.prompt_edge_list[prompt_edge_str][0]

            if self.prompt_edge_list[prompt_edge_str][1] is None:
                edge_emb = self.prompt_edge_emb
            else:
                edge_emb = self.prompt_edge_emb[self.prompt_edge_list[prompt_edge_str][1]]
            # If the number of edge emb is 1, repeat it for each prompt edge.
            # If not, assume the number of edge emb equal to the number of prompt edge.
            # Currently, only n2f and f2n in KG dataset will have number of edge emb larger than 1.
            num_edge_emb = len(self.prompt_edge_list[prompt_edge_str][1])
            assert num_edge_emb == 1 or num_edge_emb == len(prompt_e_index[0])
            if num_edge_emb > 1:
                prompt_edge_feat = edge_emb
            else:
                prompt_edge_feat = edge_emb.repeat(len(prompt_e_index[0]), axis=0)
            prompt_edge_lst.append(prompt_e_index)
            prompt_edge_type_lst.append(prompt_edge_types)
            prompt_edge_feat_lst.append(prompt_edge_feat)
        edge_index = torch.cat([edge_index] + prompt_edge_lst, dim=-1, )
        e_type = torch.cat([e_type] + prompt_edge_type_lst)
        edge_feat = np.concatenate([edge_feat] + prompt_edge_feat_lst, axis=0)
        return feat, edge_index, label, edge_feat, e_type

    def to_pyg(self, feature_graph, prompted_graph):
        feat, edge_index, label, edge_feat, e_type = prompted_graph
        new_subg = pyg.data.Data(feat, edge_index, y=label, edge_attr=edge_feat, edge_type=e_type)
        num_class = len(feature_graph[-3])
        bin_labels = torch.zeros(new_subg.num_nodes, dtype=torch.float)
        bin_labels[new_subg.num_nodes - num_class:] = feature_graph[-1]
        new_subg.bin_labels = bin_labels
        set_mask(new_subg, "true_nodes_mask", list(range(new_subg.num_nodes - num_class, new_subg.num_nodes)))
        set_mask(new_subg, "noi_node_mask", new_subg.num_nodes - num_class - 1)
        set_mask(new_subg, "target_node_mask", feature_graph[-4])
        set_mask(new_subg, "feat_node_mask", list(range(len(feature_graph[0]))))
        new_subg.sample_num_nodes = new_subg.num_nodes
        new_subg.num_classes = num_class
        return new_subg

    def add_llm_tokenizer(self, tokenizer, llm_max_length):
        """
        add llm tokenizer for collater.
        """
        self.llm_tokenizer = tokenizer
        self.llm_max_length = llm_max_length

    def get_collate_fn(self):
        return OFA_collater(self.llm_tokenizer, self.llm_max_length)

    def process_label(self, label):
        """
        Process labels into one-/multi-hot format using self.process_label_func
        """
        if self.process_label_func is None:
            trimed_class = torch.zeros((1, len(self.class_emb)))
            trimed_class[0, label] = 1
            return label, self.class_emb, trimed_class
        else:
            return self.process_label_func(self.class_emb, label)

class SubgraphDataset(GraphTextDataset):
    """
    Build feature subgraphs from a large graph, used mostly in node/link tasks
    """

    def __init__(self, pyg_graph, class_emb, prompt_edge_emb, data_idx, hop=2, max_nodes_per_hop=10, class_mapping=None, to_undirected=False,
                 process_label_func=None, adj=None, **kwargs, ):
        super().__init__(pyg_graph, process_label_func, **kwargs)

        self.max_nodes_per_hop = max_nodes_per_hop
        self.to_undirected = to_undirected
        edge_index = self.g.edge_index
        if self.to_undirected:
            edge_index = pyg.utils.to_undirected(edge_index)
        if adj is not None:
            self.adj = adj
        else:
            self.adj = csr_array((torch.ones(len(edge_index[0])), (edge_index[0], edge_index[1]),),
                                 shape=(self.g.num_nodes, self.g.num_nodes), )
        self.class_emb = class_emb
        self.prompt_edge_emb = prompt_edge_emb
        self.hop = hop
        self.data_idx = data_idx
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.data_idx)

    def get_neighbors(self, index):
        node_id = self.data_idx[index]
        neighbors = sample_fixed_hop_size_neighbor(self.adj, [node_id], self.hop,
                                                   max_nodes_per_hop=self.max_nodes_per_hop)
        neighbors = np.r_[node_id, neighbors]
        edges = self.adj[neighbors, :][:, neighbors].tocoo()
        if self.class_mapping is not None:
            label = self.class_mapping[self.g.y[node_id]]
        else:
            label = self.g.y[node_id]
        edge_index = torch.stack(
            [torch.tensor(edges.row, dtype=torch.long), torch.tensor(edges.col, dtype=torch.long), ])
        label, emb, binary_rep = self.process_label(label)
        return edge_index, neighbors, emb, label, binary_rep, [0]

    def make_feature_graph(self, index):
        (edge_index, neighbors, emb, label, binary_rep, target_node_id,) = self.get_neighbors(index)
        #feat = self.g.node_text_feat[neighbors]
        feat=self.g.node_embs[neighbors]
        e_type = torch.zeros(len(edge_index[0]), dtype=torch.long)
        #edge_feat = self.g.edge_text_feat.repeat(len(edge_index[0]), axis=0)
        #edge_feat = self.g.edge_embs.repeat(len(edge_index[0]), axis=0)
        edge_feat=self.g.edge_embs[edge_index[0]]
        return (feat, edge_feat, edge_index, e_type, target_node_id, emb, label, binary_rep,)

    def make_prompt_node(self, feat, class_emb):
        # Only feature nodes and class nodes, no NOI node.
        if not self.no_class_node:
            feat = np.concatenate([feat, class_emb], axis=0)
        return feat

    def make_f2n_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor(
            [target_node_id * len(class_emb), [i + n_feat_node for i in range(len(class_emb))], ], dtype=torch.long, )
        return prompt_edge

    def make_n2f_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor([[i + n_feat_node for i in range(len(class_emb))], target_node_id * len(class_emb)],
                                   dtype=torch.long, )
        return prompt_edge


class SubgraphHierDataset(SubgraphDataset):
    def __init__(self, pyg_graph, class_emb, prompt_edge_emb, noi_node_emb, data_idx, hop=2, max_nodes_per_hop=10,
                 class_mapping=None, to_undirected=False, process_label_func=None, adj=None, **kwargs, ):
        super().__init__(pyg_graph, class_emb, prompt_edge_emb, data_idx, hop, max_nodes_per_hop, class_mapping,
                         to_undirected, process_label_func, adj, **kwargs, )
        self.noi_node_emb = noi_node_emb

    def __len__(self):
        return len(self.data_idx)

    def make_prompt_node(self, feat, class_emb):
        # Add class node in zero-shot scenario. In few-shot scenario, only NOI node. Class nodes will be added by
        # future dataset wrapper
        if self.no_class_node:
            feat = np.concatenate([feat, self.noi_node_emb], axis=0)
        else:
            feat = np.concatenate([feat, self.noi_node_emb, class_emb], axis=0)

        return feat

    def make_f2n_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor([target_node_id, [n_feat_node] * len(target_node_id)], dtype=torch.long, )
        return prompt_edge

    def make_n2f_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor([[n_feat_node] * len(target_node_id), target_node_id], dtype=torch.long, )
        return prompt_edge

    def make_n2c_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor(
            [[n_feat_node] * len(class_emb), [i + n_feat_node + 1 for i in range(len(class_emb))], ],
            dtype=torch.long, )
        return prompt_edge

    def make_c2n_edge(self, target_node_id, class_emb, n_feat_node):
        prompt_edge = torch.tensor(
            [[i + n_feat_node + 1 for i in range(len(class_emb))], [n_feat_node] * len(class_emb)], dtype=torch.long, )
        return prompt_edge


class SubgraphLinkHierDataset(SubgraphHierDataset):
    def __init__(self, pyg_graph, class_emb, prompt_edge_emb, noi_node_emb, edges, remove_edge=False, hop=2,
                 max_nodes_per_hop=100, class_mapping=None, to_undirected=False, process_label_func=None, adj=None, **kwargs, ):
        super().__init__(pyg_graph, class_emb, prompt_edge_emb, noi_node_emb, None, hop, max_nodes_per_hop, class_mapping,
                         to_undirected, process_label_func, adj, **kwargs, )
        self.edges = edges
        self.pos_index = len(self.edges)
        self.remove_edge = remove_edge

        # Sample negative edges for training and testing
        dense_adj = self.adj.todense() == 0
        neg_row, neg_col = np.nonzero(dense_adj)
        neg_edge_idx = np.random.permutation(len(neg_row))[: self.pos_index]
        neg_row, neg_col = neg_row[neg_edge_idx], neg_col[neg_edge_idx]
        self.neg_edges = np.stack([neg_row, neg_col], axis=1)
        self.total_edges = np.concatenate([self.edges, self.neg_edges], axis=0)

    def __len__(self):
        return len(self.total_edges)

    def remove_link(self, row, col):
        remove_ind = np.logical_or(np.logical_and(row == 0, col == 1), np.logical_and(row == 1, col == 0), )
        keep_ind = np.logical_not(remove_ind)
        return row[keep_ind], col[keep_ind]

    def get_neighbors(self, index):
        edge_id = self.total_edges[index]

        if index < self.pos_index:
            label = 1
        else:
            label = 0
        node_ids = list(edge_id)
        neighbors = sample_fixed_hop_size_neighbor(self.adj, node_ids, self.hop,
                                                   max_nodes_per_hop=self.max_nodes_per_hop)
        neighbors = np.r_[node_ids, neighbors]
        edges = self.adj[neighbors, :][:, neighbors].tocoo()
        row = edges.row
        col = edges.col

        # Remove target edge from train graphs
        if self.remove_edge and index < self.pos_index:
            row, col = self.remove_link(row, col)
        edge_index = torch.stack([torch.tensor(row, dtype=torch.long), torch.tensor(col, dtype=torch.long), ])
        label, embs, binary_rep = self.process_label(label)
        return edge_index, neighbors, embs, label, binary_rep, [0, 1]


#Edge task constructor
# def ConstructLinkCls(dataset, split, split_name, prompt_feats, to_bin_cls_func, global_data, task_level, **kwargs):
#     text_g = dataset.data
#     edges = text_g.edge_index
#     train_graph = global_data
#     return SubgraphLinkHierDataset(train_graph, prompt_feats["class_node_text_feat"],
#                                    prompt_feats["prompt_edge_text_feat"], prompt_feats["noi_node_text_feat"],
#                                    edges.T[split[split_name]].numpy(), to_undirected=True, hop=3,
#                                    process_label_func=globals()[to_bin_cls_func], prompt_edge_list=dataset.get_edge_list(task_level),
#                                    **kwargs, )



#Edge task constructor
def ConstructLinkCls(dataset, split, split_name, prompt_feats, to_bin_cls_func, global_data, task_level, **kwargs):
    text_g = dataset.data
    edges = text_g.edge_index

        # 找到所有包含 'strength' 的节点
    strength_nodes = [
        int(node_idx)  # 确保是整数
        for inner_list in text_g.node_name  # 遍历外层列表
        for node_dict in inner_list         # 遍历内层的字典列表
        if isinstance(node_dict, dict)      # 确保元素是字典
        for node_idx, node_name in node_dict.items()  # 遍历字典中的键值对
        if 'strength' in node_name          # 筛选名称包含 'strength' 的节点
    ]
    strength_nodes = torch.tensor(strength_nodes, dtype=torch.long)  # 转为 tensor
    print('Strength nodes:', strength_nodes)

    # 筛选出所有只包含 strength 节点的边
    mask = (torch.isin(edges[0], strength_nodes) & torch.isin(edges[1], strength_nodes))
    strength_edges = edges[:, mask]

    # 获取指定划分的边
    selected_edges = strength_edges[:, split[split_name]]
    # 将strength_edges转换为edge_index形式
    strength_edge_index = torch.tensor(selected_edges).T
    print('Strength edge index:', strength_edge_index.shape)

    # 获取训练图
    train_graph = global_data

    # 返回子图数据集
    return SubgraphLinkHierDataset(
        train_graph,
        prompt_feats["class_node_text_feat"],
        prompt_feats["prompt_edge_text_feat"],
        prompt_feats["noi_node_text_feat"],
        strength_edge_index.numpy(),
        to_undirected=True,
        hop=2,
        process_label_func=globals()[to_bin_cls_func],
        prompt_edge_list=dataset.get_edge_list(task_level),
        **kwargs,
    )