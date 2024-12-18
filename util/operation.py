import torch
import random
import torch_geometric as pyg
def Graph_splitter(dataset):
    return dataset.get_idx_split()

def LinkSplitter(dataset):
    text_g = dataset.data
    print('text_G.node_name', text_g.node_name)
    edges = text_g.edge_index  # (2, n) 的边数组
    print('edges_shape:', edges.shape)

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
    strength_edge_index = edges[:, mask]
    print('strength_edges:', strength_edge_index.shape)
    # 随机打乱边
    edge_perm = torch.randperm(len(strength_edge_index [0]))
    train_offset = int(len(edge_perm) * 0.70)
    val_offset = int(len(edge_perm) * 0.85)
    edge_indices = {"train": edge_perm[:train_offset], "valid": edge_perm[train_offset:val_offset],
                    "test": edge_perm[val_offset:], }
    
    return edge_indices

def LinkConstructGraph(dataset, split):
    text_g = dataset.data
    edges = text_g.edge_index  # (2, n) 的边数组

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
    strength_edge_index = edges[:, mask]

    # 只使用训练集的边
    train_edge_index = strength_edge_index[:, split["train"]]

    # 构建图的字典
    graph_dict = text_g.to_dict()
    graph_dict["edge_index"] = train_edge_index
    
    # 构建 PyG 图
    train_graph = pyg.data.Data(**graph_dict)
    
    return train_graph



def Node_Splitter(dataset, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    split = {"train": [], "valid": [], "test": []}

    # 遍历 dataset 中的每个图（每个 Data 对象）
    for data in dataset:
        strength_nodes = []  # 用于存储当前图中的强度节点的索引
        for i in data.node_attributes:
            for node_name, attr in i.items():
                if attr.get('raw_text') and 'strength' in attr['raw_text']:
                    strength_nodes.append(node_name)

        # 划分训练集、验证集和测试集
        random.shuffle(strength_nodes)  # 随机打乱顺序
        num_nodes = len(strength_nodes)
        
        train_size = int(num_nodes * train_ratio)
        valid_size = int(num_nodes * valid_ratio)
        test_size = num_nodes - train_size - valid_size
        
        # 按照比例划分
        split["train"].extend(strength_nodes[:train_size])
        split["valid"].extend(strength_nodes[train_size:train_size + valid_size])
        split["test"].extend(strength_nodes[train_size + valid_size:])
    
    # 转换为 Tensor 格式
    split["train"] = torch.tensor(split["train"])
    split["valid"] = torch.tensor(split["valid"])
    split["test"] = torch.tensor(split["test"])
    
    return split

def Node_Splitter_FS(dataset, k_shot=5, valid_ratio=0.5, test_ratio=0.5):
    """
    Splits the dataset nodes into train, valid, and test sets by selecting k-shot samples per label from all strength nodes across all graphs.

    Parameters:
        dataset (list): A list of Data objects, where each contains graph node attributes.
        k_shot (int): Number of nodes to select for the training set per label across all graphs.
        valid_ratio (float): Proportion of remaining nodes for validation.
        test_ratio (float): Proportion of remaining nodes for testing.

    Returns:
        dict: A dictionary with keys 'train', 'valid', and 'test', each containing tensor indices.
    """
    assert valid_ratio + test_ratio == 1.0, "Validation and test ratios must sum to 1."

    split = {"train": [], "valid": [], "test": []}
    label_to_nodes = {}  # Map each label to its corresponding strength nodes across all graphs

    # Iterate through each graph in the dataset
    for data in dataset:
        # Identify strength nodes
        for i in data.node_attributes:
            for node_name, attr in i.items():
                if attr.get('raw_text') and 'strength' in attr['raw_text']:
                    if 'label' in attr:  # Assuming 'label' is the key for the node's label
                        label = attr['label']
                        if label not in label_to_nodes:
                            label_to_nodes[label] = []
                        label_to_nodes[label].append(node_name)

    # Shuffle nodes within each label group
    for label, nodes in label_to_nodes.items():
        random.shuffle(nodes)

    train_nodes = []
    valid_nodes = []
    test_nodes = []

    # Select k-shot nodes per label for training and split the rest
    for label, nodes in label_to_nodes.items():
        if k_shot is not None:
            k_shot = min(k_shot, len(nodes))
            train_nodes.extend(nodes[:k_shot])
            remaining_nodes = nodes[k_shot:]
        else:
            remaining_nodes = nodes

        # Split remaining nodes into validation and test sets
        num_remaining = len(remaining_nodes)
        valid_size = int(num_remaining * valid_ratio)

        valid_nodes.extend(remaining_nodes[:valid_size])
        test_nodes.extend(remaining_nodes[valid_size:])

    # Add nodes to the split dictionary
    split["train"].extend(train_nodes)
    split["valid"].extend(valid_nodes)
    split["test"].extend(test_nodes)
    print("Node split:", split)
    # Convert lists to tensors
    split["train"] = torch.tensor(split["train"], dtype=torch.long)
    split["valid"] = torch.tensor(split["valid"], dtype=torch.long)
    split["test"] = torch.tensor(split["test"], dtype=torch.long)

    return split

# 划分train,valid,test
def get_split_key(dataset_config):
        return dataset_config["dataset_name"] + "-" + dataset_config["task_level"] + "-" + dataset_config["task"]


def get_data_split(
                   # self, 
                   dataset,
                   dataset_split,
                   dataset_config = None,
                   ):
    """
    Split data based on task_level
    """
    split_key = get_split_key(dataset_config)

    if split_key not in dataset_split:# 如果已经存在，则不需要再进行处理
        dataset_splitter = dataset_config.get("dataset_splitter") # 获取键值，例如'CiteSplitter'
        split = globals()[dataset_splitter](
                dataset[dataset_config["dataset_name"]]) if dataset_splitter else None  # 从self.dataset中把之前读取的数据丢进去
        # 在全局中查询名为dataset_splitter的函数并调用，如果没有，则返回None
        dataset_split[split_key] = split

    return dataset_split,split_key

def get_stage_name(stage_config, dataset_config):
    return "-".join([stage_config["dataset"], get_split_key(dataset_config), stage_config["stage"],
                        stage_config["split_name"]])


def get_global_data(datset,dataset_split,preprocess_storage,dataset_config):
    """
    If global_data for a dataset is required, such as constructed train graph for link tasks, a preprocessing
    function is called and the returned values are stored.
    """
    split_key = get_split_key(dataset_config)
    if split_key not in preprocess_storage:
        preprocessor = dataset_config.get("preprocess")
        global_data = globals()[preprocessor](datset[dataset_config["dataset_name"]],
                                                 dataset_split[split_key]) if preprocessor else None
        # global_data = globals()[preprocessor](datset,
        #                                         dataset_split[split_key]) if preprocessor else None
        preprocess_storage[split_key] = global_data
    return preprocess_storage,split_key

def get_construct_func(task_config):
    return task_config.get("construct")  # 获取键值，例如'LinkConstruct'