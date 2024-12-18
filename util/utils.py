import gc
import os
import yaml
import time
import numpy as np
import torch
import random
import torch.nn.functional as F
from torch_geometric.utils import (to_scipy_sparse_matrix, scatter, )
from torchmetrics import AveragePrecision, AUROC
from tqdm.autonotebook import trange
from task_data.base_construct import DatasetWithCollate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def evenly_divide_list(lst, num_groups):
    # 计算每组应该包含的元素数量和余数
    n = len(lst)
    q, r = divmod(n, num_groups)
    group_sizes = [q] * num_groups
    for i in range(r):
        group_sizes[i] += 1

    # 根据每组应该包含的元素数量和余数分配元素
    groups = []
    i = 0
    for size in group_sizes:
        group = lst[i:i+size]
        groups.append(group)
        i += size

    return groups

def load_yaml(dir):
    with open(dir, "r") as stream:
        return yaml.safe_load(stream)

def combine_dict(*args):
    combined_dict = {}
    for d in args:
        for k in d:
            combined_dict[k] = d[k]
    return combined_dict


def merge_mod(params, mod_args):
    for i in range(0, len(mod_args), 2):
        if mod_args[i + 1].isdigit():
            val = int(mod_args[i + 1])
        elif mod_args[i + 1].replace(".", "", 1).isdigit():
            val = float(mod_args[i + 1])
        elif mod_args[i + 1].lower() == "true":
            val = True
        elif mod_args[i + 1].lower() == "false":
            val = False
        else:
            val = mod_args[i + 1]
        params[mod_args[i]] = val
    return params

def set_random_seed(seed):
    """Set python, numpy, pytorch global random seed.
    Does not guarantee determinism due to PyTorch's feature.

    Arguments:
        seed {int} -- Random seed to set
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def scipy_rwpe(data, walk_length):
    row, col = data.edge_index
    N = data.num_nodes

    value = data.edge_weight
    if value is None:
        value = torch.ones(data.num_edges, device=row.device)
    value = scatter(value, row, dim_size=N, reduce="sum").clamp(min=1)[row]
    value = 1.0 / value
    adj = to_scipy_sparse_matrix(data.edge_index, edge_attr=value, num_nodes=data.num_nodes)

    out = adj
    pe_list = [out.diagonal()]
    for _ in range(walk_length - 1):
        out = out @ adj
        pe_list.append(out.diagonal())
    pe = torch.tensor(np.stack(pe_list, axis=-1))

    return pe


def get_available_devices():
    r"""Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids

def compare_node(graph1,graph2):
    for node in graph1.nodes(data =True):
        if node[0] not in graph2.nodes:
            print(node)
        for key,value in node[1].items():
            if graph2.nodes()[node[0]][key] != value:
                print(node[0],key)
    
    # 反过来比较
    for node in graph2.nodes(data = True):
        if node[0] not in graph1.nodes:
            print(node)
        for key,value in node[1].items():
            if graph1.nodes()[node[0]][key] != value:
                print(node[0],key)

def compare_edge(graph1,graph2):
    for edge in graph1.edges(data =True):
        if edge[:2] not in graph2.edges():
            print(edge[:2])
        for key,value in edge[2].items():
            if graph2[edge[0]][edge[1]][0][key] != value:
                print(edge[0],edge[1],key)
    
    # 反过来比较
    for edge in graph2.edges(data = True):
        if edge[:2] not in graph1.edges():
            print(edge[:2])
        for key,value in edge[2].items():
            if graph1[edge[0]][edge[1]][0][key] != value:
                print(edge[0],edge[1],key)


def get_label_texts(labels):
    label_texts = [None] * int(len(labels) * 2)
    for entry in labels:
        label_texts[labels[entry][0]] = (
                "prompt node. molecule property description. " + "The molecule is effective to the following assay. " +
                labels[entry][1][0][:-41])
        label_texts[labels[entry][0] + len(labels)] = (
                "prompt node. molecule property description. " + "The molecule is not effective to the following "
                                                                 "assay. " +
                labels[entry][1][0][:-41])
    return label_texts


def set_mask(data, name, index, dtype=torch.bool):
    mask = torch.zeros(data.num_nodes, dtype=dtype)
    mask[index] = True
    setattr(data, name, mask)

def dict_res_summary(res_col):
    """Combine multiple dictionary information into one dictionary
    so that all entries with the same key will be concatenated into
    a list

    Arguments:
        res_col {list[dictionary]} -- a list of dictionary

    Returns:
        dictionary -- summarized dictionary information
    """
    res_dict = {}
    for res in res_col:
        for k in res:
            if k not in res_dict:
                res_dict[k] = []
            res_dict[k].append(res[k])
    return res_dict

def load_pretrained_state(model_dir, deepspeed=False):
    if deepspeed:
        def _remove_prefix(key: str, prefix: str) -> str:
            return key[len(prefix):] if key.startswith(prefix) else key
        #state_dict = get_fp32_state_dict_from_zero_checkpoint(model_dir)
        state_dict = {_remove_prefix(k, "_forward_module."): state_dict[k] for k in state_dict}
    else:
        state_dict = torch.load(model_dir)["state_dict"]
    return state_dict

class SmartTimer:
    """A timer utility that output the elapsed time between this
    call and last call.
    """

    def __init__(self, verb=True) -> None:
        """SmartTimer Constructor

        Keyword Arguments:
            verb {bool} -- Controls printing of the timer (default: {True})
        """
        self.last = time.time()
        self.verb = verb

    def record(self):
        """Record current timestamp"""
        self.last = time.time()

    def cal_and_update(self, name):
        """Record current timestamp and print out time elapsed from last
        recorded time.

        Arguments:
            name {string} -- identifier of the printout.
        """
        now = time.time()
        if self.verb:
            print(name, now - self.last)
        self.record()


def make_mask_proportion(data,proportion=[6,2,2],seed=0):
    label_list = list(set([data[i]['label'] for i in data]))
    class_list = {i:[] for i in label_list}
    for i in data:
        label = data[i]['label']
        class_list[label].append(i)
    for i in class_list:
        np.random.seed(seed)
        np.random.shuffle(class_list[i])
        length = len(class_list[i])
        train = int(length*proportion[0]/sum(proportion))
        val = int(length*proportion[1]/sum(proportion))
        test = length - train - val
        class_list[i] = {'train':class_list[i][:train],'val':class_list[i][train:train+val],'test':class_list[i][train+val:]}
        # 根据class_list的索引，给data中的数据添加mask属性
        for j in class_list[i]['train']:
            data[j]['mask'] = 'train'
        for j in class_list[i]['val']:
            data[j]['mask'] = 'valid'
        for j in class_list[i]['test']:
            data[j]['mask'] = 'test'
    return data

#k-shot
def make_fs4_mask_proportion(data, k_shot=0, seed=0):
    """
    自定义划分函数：
    - 从 Non-halogen compounds 和 Halogen compounds 中各抽取 k_shot 个样本放到训练集中。
    - 剩余样本按 1:1 比例分配到验证集和测试集，验证集和测试集样本数固定为 10。

    参数:
    - data: 样本数据，假设每个样本包含 'label' 和其他属性。
    - k_shot: 每个类别放入训练集的样本数量。
    - seed: 随机种子，确保划分可复现。

    返回:
    - 修改后的数据，其中每个样本添加了 'mask' 属性，值为 'train'、'valid' 或 'test'。
    """

    val_test_labels = {'Non-halogen compounds', 'Halogen compounds'}

    # 按标签分组样本
    train_data = {'Non-halogen compounds': [], 'Halogen compounds': []}
    val_test_data = {'Non-halogen compounds': [], 'Halogen compounds': []}

    for i in data:
        label = data[i]['label']
        if label in val_test_labels:
            # 将标签为 'Non-halogen compounds' 或 'Halogen compounds' 的样本分配到验证/测试集
            val_test_data[label].append(i)

    # 设置随机种子
    np.random.seed(seed)

    # 从 Non-halogen compounds 和 Halogen compounds 中各抽取 k_shot 个样本到训练集中
    for label, samples in val_test_data.items():
        if samples:
            selected_samples = np.random.choice(samples, min(k_shot, len(samples)), replace=False)
            train_data[label] = selected_samples  # 更新训练数据
            for sample in selected_samples:
                samples.remove(sample)  # 从验证/测试集中移除已选样本

    # 给训练集样本分配 mask
    train_samples = []
    train_samples.extend(train_data['Non-halogen compounds'])
    train_samples.extend(train_data['Halogen compounds'])
    for idx in train_samples:
        data[idx]['mask'] = 'train'

    # 将剩余的验证和测试集样本合并，并按 1:1 比例划分
    remaining_samples = val_test_data['Non-halogen compounds'] + val_test_data['Halogen compounds']
    
    # 随机打乱剩余的样本
    np.random.shuffle(remaining_samples)
    
    # 确保验证集和测试集的样本数分别为 10
    val_size = 10
    test_size = 10
    
    # 划分验证集
    for idx in remaining_samples[:val_size]:
        data[idx]['mask'] = 'valid'
    
    # 划分测试集
    for idx in remaining_samples[val_size:val_size + test_size]:
        data[idx]['mask'] = 'test'

    return data

def make_fs3_mask_proportion(data, k_shot=0, seed=0):
    """
    自定义划分函数：
    - 从 Non-halogen compounds 和 Halogen compounds 中各抽取 k_shot 个样本放到训练集中。
    - 剩余样本按 1:1 比例分配到验证集和测试集，验证集和测试集样本数固定为 10。

    参数:
    - data: 样本数据，假设每个样本包含 'label' 和其他属性。
    - k_shot: 每个类别放入训练集的样本数量。
    - seed: 随机种子，确保划分可复现。

    返回:
    - 修改后的数据，其中每个样本添加了 'mask' 属性，值为 'train'、'valid' 或 'test'。
    """

    val_test_labels = {'No(is Non-halogen compounds)', 'Yes(is Halogen compounds)'}

    # 按标签分组样本
    train_data = {'No(is Non-halogen compounds)': [], 'Yes(is Halogen compounds)': []}
    val_test_data = {'No(is Non-halogen compounds)': [], 'Yes(is Halogen compounds)': []}

    for i in data:
        label = data[i]['label']
        if label in val_test_labels:
            # 将标签为 'Non-halogen compounds' 或 'Halogen compounds' 的样本分配到验证/测试集
            val_test_data[label].append(i)

    # 设置随机种子
    np.random.seed(seed)

    # 从 Non-halogen compounds 和 Halogen compounds 中各抽取 k_shot 个样本到训练集中
    for label, samples in val_test_data.items():
        if samples:
            selected_samples = np.random.choice(samples, min(k_shot, len(samples)), replace=False)
            train_data[label] = selected_samples  # 更新训练数据
            for sample in selected_samples:
                samples.remove(sample)  # 从验证/测试集中移除已选样本

    # 给训练集样本分配 mask
    train_samples = []
    train_samples.extend(train_data['No(is Non-halogen compounds)'])
    train_samples.extend(train_data['Yes(is Halogen compounds)'])
    for idx in train_samples:
        data[idx]['mask'] = 'train'

    # 将剩余的验证和测试集样本合并，并按 1:1 比例划分
    remaining_samples = val_test_data['No(is Non-halogen compounds)'] + val_test_data['Yes(is Halogen compounds)']
    
    # 随机打乱剩余的样本
    np.random.shuffle(remaining_samples)
    
    # 确保验证集和测试集的样本数分别为 10
    val_size = 10
    test_size = 10
    
    # 划分验证集
    for idx in remaining_samples[:val_size]:
        data[idx]['mask'] = 'valid'
    
    # 划分测试集
    for idx in remaining_samples[val_size:val_size + test_size]:
        data[idx]['mask'] = 'test'

    return data



