import numpy as np
from torch.utils.data import Dataset
from abc import ABC, ABCMeta, abstractmethod


class DatasetWithCollate(Dataset, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def get_collate_fn(self):
        pass

class MultiDataset(DatasetWithCollate):
    """
    One dataset that wraps different GraphTextDataset for training. It also dynamically manage the portion of
    the training datasets in each epoch based on validation results.
    """

    def __init__(
                self, 
                datas, 
                data_val_index=None, 
                dataset_multiple=1, 
                min_ratio=0.1,
                window_size=5, 
                patience=3, 
                mode=None, 
                ):
        self.datas = datas
        self.sizes = np.array([len(d) for d in datas])
        self.performance_record = []

        self.data_val_index = data_val_index
        if self.data_val_index is None:
            self.data_val_index = [[i] for i in range(len(self.datas))]

        self.patience = patience
        if isinstance(self.patience, int):
            self.patience = np.zeros(len(self.sizes)) + self.patience
        self.inpatience = np.zeros(len(self.patience))

        self.window_size = window_size
        if isinstance(self.window_size, int):
            self.window_size = np.zeros(len(self.sizes)) + self.window_size
        
        self.dataset_multiple = dataset_multiple
        if not isinstance(self.dataset_multiple, list):
            self.dataset_multiple = (np.zeros(len(self.sizes), dtype=float) + self.dataset_multiple)

        self.min_ratio = min_ratio
        if isinstance(self.min_ratio, float):
            self.min_ratio = np.zeros(len(self.sizes), dtype=float) + self.min_ratio

        self.mode = mode
        if mode is not None:
            self.mode = np.array([1 if m == "max" else -1 for m in self.mode])
        # self.walk_length = walk_length
        self.compute_sizes()

    def compute_sizes(self):
        self.aug_sizes = (self.sizes * np.array(self.dataset_multiple)).astype(int) # additional coefficient
        self.size_seg = np.cumsum(self.aug_sizes)
        # used to segment different datasets, easy to multiple different coefficient for different datasets
        self.ind2dataset = np.arange(len(self.datas)).repeat(self.aug_sizes) # index for different datasets
        self.sample_ind = (np.random.rand(len(self.ind2dataset)) * self.sizes.repeat(self.aug_sizes)).astype(int) # get sample index, note that have repetitive index
        self.data_start_index = np.r_[0, self.size_seg[:-1]]

    def __len__(self):
        return np.sum(self.aug_sizes)

    def __getitem__(self, index):
        dataset_ind = self.ind2dataset[index]
        dataset = self.datas[dataset_ind]  # get particular dataset
        ret_data = dataset[self.sample_ind[index]]
        return ret_data

    def get_collate_fn(self):
        return self.datas[0].get_collate_fn()

    def update(self, metric):   # 可能存在初始化记录，如果在最开始设置了运转验证集进行检查
        metric = np.array(metric)  # 评估结果
        p_records = np.array(self.performance_record)   # 评估的表现记录
        for i in range(len(self.datas)):
            if len(p_records) < self.window_size[i] or len(self.data_val_index[i]) == 0:
                continue

            vals = p_records[-int(self.window_size[i]):, self.data_val_index[i]]  # 从末尾取windoe_size个数据
            if self.mode is None:
                mode = np.ones(len(vals[0]), dtype=float)
            else:
                mode = self.mode[self.data_val_index[i]]
            mean = vals.mean()

            metric_vals = metric[self.data_val_index[i]]
            mean_improvement = (((metric_vals - mean) / mean) * mode).sum()
            if mean_improvement > 0:
                self.inpatience[i] = 0
            else:
                self.inpatience[i] += 1
            if self.inpatience[i] > self.patience[i]:
                self.dataset_multiple[i] = max(self.min_ratio[i],
                                               self.dataset_multiple[i] / 2)  # self.inpatience[i] = 0
        self.compute_sizes()
        self.performance_record.append(metric)