import torch
from task_data.base_construct import MultiDataset
from torch_geometric.data import Dataset
from typing import Optional, Callable, Any, Tuple, Union, List
from torch.utils.data import Dataset,DataLoader, RandomSampler, DistributedSampler

def binary_auc_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])    # batch表示一张图里面，存在'num_classes'这个属性
    # score = torch.sigmoid(output)[:, -1]
    score = torch.nn.functional.softmax(output, dim=-1)[:, -1]   # 小于0.5表示标签0，大于0.5表示标签1
    return func(score, batch.y[:, -1].view(-1))

class DataWithMeta:
    def __init__(
            self,
            data: Dataset,
            batch_size: int,
            state_name: Optional[str] = None,
            feat_dim: int = 0,
            metric: Optional[str] = None,
            classes: Union[int, List[int]] = 2,
            is_regression: bool = False,
            meta_data: Any = None,
            sample_size: Optional[int] = -1,
    ):
        self.data = data
        self.batch_size = batch_size
        self.state_name = state_name
        self.feat_dim = feat_dim
        self.meta_data = meta_data
        self.metric = metric
        self.sample_size = sample_size
        if classes == -1:
            self.classes = data[0].num_classes
        self.classes = classes
        if isinstance(classes, list):
            self.num_tasks = len(classes)
        else:
            self.num_tasks = None
        self.is_regression = is_regression

    def pred_dim(self):
        if self.is_regression:
            return 1
        if self.num_tasks is not None:
            return self.num_tasks
        return self.classes

def make_data(name, data, split_name, metric, eval_func, num_classes, **kwargs):
    # Wrap GraphTextDataset with DataWithMeta for easy evaluator construction
    return DataWithMeta(data, kwargs["batch_size"], sample_size=kwargs["sample_size"], metric=metric,
                        state_name=split_name + "_" + name, classes=num_classes,
                        meta_data={"eval_func": globals()[eval_func], "eval_mode": kwargs["eval_mode"]}, )

def make_train_data(datasets, multiple, min_ratio, data_val_index=None):
        train_data = MultiDataset(datasets["train"], data_val_index=data_val_index, dataset_multiple=multiple,
                                  patience=3, window_size=5, min_ratio=min_ratio, )
        return train_data

def make_full_dm_list(datasets, multiple, min_ratio, train_data=None,**kwargs):
        text_dataset = {
            "train": DataWithMeta(
                            make_train_data(datasets, multiple, min_ratio) if not train_data else train_data,
                            #batch_size=20, 
                            batch_size=getattr(kwargs, "batch_size", 20),
                            #sample_size=-1, 
                            sample_size=getattr(kwargs, "sample_size", -1),
                            ),
            "val": datasets["valid"],   #  DataWithMeta
            "test":datasets["test"], }  #  DataWithMeta
        return text_dataset