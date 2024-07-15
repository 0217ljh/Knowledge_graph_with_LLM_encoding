import torch
import os.path as osp
import copy
from torchmetrics import Metric
from typing import Any, Callable, Optional, Literal, List, Union

class EvalKit(torch.nn.Module):
    def __init__(
        self,
        metric_name: Union[str, List[str]],   #  ['auc', 'auc', 'auc', 'auc']
        evlter: Any,   #  ['BinaryAUC()', 'BinaryAUC()', 'BinaryAUC()', 'BinaryAUC()']
        loss: Any,     #  BCEWithLogitsLoss()
        evlter_func: Union[Callable, List[Callable]] = None,  # [binary_auc_func, binary_auc_func, binary_auc_func, binary_auc_func]
        loss_func: Callable = None,   # flat_binary_func
        val_monitor_state: Optional[str] = "valid",  # valid_small_molecule
        test_monitor_state: Optional[str] = "test",  # test_small_molecule
        eval_mode: Literal["min", "max"] = "min",    # 'max'
        exp_prefix: str = "",
        eval_state: List[str] = ["train_eval", "test", "valid"],
    ):
        super().__init__()
        self.eval_states = eval_state  # 阶段文本描述
        self.loss = loss  # BCEWithLogitsLoss()
        self.eval_mode = eval_mode  # 'max'
        self.val_monitor_state = val_monitor_state   # 开始的位置
        self.test_monitor_state = test_monitor_state # 开始的位置
        self.exp_prefix = exp_prefix
        self.evlters = torch.nn.ModuleDict()
        self.loss_func = loss_func   # flat_binary_func
        self.evlter_func = {}
        self.metric_name = {}
        for i, state in enumerate(eval_state):   # 这里指的是包括验证集和测试集在内所有需要评估的部分
            if isinstance(evlter, Metric):
                self.metric_name[state] = osp.join(
                    exp_prefix, state, metric_name
                )
                self.evlters[state] = copy.deepcopy(evlter)
                self.evlter_func[state] = evlter_func
            else:
                self.metric_name[state] = osp.join(
                    exp_prefix, state, metric_name[i]
                )
                self.evlters[state] = evlter[i]
                self.evlter_func[state] = evlter_func[i]   

        self.val_metric = self.metric_name[self.val_monitor_state]
        self.test_metric = self.metric_name[self.test_monitor_state]

    def compute_loss(self, output: Any, batch: Any):
        # 用在训练/验证/测试过程中，作为损失函数进行计算，先获取class_node
        return self.loss_func(self.loss, output, batch)   # flat_binary_func(BCEWithLogitsLoss())

    def has_eval_state(self, state: str):# 判断是否存在，并返回bool值
        return state in self.eval_states   # ['valid_small_molecule','test_small_molecule','train_small_molecule','valid_small_molecule']

    def get_evlter(self, state: str):
        return self.evlters[state]

    def eval_step(self, output: Any, batch: Any, state: str):
        # 用于评估/测试部分，用在step上
        evlter = self.get_evlter(state)  # BinaryAUC()
        return self.evlter_func[state](evlter, output, batch)  # binary_auc_func(BinaryAUC())   

    def eval_epoch(self, state: str):
        # 用于训练/验证/测试部分，处理一个完整的epoch
        evlter = self.get_evlter(state)
        return evlter.compute()   # BinaryAUC().compute()。应该是收集之前step中所有的结果，输出平均值之类的处理方法

    def eval_reset(self, state: str):
        evlter = self.get_evlter(state)
        evlter.reset()

    def get_metric_name(self, state: str):
        return self.metric_name[state]

def flat_binary_func(func, output, batch):
    labels = batch.bin_labels[batch.true_nodes_mask]    # true_nodes_mask指的是用于获取class_node的掩膜。这一步得到的是真实的标签
    valid_ind = labels == labels  # 这个其实就是个掩膜。 [True,...,True,...,True]
    return func(output.view(-1)[valid_ind], labels[valid_ind])