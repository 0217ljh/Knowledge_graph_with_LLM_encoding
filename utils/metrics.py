from torchmetrics import AveragePrecision, AUROC,Accuracy
import torch

class select_metric():
    def __init__(self,name,**kwargs):
        self.name = name
        self.metric = None
        if self.name == 'acc':
            self.metric = Accuracy(task="multiclass", num_classes = getattr(kwargs, "num_classes", 2))
        elif self.name == 'auc':
            self.metric = AUROC(task=getattr(kwargs, "task", "binary"))
        elif self.name == 'apr':
            self.metric = MultiApr(num_labels=getattr(kwargs, "num_labels", 2))
        elif self.metric == "aucmulti":
            self.metric = MultiAuc(num_labels=getattr(kwargs, "num_labels", 2))
        else:
            raise ValueError(f"Metric: {self.name} not implemented")

class MultiApr(torch.nn.Module):
    def __init__(self, num_labels=1):
        super().__init__()
        self.metrics = torch.nn.ModuleList([AveragePrecision(task="binary") for i in range(num_labels)])

    def update(self, preds, targets):
        for i, met in enumerate(self.metrics):
            pred = preds[:, i]
            target = targets[:, i]
            valid_idx = target == target
            # print(pred[valid_idx])
            # print(target[valid_idx])
            met.update(pred[valid_idx], target[valid_idx].to(torch.long))

    def compute(self):
        full_val = []
        for met in self.metrics:
            try:
                res = met.compute()
                if res == res:
                    full_val.append(res)
            except BaseException:
                pass
        return torch.tensor(full_val).mean()

    def reset(self):
        for met in self.metrics:
            met.reset()


class MultiAuc(torch.nn.Module):
    def __init__(self, num_labels=1):
        super().__init__()
        self.metrics = torch.nn.ModuleList([AUROC(task="binary") for i in range(num_labels)])

    def update(self, preds, targets):
        for i, met in enumerate(self.metrics):
            pred = preds[:, i]
            target = targets[:, i]
            valid_idx = target == target
            # print(pred[valid_idx])
            # print(target[valid_idx])
            met.update(pred[valid_idx], target[valid_idx].to(torch.long))

    def compute(self):
        full_val = []
        for met in self.metrics:
            try:
                res = met.compute()
                if res == res:
                    full_val.append(res)
            except BaseException:
                pass
        return torch.tensor(full_val).mean()

    def reset(self):
        for met in self.metrics:
            met.reset()