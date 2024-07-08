import os
import yaml
import torch
import numpy as np
import pandas as pd
import networkx as nx
from utils import utils
from utils.operation import *
from datetime import datetime
import matplotlib.pyplot as plt
from types import SimpleNamespace
from utils.metrics import select_metric

from Data.data_collect.Merge_collect import Merge_PygDataset
from Data.data_collect.datatype.excel.excel_collect import Excel_PygDataset
from Data.data_collect.datatype.spectrum.spe_collect import Spectrum_PygDataset
from Data.data_collect.datatype.knowledge_map.k_p_collect import Knowledge_graph_PygDataset

from task_data.graph.graph_task_dataset import MolOFADataset
from task_data.graph.graph_construct import ConstructMolCls
from task_data.eval_data_construct import make_data,make_train_data,make_full_dm_list

from Models.LLM.LLM_Encoder import SentenceEncoder
from Models.models import PyGRGCNEdge,BinGraphModel,BinGraphAttModel


from Light.train import lightning_fit
from Light.data_module import DataModule
from Light.metric import flat_binary_func,EvalKit
from Light.template import ExpConfig,GraphPredLightning
from pytorch_lightning.loggers import WandbLogger

configs = []
configs.append(
    utils.load_yaml(
        os.path.join(
            #os.path.dirname(__file__), "configs", "default_config.yaml"
            './', "Configs", "default_config.yaml"
        )
    )
)

# 可选的配置更新
# 添加新的config
override_config = utils.load_yaml('./Configs/Override/e2e_all_config.yaml')
configs.append(override_config)

# 更新配置
mod_params = utils.combine_dict(*configs)
#mod_params = merge_mod(mod_params, params.opts)
mod_params = utils.merge_mod(mod_params, [])

# 信息存储路径
curtime = datetime.now()
exp_name = str(curtime).replace(" ", "_")
exp_name = exp_name.replace(":", "_")
exp_name = exp_name.replace(".", "_")
exp_name = exp_name.split("_")[0]
exp_dir = os.path.join("./saved_exp", exp_name)
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
# 保存并转换配置文件
with open(os.path.join(exp_dir, "command"), "w") as f:  # 保存配置文件
    yaml.dump(mod_params, f)
mod_params["exp_dir"] = exp_dir
params = SimpleNamespace(**mod_params)
utils.set_random_seed(params.seed)
torch.set_float32_matmul_precision("high")
params.log_project = "full_cdm"

params.exp_name += f"_{params.llm_name}_ofa1"

task_config_lookup = utils.load_yaml(
    os.path.join(
        #os.path.dirname(__file__), "configs", "task_config.yaml"
        r'.', "Configs", "task_config.yaml"
        )
)
data_config_lookup = utils.load_yaml(
    os.path.join(
        #os.path.dirname(__file__), "configs", "data_config.yaml"
        r'.',"Configs", "data_config.yaml"
        )
    )
if isinstance(params.task_names, str):
    task_names = [a.strip() for a in params.task_names.split(",")]
else:
    task_names = params.task_names

#*************************************************

device, gpu_ids = utils.get_available_devices()
gpu_size = len(gpu_ids)

utils.set_random_seed(0)

encoder = SentenceEncoder('ST',batch_size = 1)

dataset_output = MolOFADataset(name = 'Base_classification', load_texts = False,encoder=encoder,force_reload = True)

#*****************************************************
task_config = task_config_lookup[task_names[0]]
Stage_Config = task_config['eval_set_constructs']
stage_config = Stage_Config[0]

dataset_config = data_config_lookup[task_names[0]]


# 数据集字典化
test_dataset = {}   # 储存总数据集
test_dataset_split = {}    # 储存划分子数据集的掩膜
test_preprocess_storage = {}   # 储存预处理结果
test_datasets = {"train": [], "valid": [],"test": []}   # 储存子数据集
test_stage_names = {"train": [], "valid": [], "test": []}

result = []
result_valid = []
for i in Stage_Config:
    if "dataset" not in i:  # 如果没有更换不同的数据集，此时默认的数据集为 config["dataset"]，即外层标的那个
            i["dataset"] = task_config["dataset"]
    test_dataset[dataset_config['dataset_name']] = dataset_output   # 加载总数据集
    test_dataset_split, split_key = get_data_split(test_dataset,test_dataset_split,dataset_config)   # 根据stage划分不同的子数据集
    stage_name = get_stage_name(i, dataset_config)    # 获取对应的子数据集名称
    #if i["stage"] != "train" and stage_name in test_stage_names[i["stage"]]: #包括验证和测试的数据集只构建一次
    if stage_name in test_stage_names[i["stage"]]: #子数据集只构建一次
        result.append(test_stage_names[i["stage"]].index(stage_name)) # 返回这个eval数据集的索引
        if i["stage"] == "valid":
            result_valid.append([test_stage_names[i["stage"]].index(stage_name)])
        continue
    test_preprocess_storage, split_key = get_global_data(test_dataset,test_dataset_split,test_preprocess_storage,dataset_config)
    prompt_feats = dataset_output.get_prompt_text_feat(dataset_config["task_level"]) # Prompt的相关部分
    data = ConstructMolCls(dataset = test_dataset[dataset_config['dataset_name']],
                            split = test_dataset_split[split_key],
                            split_name = i["split_name"],
                            prompt_feats = prompt_feats,
                            to_bin_cls_func = dataset_config["process_label_func"] if dataset_config.get("process_label_func") else None,
                            task_level = dataset_config["task_level"],
                            global_data = test_preprocess_storage[split_key],
                            **dataset_config["args"],
                            )
    if i["stage"] == "train":
        test_datasets[i["stage"]].append(data)
    else:
        eval_data = make_data(i["dataset"],
                            data,
                            i["split_name"],
                            dataset_config["eval_metric"],
                            dataset_config["eval_func"],
                            dataset_config["num_classes"],
                            batch_size=20,
                            sample_size=-1,
                            eval_mode=dataset_config["eval_mode"])
        test_datasets[i["stage"]].append(eval_data)
    test_stage_names[i["stage"]].append(stage_name)
    result.append(test_stage_names[i["stage"]].index(stage_name))
    if i["stage"] == "valid":
        result_valid.append([test_stage_names[i["stage"]].index(stage_name)])

val_task_index_lst = result_valid
val_pool_mode = task_config['eval_pool_mode']
#print(result,'\n',result_valid,'\n',val_pool_mode)

#*******************************************************

if encoder is not None:
    encoder.flush_model()

out_dim = params.emb_dim + (params.rwpe if params.rwpe is not None else 0)

gnn = PyGRGCNEdge(
    params.num_layers,
    5,
    out_dim,
    out_dim,
    drop_ratio=params.dropout,
    JK=params.JK,
)

bin_model = BinGraphAttModel if params.JK == "none" else BinGraphModel
model = bin_model(model=gnn, llm_name=params.llm_name, outdim=out_dim, task_dim=1,
                    add_rwpe=params.rwpe, dropout=params.dropout)

#*************************************************************************************
if hasattr(params, "d_multiple"):
    if isinstance(params.d_multiple, str):
        data_multiple = [float(a) for a in params.d_multiple.split(",")]
    else:
        data_multiple = params.d_multiple
else:
    data_multiple = [1]

if hasattr(params, "d_min_ratio"):
    if isinstance(params.d_min_ratio, str):
        min_ratio = [float(a) for a in params.d_min_ratio.split(",")]
    else:
        min_ratio = params.d_min_ratio
else:
    min_ratio = [1]

train_data = make_train_data(test_datasets,data_multiple, min_ratio, data_val_index=val_task_index_lst)
text_dataset = make_full_dm_list(
    test_datasets, data_multiple, min_ratio, train_data
)
params.datamodule = DataModule(
    text_dataset, gpu_size=gpu_size, num_workers=params.num_workers
)

#print(text_dataset)
#**************************************************************************************************************
eval_data = text_dataset["val"] + text_dataset["test"]
val_state = [dt.state_name for dt in text_dataset["val"]]
test_state = [dt.state_name for dt in text_dataset["test"]]
eval_state = val_state + test_state
eval_metric = [dt.metric for dt in eval_data]
eval_funcs = [dt.meta_data["eval_func"] for dt in eval_data]
loss = torch.nn.BCEWithLogitsLoss()  # 是二分类
evlter = []

for dt in eval_data:
    if dt.metric == "acc":
        evlter.append(select_metric(name = dt.metric, num_classes = dt.classes).metric)
        #evlter.append(Accuracy(task="multiclass", num_classes=dt.classes))
    elif dt.metric == "auc":
        evlter.append(select_metric(name = dt.metric, task = 'binary').metric)
        #evlter.append(AUROC(task="binary"))
    elif dt.metric == "apr" or dt.metric == "aucmulti":
        evlter.append(select_metric(name = dt.metric, num_labels = dt.classes).metric)

metrics = EvalKit(
    eval_metric,
    evlter,
    loss,
    eval_funcs,
    flat_binary_func,
    eval_mode="max",
    exp_prefix="",
    eval_state=eval_state,
    val_monitor_state=val_state[0],
    test_monitor_state=test_state[0],
)

#****************************************************
optimizer = torch.optim.Adam(
    model.parameters(), lr=params.lr, weight_decay=params.l2
)
lr_scheduler = {
    "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, 15, 0.5),
    "interval": "epoch",
    "frequency": 1,
}

exp_config = ExpConfig(
    "",
    optimizer,
    dataset_callback=train_data.update,
    lr_scheduler=lr_scheduler,
)
exp_config.val_state_name = val_state
exp_config.test_state_name = test_state

pred_model = GraphPredLightning(exp_config, model, metrics)

#***************************************************************调试的相关测试
# train_datasets_in_test = params.datamodule.train_dataloader()
# #print(len(train_datasets_in_test))
# for batch in train_datasets_in_test:
#     print(batch)
#     break

wandb_logger = WandbLogger(
    project=params.log_project,
    name=params.exp_name,
    save_dir=params.exp_dir,
    offline=params.offline_log,
)

strategy = "deepspeed_stage_2" if gpu_size > 1 else "auto"
val_res, test_res = lightning_fit(
    wandb_logger,
    pred_model,
    params.datamodule,
    metrics,
    params.num_epochs,
    strategy=strategy,
    save_model=False,
    load_best=params.load_best,
    reload_freq=1,
    test_rep=params.test_rep,
    val_interval=params.val_interval
)