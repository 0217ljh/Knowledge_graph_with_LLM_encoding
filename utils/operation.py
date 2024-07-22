def Graph_splitter(dataset):
    return dataset.get_idx_split()


# 划分train,valid,test
def get_split_key(dataset_config):
        return dataset_config["dataset_name"] + "-" + dataset_config["task_level"] + "-" + dataset_config["task_name"]


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
    return "-".join([stage_config["dataset_names"], get_split_key(dataset_config), stage_config["stage"],
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
        preprocess_storage[split_key] = global_data
    return preprocess_storage,split_key

def get_construct_func(task_config):
    return task_config.get("construct")  # 获取键值，例如'LinkConstruct'