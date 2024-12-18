from abc import ABC, abstractmethod
from Data_process.base_loader import *
from typing import Optional, Callable, Any, Tuple

class MSAPygDataset(ABC):
    r"""
    实现多模态光谱分析的基础数据类型。主要实现功能包括：
    1. 数据集加载
    2. 数据集的预处理
    3. 使用LLM进行文本到特征的转换
    ...
    """

    def __init__(self, 
                 #name: str, load_texts: bool, 
                 #encoder: Optional[SentenceEncoder] = None,
                 root: str = "./cache_data", 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, 
                 ):
        self.loader = DataLoader()
        self.raw_data = None
        self.pre_data = None


        #self.name = name
        #self.load_texts = load_texts
        #self.root = root
        #self.encoder = encoder   # 补充了 'processed_dir' , 'processed_paths' , 'raw_dir' 三个属性
        #if not self.load_texts:
        #    assert self.encoder is not None
        #    suffix = self.encoder.llm_name
        #else:
        #    suffix = 'raw'
        #self.data_dir = osp.join(self.root, self.name, suffix)

        ##super().__init__(self.data_dir, transform, pre_transform)   # 这里super方法，调用了下面的process函数
        ##safe_mkdir(self.data_dir)

        # # load text to the dataset instance
        # if self.load_texts:
        #     self.texts = torch.load(self.processed_paths[1])

        ##self.data, self.slices = torch.load(self.processed_paths[0])
        ##self.side_data = pth_safe_load(self.processed_paths[2])

    def data_loader(self, 
                    path: str,
                    type = 'text',
                    #batch_size: int, 
                    #shuffle: bool = False, 
                    **kwargs
                    ) -> DataLoader:
        # 用来读取不同种类的输入文件，返回一个DataLoader对象.
        file_loader = self.loader 
        file_loader.load_data(path, type)
        self.raw_data = file_loader.data   # 保存原始数据
        return file_loader
    
    @abstractmethod
    def preprocess(self):
        # 保存为pre_data
        pass

    @abstractmethod
    def construct_graph(self):
        pass
    
    def graph_visualisation(self, 
                            graph = None, 
                            **kwargs
                            ) -> Any:
        # 对构造的图谱进行可视化
        if graph is None:
            graph = self.GP.graph
        self.GP.draw_graph(graph,**kwargs)