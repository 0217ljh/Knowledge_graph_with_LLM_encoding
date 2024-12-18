import os
import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
from typing import Optional, Callable, Any, Tuple,Dict

class DataLoader():
    """
    Used to load different forms of data, including text data, image data, spectral data, etc.
    """
    def __init__(self, **Configs):
        # Initialisation Configuration
        self.data_path: Optional[str] = None 
        self.data: Optional[Dict] = None 
        self.dataset: Optional[Dict] = None

        for key, value in Configs.items():
             setattr(self, str(key), value)
         
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
    def load_data(self, 
                  data_path: str, 
                  type: str, 
                  split: float = 0.1
                  ):
        """
        Load data, if type input for text, then read text files, including a variety of excel files, json files, csv files, etc.; if typr input for picture, then read picture files, including jpg, png, etc.
        """
        # 
        # 
        self.data_path = data_path
        if type == 'text':
            self.data = self.load_text_data(data_path)
        elif type == 'graph':
            self.data = self.load_graph_data(data_path)
        elif type == 'picture':
            self.data = self.load_picture_data(data_path)
        elif type == 'spectrum':
            self.data = self.load_spectrum_data(data_path)

    def load_text_data(self, 
                       data_path: str, 
                       is_shut_down: bool = False, 
                       shut_down_index: int = 5
                       ) -> dict:
        """
        Load text data, including a variety of excel files, json files, csv files, etc.
        """
        # 先判断data_path路径是否存在
        dataset = {}
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} not found.")
        # 判断data_path是文件夹还是文件
        if not os.path.isdir(data_path):
            data = pd.read_excel(data_path) # 是文件直接读取就行
            output = {'Path': data_path,
                  'Name': None,
                  'Data': data}
            dataset['1'] = output
        else:
            path_list = os.listdir(data_path) # 默认处理一层子文件夹
            for i in tqdm(range(len(path_list))):
                if is_shut_down: # Setting the abort position
                    if i+1 > shut_down_index:break 
                output = self.load_item(
                    os.path.join(data_path, path_list[i])
                    )
                dataset[str(i+1)] = output
        #self.dataset = dataset
        
        return dataset
    
    def load_item(self, subfile_path: str):
        """""
        读取文本数据
        """
        # 查看subfile_path下的所有文件
        suffix = ['csv','json','txt','xlsx','xls']
        name_list = os.listdir(subfile_path)
        # 查找name_list中的文件后缀存在于suffix中的文件并输出为变量name
        name = [name for name in name_list if name.split('.')[-1] in suffix][0]
        try:
            data = pd.read_csv(os.path.join(subfile_path, name))
        except:# 报错并输出此时的文件地址
            try:
                data = nx.read_gexf(subfile_path)
            except:
                raise FileNotFoundError(f"File {os.path.join(subfile_path, name)} load fail.")
        

        output = {'Path': os.path.join(subfile_path, name),
                  'Name': name,
                  'Data': data}
        return output

    def load_graph_data(self, 
                       data_path: str, 
                       is_shut_down: bool = False, 
                       shut_down_index: int = 5):
        """
        读取图数据
        """
        # 先判断data_path路径是否存在
        dataset = {}
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} not found.")
        # 判断data_path是文件夹还是文件
        if not os.path.isdir(data_path):
            KM = nx.read_gexf(data_path) # 是文件直接读取就行
            #KM = nx.read_gexf('.\Datasets\Knowledge graph(chemistry)\knowledge_graph.gexf')
            output = {'Path': data_path,
                  'Name': None,
                  'Data': KM}
            dataset['1'] = output
        else:
            path_list = os.listdir(data_path) # 默认处理一层子文件夹
            for i in tqdm(range(len(path_list))):
                if is_shut_down: # Setting the abort position
                    if i+1 > shut_down_index:break 
                output = self.load_item(
                    os.path.join(data_path, path_list[i])
                    )
                dataset[str(i+1)] = output
        #self.dataset = dataset
        
        return dataset

    def load_picture_data(self, data_path: str, split: float = 0.1):
        """"
        读取图像数据
        """
        pass
      
    def load_spectrum_data(self, data_path: str):
        """
        读取光谱数据
        """
        pass