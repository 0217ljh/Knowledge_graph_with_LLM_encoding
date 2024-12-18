import os
import sys
import fire
import json
import torch
import evaluate
import transformers
import numpy as np
import torch.nn as nn
import pandas as pd
import datetime
import wandb
import bitsandbytes
import collections
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Union
from datasets import load_dataset, Dataset, DatasetDict
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, GenerationConfig

class DataLoader():
    def __init__(self, **Configs):
        # Initialisation Configuration
        self.data_path: Optional[str] = None # 数据文件存放路径
        self.dataset: Optional[Dict] = None

        self.B_data: Optional[Dataset] = None
        self.O_data: Optional[Dataset] = None
        self.predict_result: Optional[Dataset] = None
        self.predict_data: Optional[DatasetDict] = None
        self.data_preprocess: Optional[DatasetDict] = None
        self.predict_data_path: Optional[str] = None

        self.tokenizer:Optional[AutoTokenizer] = None
        self.tokenizer_name: Optional[str] = None

        self.prompt_template: Optional[str] = None
        #self.prompter: Optional[Prompter] = None

        self.model: Optional[AutoModelForCausalLM] = None
        self.model_name: Optional[str] = None
        self.lora_path: Optional[str] = None
        self.output_dir = None   

        self.default_model_tokenizer: str = 'meta-llama/Llama-2-7b-hf'
        self.add_eos_token: Optional[bool] = None
        self.generation_config: Optional[GenerationConfig] = None

        for key, value in Configs.items():
             setattr(self, str(key), value)
         
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
    def load_data(self, data_path: str, type: str, split: float = 0.1):
        # 如果type输入为text，则读取文本文件，包括各种excel文件，json文件，csv文件等
        # 如果typr输入为picture，则读取图片文件，包括jpg，png等
        self.data_path = data_path
        if type == 'text':
            data = self.load_text_data(data_path)
        elif type == 'picture':
            data = self.load_picture_data(data_path)
        return data

    def load_text_data(self, data_path: str, is_shut_down: bool = False, shut_down_index: int = 5):
        """"
        读取文本数据
        """
        # 先判断data_path路径是否存在
        dataset = {}
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} not found.")
        # 判断data_path是文件夹还是文件
        if not os.path.isdir(data_path):
            data = pd.read_csv(data_path) # 是文件直接读取就行
        else:
            path_list = os.listdir(data_path) # 默认处理一层子文件夹
            for i in tqdm(range(len(path_list))):
                if is_shut_down: # Setting the abort position
                    if i+1 > shut_down_index:break 
                output = self.load_item(
                    os.path.join(data_path, path_list[i])
                    )
                dataset[str(i+1)] = output
        self.dataset = dataset
        
        return self.dataset
    
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
            raise FileNotFoundError(f"File {os.path.join(subfile_path, name)} load fail.")
        

        output = {'Path': os.path.join(subfile_path, name_list[1]),
                  'Name': name,
                  'Data': data}
        return output

    def load_picture_data(self, data_path: str, split: float = 0.1):
        """"
        读取图像数据
        """
        data = pd.read_csv(data_path)
        return data
    
    