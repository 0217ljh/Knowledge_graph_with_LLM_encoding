import pandas as pd
from typing import Optional, Callable, Any, Tuple
from Data.base_collect import MSAPygDataset

class Excel_PygDataset(MSAPygDataset):
    def __init__(self,
                    root: str = "./cache_data", 
                    transform: Optional[Callable] = None,
                    pre_transform: Optional[Callable] = None, 
                    ):
            # 继承父类中的loader属性
            super(Excel_PygDataset, self).__init__()
            self.GP = None


    def preprocess(self, 
                                  data: Any = None, 
                                  #loader: Any, 
                                  **kwargs
                                  ) -> dict:
        # 用来对输入的数据进行预处理
        if data is None:
            data = self.loader.data['1']['Data']
        format_dataframe = pd.DataFrame(columns=['Type', 'Subtype', 'Functional group', 'Absorb', 'Range', 'S-Strength'])
        for i in range(len(data)):
            # 将df1的每一行数据输入到format_df中，进行相应的格式化
            format_dataframe.loc[i,'Type'] = data.loc[i,'Key1(大类物质)'] # 烃类
            format_dataframe.loc[i,'Subtype'] = data.loc[i,'Key2(子类物质)'] # 烯烃
            if data.loc[i,'Key3(官能团名)'] != data.loc[i,'Key2(子类物质)'] :
                #format_df.loc[i,'Functional group'] = 'feature node. Functional group: <{}>.'.format(supplement_data.loc[i,'Key3(官能团名)'])
                format_dataframe.loc[i,'Functional group'] = data.loc[i,'Key3(官能团名)']
            else:
                format_dataframe.loc[i,'Functional group'] = None
            if data.loc[i,'Key4(子结构/振动类型)'] != data.loc[i,'Key3(官能团名)'] :
                format_dataframe.loc[i,'Absorb'] = 'feature node. Absorbtion: <{}>'.format(data.loc[i,'Key4(子结构/振动类型)'])
            else:
                format_dataframe.loc[i,'Absorb'] = None
            format_dataframe.loc[i,'Range'] = 'feature node.Range: <{}>'.format(data.loc[i,'范围'])
            format_dataframe.loc[i,'S-Strength'] = 'feature node.S-Strength: <{}>'.format(data.loc[i,'强度'])
        
        return format_dataframe
    
    def construct_graph(self, 
                                  data: Any, 
                                  key: str = '3.烯烃的骨架振动',
                                  #loader: Any, 
                                  **kwargs
                                  ) -> Any:
        # 从文本文件构造图谱
        if data is None:
            data = self.pre_data
        df1 = data[key]
        # 创建一个空的dataframe，列名为'Type', 'Subtype', 'Functional group', 'Absorb', 'Range', 'S-Strength'
        format_df = pd.DataFrame(columns=['Type', 'Subtype', 'Functional group', 'Absorb', 'Range', 'S-Strength'])
        for i in range(len(df1)):
            # 将df1的每一行数据输入到format_df中，进行相应的格式化
            format_df.loc[i,'Type'] = 'hydrocarbons' # 烃类
            format_df.loc[i,'Subtype'] = 'alkene' # 烯烃
            format_df.loc[i,'Functional group'] = 'feature node. Functional group: <{}>. <{}>'.format(df1.loc[i,'文本名称'],df1.loc[i,'Latex'])
            format_df.loc[i,'Absorb'] = 'feature node. Absorbtion: <{}>'.format(df1.loc[i,'备注'])
            format_df.loc[i,'Range'] = 'feature node.Range: <{}>'.format(df1.loc[i,'范围'])
            format_df.loc[i,'S-Strength'] = 'feature node.S-Strength: <{}>'.format(df1.loc[i,'强度'])
            graph = self.GP.construct_graph(format_df)
        return graph