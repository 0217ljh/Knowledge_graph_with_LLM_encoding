import pandas as pd
from typing import Optional, Callable, Any, Tuple
from Data.base_collect import MSAPygDataset
from Data.data_collect.datatype.knowledge_map.k_p_process import Function_group_GraphText

class Knowledge_graph_PygDataset(MSAPygDataset):
    def __init__(self,
                    root: str = "./cache_data", 
                    transform: Optional[Callable] = None,
                    pre_transform: Optional[Callable] = None, 
                    ):
            # 继承父类中的loader属性
            super(Knowledge_graph_PygDataset, self).__init__()
            self.GP = Function_group_GraphText()


    def preprocess(self, 
                                  data: Any, 
                                  #loader: Any, 
                                  **kwargs
                                  ) -> dict:
        # 用来对输入的数据进行预处理
        if data is None:
            data = self.raw_data
        data = data.drop([0, 1])
        # 筛选出一行中只有1列存在数值的行，将对应的值与行号保存为字典
        FG_dict = {}
        for i in range(len(data)):
            if len(data.iloc[i].dropna()) == 1:
                FG_dict[i] = data.iloc[i].dropna()['Latex']
        # 根据字典，将对应的功能团数据保存到字典中
        FG_data = {}
        # 从FG_dict的键中两两创建索引，举例，[i, i+1]，[i+2, i+3]，[i+4, i+5]...
        Index = []
        for i in range(len(FG_dict.keys())):
            if i+1 < len(FG_dict.keys()):
                Index.append([list(FG_dict.keys())[i]+1,list(FG_dict.keys())[i+1]])
            else:
                Index.append([list(FG_dict.keys())[i]+1, len(data)])

        # 

        #用Index作为索引，从test_data截取数据，分别放入字典中，字典名为FG_dict的键值
        FG_data = {}
        for i in range(len(Index)):
            FG_data[list(FG_dict.values())[i]] = data.iloc[Index[i][0]:Index[i][1]]
            
        for key,value in FG_data.items():
            print(key)
            # 在value中，对备注一列，全部用key值进行填充
            value.loc[:,'备注'] = key
            # 在calue中，对所有空值用None进行填充
            value = value.fillna('None')
            # 重设索引，从0开始
            value = value.reset_index(drop=True)
            FG_data[key] = value

        self.pre_data = FG_data
        return FG_data
    
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