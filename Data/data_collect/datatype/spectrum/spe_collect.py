import numpy as np
import pandas as pd
from utils.utils import evenly_divide_list
from typing import Optional, Callable, Any, Tuple
from Data.base_collect import MSAPygDataset
from Data.data_collect.datatype.spectrum.spe_process import Spectrum_GraphText

class Spectrum_PygDataset(MSAPygDataset):
    def __init__(self,
                        root: str = "./cache_data", 
                        transform: Optional[Callable] = None,
                        pre_transform: Optional[Callable] = None, 
                        ):
                # 继承父类中的loader属性
                super(Spectrum_PygDataset, self).__init__()
                self.GP = Spectrum_GraphText()

    def preprocess(self, 
                                      data: Any, 
                                      #loader: Any, 
                                      **kwargs
                                      ) -> Any:
        # 从光谱文件构造图谱
        # 生成波长节点
        # strength node.5个区间：1.弱 2.较弱 3.中等 4.较强 5.强
        if data is None:
            data = self.raw_data
        strength_dict = {'5':[0,0.2],'4':[0.2,0.4],'3':[0.4,0.6],'2':[0.6,0.8],'1':[0.8,1]}
        for i in data:
        
            #f p > 1:
            #    break
            box = {}
            p = 1
            sample = data[i]['Data']
            # 将sample的第一列均匀分为20个区间,输出为字典，格式如下：{'区间1':[{行索引1：波长}，{行索引2：波长}]}。行索引1是这个区间的开始，行索引2是这个区间的末端
            a = [i for i in sample.iloc[:,0].index]
            b = evenly_divide_list(a, 20)
            # 根据b的索引，在data中选取行
            sample_split = [sample.loc[i] for i in b]
            for j in sample_split:
                Interval = {}
                start = j.iloc[0,0]
                end = j.iloc[-1,0]
                # 判断data_split第二列中的最大值的大小，按照strength_dict的区间划分
                min_value = j.iloc[:,1].min()
                for key in strength_dict:
                    if min_value >= 1:
                        strength = 1
                        break
                    if min_value >= strength_dict[key][0] and min_value < strength_dict[key][1]:
                        strength = key
                        break
                Interval['start'] = start
                Interval['end'] = end
                Interval['strength'] = strength
                box['Node{}'.format(p)] = Interval
                p = p + 1
            all_min = sample.iloc[:,1].min()
            # 查找all_min在data中的索引
            Min_index = sample[sample.iloc[:,1] == all_min].index.tolist()
            Min_index = np.linspace(0,20,len(sample))[Min_index[0]]
            # 判断Min_index在哪个区间，举例：如果0<Min_index<1，则位于区间1
            for k in range(20):
                if Min_index > k and Min_index < k+1:
                    Min_poistion = f'Node{k+1}'
                    break
            box['Min_index'] = [Min_index,Min_poistion]
            data[i]['Node'] = box
        self.pre_data = data
        return data
    
    def construct_graph(self, 
                                      data: Any, 
                                      get_meta: Optional[bool] = True,
                                      #loader: Any, 
                                      **kwargs
                                      ) -> Any:
        Graph_dict = {}
        test_data = [value['Node'] for key,value in data.items()]
        name_data = [value['Path'].split('\\')[-2] for key, value in data.items()]
        format_df = pd.DataFrame(columns=['wavelength', 'strength','min_index'])
        for i in range(len(test_data)):
            # 将df1的每一行数据输入到format_df中，进行相应的格式化
            for key,value in test_data[i].items():
                if key == 'Min_index':
                    format_df.loc[value[1][4:],'min_index'] = 'Feature Node. Minimum transmittance position: {}'.format(value[0])
                    'The minimum value of the second column is in the <{}> interval'.format(value)
                    #format_df.loc[key,'strength'] = 'The minimum value of the second column is <{}>'.format(spectral_data['26']['Data'].iloc[:,1].min())
                    continue
                else:
                    format_df.loc[key[4:],'wavelength'] = 'Feature {}. Wavelength: from <{}> to <{}>.(Unit: number of waves)'.format(key,value['start'],value['end'])
                    format_df.loc[key[4:],'strength'] = 'Feature {}. strength: <{}>'.format(key,value['strength'])
                    format_df.loc[key[4:],'min_index'] = False
            graph = self.GP.construct_graph(format_df,name_data[i])
            
            if get_meta:
                Graph_dict[name_data[i]] = {'graph':graph,
                                            'Mask':kwargs['kwargs'].get(name_data[i])['mask'],
                                            'Label':kwargs['kwargs'].get(name_data[i])['label']}
                
            else:
                Graph_dict[name_data[i]] = graph
        return Graph_dict