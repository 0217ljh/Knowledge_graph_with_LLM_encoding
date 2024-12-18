import numpy as np
import pandas as pd
from util.utils import evenly_divide_list
from typing import Optional, Callable, Any, Tuple
from Data_process.base_collect import MSAPygDataset
from Data_process.data_collect.datatype.spectrum.spe_process import Spectrum_GraphText

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
                                      get_standrd_meta: Optional[bool] = True,
                                      get_kwargs: Optional[bool] = False,
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
                    format_df.loc[key[4:],'strength'] = 'Feature {}. strength: <{}> of spectral peak ranged from <{}> to <{}>. '.format(key,value['strength'],value['start'],value['end'])
                    format_df.loc[key[4:],'min_index'] = False
            graph = self.GP.construct_graph(format_df,name_data[i])
            
            if get_standrd_meta:
                Graph_dict[name_data[i]] = {'graph':graph,
                                            'Mask':kwargs['kwargs'].get(name_data[i])['mask'],
                                            'Label':kwargs['kwargs'].get(name_data[i])['label']}
                
            elif get_kwargs:
                Graph_dict[name_data[i]] = {'graph':graph,
                                            'kwargs':[i for i in kwargs if i['Molecule']==name_data[i]] if kwargs else None}     
            else:
                Graph_dict[name_data[i]] = {'graph':graph,}                               
        return Graph_dict


# import numpy as np
# import pandas as pd
# from typing import Optional, Callable, Any
# from Data.base_collect import MSAPygDataset
# from Data.data_collect.datatype.spectrum.spe_process import Spectrum_GraphText
# import networkx as nx
# from scipy.ndimage import gaussian_filter1d
# class Spectrum_PygDataset(MSAPygDataset):
#     def __init__(self, root: str = "./cache_data", transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
#         super(Spectrum_PygDataset, self).__init__()
#         self.GP = Spectrum_GraphText()


#     def preprocess(self, data: Any, **kwargs) -> Any:
#         if data is None:
#             data = self.raw_data
#         for i in data:
#             box = {}
#             p = 1
#             sample = data[i]['Data']
#             wavelengths = sample.iloc[:, 0].values
#             intensities = sample.iloc[:, 1].values
#             # 计算一阶导数和二阶导数
#             first_derivative = np.gradient(intensities, wavelengths)
#             second_derivative = np.gradient(first_derivative, wavelengths)
#             # 寻找极小值峰值：一阶导数过零点且二阶导数为正
#             valleys = (np.diff(np.sign(first_derivative)) > 0) & (second_derivative[:-1] > 0)
#             valley_indices = np.where(valleys)[0]
#             # 确定强度阈值（这里选取均值）
#             mean_intensity = np.mean(intensities)
#             intensity_threshold = mean_intensity
#             # 筛选出强度远小于其他波长的峰值
#             valid_valleys = [idx for idx in valley_indices if intensities[idx] < intensity_threshold]
#             filtered_valleys = []
#             for idx in valid_valleys:
#                 # 找到该峰顶的半高宽（FWHM）
#                 half_min = (1-intensities[idx])/2+intensities[idx]
#                 left_idx = idx
#                 right_idx = idx
#                 # 向左查找半高点
#                 while left_idx > 0 and intensities[left_idx] < half_min:
#                     left_idx -= 1
#                 # 向右查找半高点
#                 while right_idx < len(intensities) - 1 and intensities[right_idx] < half_min:
#                     right_idx += 1
#                 # FWHM 范围
#                 start = wavelengths[left_idx]
#                 end = wavelengths[right_idx]
#                 filtered_valleys.append((idx, start, end))
#             # 只保留强度最小的前8个极小值
#             filtered_valleys = sorted(filtered_valleys, key=lambda x: intensities[x[0]])[:5]
#             # 按照波长对有效极小值进行排序
#             filtered_valleys.sort(key=lambda x: wavelengths[x[0]])
#             print(f"Final valid valleys {i}: {filtered_valleys}")
#             # 使用 filtered_valleys 来生成节点
#             if filtered_valleys:
#                 for idx, start, end in filtered_valleys:
#                     Interval = {}
#                     Interval['start'] = start
#                     Interval['end'] = end
#                     Interval['strength'] = intensities[idx]
#                     box[f'Node{p}'] = Interval
#                     p += 1
#             else:
#                 print(f"No valid valleys found for sample {i}. Skipping.")
#             data[i]['Node'] = box
#         self.pre_data = data
#         return data


#     def construct_graph(self, 
#                                         data: Any, 
#                                         get_meta: Optional[bool] = True,
#                                         #loader: Any, 
#                                         **kwargs
#                                         ) -> Any:
#             Graph_dict = {}
#             test_data = [value['Node'] for key,value in data.items()]
#             name_data = [value['Path'].split('\\')[-2] for key, value in data.items()]
#             format_df = pd.DataFrame(columns=['wavelength', 'strength'])
#             for i in range(len(test_data)):
#                 # 将df1的每一行数据输入到format_df中，进行相应的格式化
#                 for key,value in test_data[i].items():
#                         format_df.loc[key[4:],'wavelength'] = 'Feature {}. Wavelength: from <{}> to <{}>.(Unit: number of waves)'.format(key,value['start'],value['end'])
#                         format_df.loc[key[4:],'strength'] = 'Feature {}. strength: <{}>'.format(key,value['strength'])
#                         #format_df.loc[key[4:],'min_index'] = False
#                 print(format_df.head())
#                 graph = self.GP.construct_graph(format_df,name_data[i])
                
#                 if get_meta:
#                     meta_data = kwargs.get('kwargs', {})  # 使用 get 提供默认值
#                     Graph_dict[name_data[i]] = {
#                         'graph': graph,
#                         'Mask': meta_data.get(name_data[i], {}).get('mask', None),
#                         'Label': meta_data.get(name_data[i], {}).get('label', None)
#                     }
#                 else:
#                     Graph_dict[name_data[i]] = graph
#             return Graph_dict