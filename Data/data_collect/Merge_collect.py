from Data.base_collect import MSAPygDataset
from Data.data_collect.Merge_process import Merge_GraphText
from typing import Optional, Callable, Any, Tuple

class Merge_PygDataset(MSAPygDataset):
    def __init__(self,
                        text_graph,
                        spectrum_graph,
                        root: str = "./cache_data", 
                        transform: Optional[Callable] = None,
                        pre_transform: Optional[Callable] = None, 
                        **kwargs: Any
                        ):
                # 继承父类中的loader属性
                super(Merge_PygDataset, self).__init__()
                self.text_graph = text_graph
                #self.spectrum_graph = spectrum_graph
                #self.spectrum_graph = [{key:value['graph']} for key,value in spectrum_graph.items()]
                #self.Mask = [{key:value['Mask']} for key,value in self.spectrum_graph.items()]
                #self.Label = [{key:value['Label']} for key,value in self.spectrum_graph.items()]
                self.spectrum_graph = spectrum_graph['graph']
                self.Mask = spectrum_graph['Mask']
                self.Label = spectrum_graph['Label']
                self.GP = Merge_GraphText(text_graph, self.spectrum_graph)
                for key, value in kwargs.items():
                    setattr(self, key, value)
    
    def preprocess(self):
        # 查找与吸收峰相对应的官能团
        range_node, wave = self.GP.get_information()
        
        ######### 输出对应的特征波段信息
        #for i in wave:
            # 删除i[0]中的'wave'，并与'wavelength'拼接
            #index_symbol = i[0]
            #print(self.spectrum_graph.nodes(data=True)[index_symbol])
        # 获取连接在不同节点上的可能存在的官能团序列
        connected_dict = self.GP.map_information(range_node, wave)

        ######### connected_dict对应的官能团信息
        #for key,value in connected_dict.items():
        #    print(key)
        #    for k in value:
        #        print([i for i in self.text_graph.predecessors(k.replace('Functional group','Range'))])
        #    print('-----------------')
        return connected_dict

    def Negative_relevant_information(self,is_print = False):
        # 确认不存在的官能团内容
        #{'胺，醇':[4000,3200],'炔烃':[3310,3300],'芳香族化合物,烯族':[3100,3000],'异氰酸盐，异氰酸酯，硫氰酸盐，硫氰酸酯，异硫氰酸酯，异硫氰酸盐':[2500,2000],'酯，酮，羧酸，羧酸盐':[1870,1550],'烯族化合物':[1690,1620],'硝酸盐化合物':[1655,1610],'硝基化合物':[1600,1510],'芳环系统':[1600,1450],'硫代氧化物，砜，磺酸酯基':[1420,990]}

        non_dict = {'#Amines,#Hydroxyl':[4000,3200],'#Alkynyl':[3310,3300],'#Phenyl,#Alkenyl':[3100,3000],'#Carboalkoxy,#Ketone,#Carboxyl,#Aldehyde':[1870,1550],'#Alkenyl':[1690,1620],'#Nitro':[1600,1510],'#Phenyl,#Pyridyl':[1600,1450],'#Sulfide':[1420,990],
        '#Phenyl':[2000,1600]
                    }
        # 在SG的不同波段节点中，查找以上的官能团是否存在
        all_Table,isexist,score_dict = self.GP.find_no_exist_FG(table=non_dict,is_print=is_print)
        return all_Table,isexist,score_dict

    def construct_graph(self,connected_dict,all_Table,is_print = False):
        # 构造图谱
        # 这里不同的波长节点需要连接上'Range节点'/或者官能团节点
        output_sample = self.GP.construct_graph(connected_dict,no_FG=all_Table)
        #[i for i in output_sample.successors('wavelength1')]
        output_sample = output_sample.copy()
        # 添加否认部分
        # 在'wavelength1'节点上添加'sample'节点
        output_sample.add_node('sample',raw_text='sample',color='brown')
        output_sample.add_edge('sample','wavelength1',relation='feature_edge.This sample has a sequence of spectra represented by "wavelength" nodes')
        # 根据all_Table中的信息，添加不存在的官能团节点
        output_sample.add_node('feature node. Functional group not exist: <{}>.'.format(','.join([i for i in all_Table])),raw_text = 'feature node. Functional group: <{}>.'.format(','.join([i for i in all_Table])),color = 'cyan')
        output_sample.add_edge('sample','feature node. Functional group not exist: <{}>.'.format(','.join([i for i in all_Table])),relation='feature_edge.Based on the sample spectra, there is a high probability that the following functional groups are not present in the molecular structure of the sample.')
        self.GP.merge_graph = output_sample

        if is_print:
            clean_FP = []
            for key,value in connected_dict.items():
                print(key)
                for k in value:
                    predecess_node = [i for i in output_sample.predecessors(k)]
                    for i in predecess_node:
                        if 'wave' not in i and i not in all_Table:
                            print(i)
                            clean_FP.append(i)
                print('-----------------')
            print('-----------------')
            print('Number calculation:')
            clean_FP_dict = {}
            for i in clean_FP:
                if i not in clean_FP_dict:
                    clean_FP_dict[i] = 1
                else:
                    clean_FP_dict[i]+=1
            # 根据clean_FP_dict对应不同key的value值，进行排序
            sorted_clean_FP = sorted(clean_FP_dict.items(),key=lambda x:x[1],reverse=True)
            print(sorted_clean_FP)

        return output_sample
    
    def subgraph_extraction(self,
                            hop=2,
                            is_sub=False,):
        # 提取子图
        subgraph = self.GP.extract_subgraph(hop=hop,
                                               is_sub=is_sub,)  # 以光谱图为核心区域
        
        return subgraph