from Data_process.base_process import GraphTextDataset
import re
import numpy as np
import networkx as nx

def extend_one_shot(subgraph,
                    graph,
                    without_point=None):
    """获取一跳子图。可能还会需要判断是否存在多条连线的情况。
    """
    huoqu=subgraph.copy()
    success_node_liat = []
    predecess_node_list = []
    for node in subgraph.nodes():
        # if without_point!=None:
        #     if without_point in node:
        #         continue
        for success_node in [i for i in graph.successors(node)]:
            if success_node not in subgraph.nodes():
                success_node_liat.append(success_node)
                huoqu.add_node(success_node,**graph.nodes()[success_node])
                huoqu.add_edge(node,success_node,**graph[node][success_node][0])
                # try:
                #     huoqu.add_edge(node,success_node,relation=graph[node,success_node][0]['relation'])
                # except:
                #     huoqu.add_edge(node,success_node)
        for predecess_node in [i for i in graph.predecessors(node)]:
            if predecess_node not in subgraph.nodes():
                predecess_node_list.append(predecess_node)
                huoqu.add_node(predecess_node,**graph.nodes()[predecess_node])
                huoqu.add_edge(predecess_node,node,**graph[predecess_node][node][0])
                # try:
                #     huoqu.add_edge(predecess_node,node,relation=graph[predecess_node,node][0]['relation'])
                # except:
                #     huoqu.add_edge(predecess_node,node)

        #分析：在graph中，success_node和predecess_node之间是否存在连线，如果存在，则在huoqu中添加连线
        for success_node in success_node_liat:
            for predecess_node in predecess_node_list:
                if graph.has_edge(success_node,predecess_node) and huoqu.has_edge(success_node,predecess_node)==False:
                    #print(success_node,predecess_node)
                    huoqu.add_edge(success_node,predecess_node,**graph[success_node][predecess_node][0])
        
        for success_node in success_node_liat:
            for succcess_node_2 in success_node_liat:
                if graph.has_edge(success_node,succcess_node_2) and huoqu.has_edge(success_node,succcess_node_2)==False:
                    #print(success_node,succcess_node_2)
                    huoqu.add_edge(success_node,succcess_node_2,**graph[success_node][succcess_node_2][0])
        
        for predecess_node in predecess_node_list:
            for succcess_node in success_node_liat:
                if graph.has_edge(predecess_node,succcess_node) and huoqu.has_edge(predecess_node,succcess_node)==False:
                    #print(predecess_node,succcess_node)
                    huoqu.add_edge(predecess_node,succcess_node,**graph[predecess_node][succcess_node][0])

        for predecess_node in predecess_node_list:
            for predecess_node_2 in predecess_node_list:
                if graph.has_edge(predecess_node,predecess_node_2) and huoqu.has_edge(predecess_node,predecess_node_2)==False:
                    #print(predecess_node,predecess_node_2)
                    huoqu.add_edge(predecess_node,predecess_node_2,**graph[predecess_node][predecess_node_2][0])
    
    return huoqu

class Merge_GraphText(GraphTextDataset):
    def __init__(self,
                    text_graph: GraphTextDataset,
                    spectrum_graph: GraphTextDataset,
                    ):
            # 继承父类中的loader属性
            self.text_graph = text_graph
            self.spectrum_graph = spectrum_graph
            self.connection_information = None
            self.merge_graph = None
            self.finnal_graph = None

    def get_information(self,
                        text_graph=None,
                        spectrum_graph=None,):
        # 这里只接收单个处理   
        if not text_graph:
             text_graph = self.text_graph
        if not spectrum_graph:
             spectrum_graph = self.spectrum_graph
        # 1.查询graph_function_group中所有官能团的Range节点
        range_node = []
        for name,information in text_graph.nodes(data=True):
            if 'Range' in name:
                range_node.append(name)
        # 2.检索光谱图像G中的strength节点,并查找与其相连的Wavelength节点
        wave = []
        for name,information in spectrum_graph.nodes(data=True):
            if 'strength' in name:
                #wave.append([n for n in spectrum_graph.neighbors(name)])
                wave.append([n for n in spectrum_graph.predecessors(name)])
        return range_node, wave
    
    def map_information(self, range_node, wave):
        # 查找光谱图像G中的波长节点可能对应的官能团种类,输出的格式为{波长节点:[官能团节点,]}
        connected_dict = {}
        for i in wave:  # 遍历对应了强度的波长节点
            spectrum_node = self.spectrum_graph.nodes[i[0]]
            start_wave =  eval(re.findall(r'<(.*?)>', spectrum_node['raw_text'])[0])
            end_wave = eval(re.findall(r'<(.*?)>', spectrum_node['raw_text'])[1])
            #print('--------------------------------')
            #print(i[0])
            connected_node = []
            for j in range_node: # 遍历官能团信息
                #print('--')
                # 根据分号进行划分
                function_node = self.text_graph.nodes[j]
                Peak = re.findall(r'<(.*?)>', function_node['raw_text'])[0].split(';')
                #print(Peak)
                cal = len(Peak)
                for k in Peak:
                    #print(k)
                    # 这里有两种情况,一种是单独的点,另一种是范围
                    if '-' in k: # 这是范围的
                        end = eval(k.split('-')[0])
                        start = eval(k.split('-')[1])
                        #discrete_point = np.linspace(start, end, 100)
                        if end_wave >= end and start_wave >= start:
                            #discrete_point = np.linspace(start_wave, end_wave, 100)  # 原来的
                            discrete_point = np.linspace(start, end, 100)  # 修改后
                            # 统计discrete_point中有多少个点落在start_wave和end_wave之间,输出这些点的总数
                            discrete_number = 0
                            for p in discrete_point:
                                #if p >= start_wave and p <= end_wave:
                                #if p >= start and p <= end:  # 原来的
                                if p >= start_wave and p <= end_wave:   # 修改后
                                    discrete_number += 1
                            #print('占比数为{}'.format(discrete_number/100))
                            if discrete_number/100 > 0.5:
                                #print(f'{i[0]} is in the {k} range')
                                #print('{}-{} is in the {}'.format(k,j,i))
                                ###connected_node.append(j)
                                cal -= 1
                        elif end >= end_wave and start >= start_wave:
                            #discrete_point = np.linspace(start_wave, end_wave, 100)   # 
                            discrete_point = np.linspace(start, end, 100)  # 修改后
                            # 统计discrete_point中有多少个点落在start_wave和end_wave之间,输出这些点的总数
                            discrete_number = 0
                            for p in discrete_point:
                                #if p >= start_wave and p <= end_wave:
                                #if p >= start and p <= end:  # 原来的
                                if p >= start_wave and p <= end_wave:   # 修改后
                                    discrete_number += 1
                            #print('占比数为{}'.format(discrete_number/100))
                            if discrete_number/100 > 0.5:
                                #print(f'{i[0]} is in the {k} range')
                                #print('{}-{} is in the {}'.format(k,j,i))
                                ###connected_node.append(j)
                                cal -= 1
                        elif end >= end_wave and start <= start_wave:
                             ###connected_node.append(j)
                             cal -= 1
                        elif end <= end_wave and start >= start_wave:
                             ###connected_node.append(j)
                             cal -= 1
                        else:
                            pass
                        #if start >= start_wave and end <= end_wave:
                        #    print(f'{i[0]} is in the {k} range')
                    else: # 这是单独的点
                        #print(k)
                        if eval(k) >= start_wave and eval(k) <= end_wave:
                            #print(f'{i[0]} is in the {k} range')
                            #print('{}-{} is in the {}'.format(k,j,i))
                            ###connected_node.append(j)
                            cal -= 1
                            continue
                if cal == 0: # 修改的部分，若官能团存在多个区间时，仅当区间同时满足时，才添加
                    connected_node.append(j)
            connected_dict[i[0]] = connected_node

        for key,value in connected_dict.items():
            # 删除value中的重复元素
            value = list(set(value))
            connected_dict[key] = value
        
        # 将connected_dict中的所有值，提取数字后修改为形如'Functional group{}'的形式
        #connected_dict = {key:['Functional group{}'.format(re.findall(r"\d+", value)[0]) for value in values] for key,values in connected_dict.items()}

        return connected_dict
    
    def find_no_exist_FG(self,
                         table,
                         is_print = False
                         ):
        total_dict = {}
        for key,value in table.items(): # 获取特征频率的负相关表
            start = value[1]  
            end = value[0]
            target_range=[]
            for node,attr in self.spectrum_graph.nodes(data=True):  # 获取负相关表中不同的官能团对应的波长节点
                if 'wavelength' in node:  # 获取样本光谱中不同的波长节点
                    pattern = re.compile(r'<(.*?)>')
                    result = pattern.findall(attr['raw_text'])
                    range_from_node = [eval(i) for i in result]
                    # 判断start和end是否落在range_from_node中
                    if range_from_node[1]>=start>=range_from_node[0]:
                        #print('start:',node)
                        target_range.append(node)
                    if range_from_node[1]>=end>=range_from_node[0]:
                        #print('end:',node)
                        target_range.append(node)
            #print('-----------------')
            # 如果target_range中的两个节点不连续，即末尾的数字相差不为1，则将其中的节点补充连续
            if len(target_range)>1 and len(target_range) != 0:
                target_range = sorted(target_range,key=lambda x:int(x.split('h')[-1]))
                start_num = int(target_range[0].split('h')[-1])
                end_num = int(target_range[-1].split('h')[-1])
                if end_num-start_num+1!=len(target_range):
                    for i in range(start_num,end_num+1):
                        if f'wavelength{i}' not in target_range:
                            target_range.append(f'wavelength{i}')
                target_range = sorted(target_range,key=lambda x:int(x.split('h')[-1]))
            if len(target_range) == 0:
                total_dict[key] = None
            else:
                total_dict[key] = target_range
        # 根据对应特定官能团的波长节点是否连接着强度节点，即是否有峰值，计算特定官能团的存在评分
        score_dict = {}
        for key,vlaue in total_dict.items():
            if vlaue == None:
                score_dict[key] = 0
            else:
                denominator = len(vlaue)
                numerator = 0
                for j in vlaue:
                    # 判断在SG中的节点j是否连接着‘Strength’节点
                    for neigh_node in [i for i in self.spectrum_graph.neighbors(j)]:
                        if 'strength' in neigh_node:
                            numerator+=1
                score = numerator/denominator
                if is_print:
                    print(key,':',score)
                score_dict[key] = score
        # 根据官能团的存在评分，判断该官能团是否存在
        isexist =[]
        for key,value in score_dict.items():
            if value>0.5:
                isexist.append(key.split(','))
            elif value != 0:
                for j in key.split(','):
                    for k in score_dict.keys():
                        if j in k and score_dict[k]>0.5:
                            isexist.append([j])
        #isexist
        # 将isexist拆解为1个列表，并去重
        exist_table = list(set([i for j in isexist for i in j]))
        # 将score_dict.keys()中的元素拆解为1个列表
        all_Table = list(set([i for j in score_dict.keys() for i in j.split(',')]))

        # 对于all_table,删除其中存在于isexist中的元素
        for i in exist_table:
            all_Table.remove(i)
        all_Table
        return all_Table,exist_table,score_dict
    def construct_graph(self,
                        connected_dict,
                        **kwargs
                        ):
        SG = self.spectrum_graph.copy()
        #connected_dict = {key:['Range{}'.format(re.findall(r"\d+", value)[0]) for value in values] for key,values in connected_dict.items()}
        # 1.查询
        target_function_group = {}
        for key,value in connected_dict.items():
            # 查询与波长节点相连的官能团分子节点
            TFG = []
            for i in value:
                #TFG.append([n for n in self.text_graph.neighbors(i)][0])
                TFG.append(i)  # 对应新的KM
            target_function_group[key] = TFG
        # 2.添加
        # Method1：将知识图谱中的节点添加进光谱图谱中
        # Method2：将self.spectrum中的节点添加进self.text_graph中
        new_graph_method_2 = nx.compose(self.text_graph, SG)

        for key,value in target_function_group.items():
            for i in value:
                # # Method1：将知识图谱中的节点添加进光谱图谱中
                # new_graph = nx.generators.ego_graph(self.text_graph, i, radius=1)
                # # 将new_graph中的节点添加到G_copy中
                # # 检测new_graph中的节点是否存在于G_copy中,如果不存在的话,添加该节点
                # for name,information in new_graph.nodes(data=True):
                #     if SG.has_node(name):
                #         pass
                #     else:
                #         SG.add_node(name,**information)
                # # 检测new_graph中的边是否存在于G_copy中,如果不存在的话,添加该条边线
                # for edge in new_graph.edges:
                #     if SG.has_edge(edge[0],edge[1]):
                #         pass
                #     else:
                #         SG.add_edge(edge[0],edge[1])
            
                # # 3.连线
                # # 检测key和i是否相连,如果不相连,则添加连线
                # if SG.has_edge(key,i):
                #     pass
                # else:
                #     SG.add_edge(key,i)
                
                # Method2：将self.spectrum中的节点添加进self.text_graph中
                # if new_graph_method_2.has_edge(key,i):
                #     pass
                # else:
                #     new_graph_method_2.add_edge(key,i,relation='feature edge.The absorption peaks present at this wavelength node may correspond to this characteristic interval')

                # Method3：将self.spectrum中的节点添加进self.text_graph中，此外，去除根据负相关表不存在的官能团
                is_add = True
                pre_node = [node for node in new_graph_method_2.predecessors(i)]
                for node in pre_node:
                    if node in kwargs['no_FG']:
                        is_add = False
                        break
                if is_add:
                    if new_graph_method_2.has_edge(key,i):
                        pass
                    else:
                        new_graph_method_2.add_edge(key,i,relation='feature edge.The absorption peaks present at this wavelength node may correspond to this characteristic interval')


        #self.merge_graph = SG
        self.merge_graph = new_graph_method_2
        return self.merge_graph
    
    def extract_subgraph(self, 
                         hop: int,
                         is_sub=False):
        """
        获取子图
        """
        s = nx.subgraph(self.merge_graph,self.spectrum_graph.nodes())
        sub_list = {}
        get_graph = s.copy()
        for i in range(hop):
            get_graph = extend_one_shot(get_graph,self.merge_graph)
            sub_list[i+1] = get_graph
            i += 1
        if is_sub:
            # 子图判断
            print(nx.isomorphism.DiGraphMatcher(self.merge_graph, get_graph).subgraph_is_isomorphic())
        
        self.finnal_graph = get_graph

        return self.finnal_graph    