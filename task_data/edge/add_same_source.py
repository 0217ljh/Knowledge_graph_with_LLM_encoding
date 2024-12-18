import re
import tqdm
from Data.data_collect.Merge_collect import Merge_PygDataset

def get_related_fp_from_strength(node_list,sub_graph):
        strength_dict = {}
        for i in node_list:
            index = re.sub(u"([^\u0030-\u0039])","",i[0])
            link_fp = [i for i in sub_graph.successors(f'wavelength{index}') if 'Range' in i]
            interval = []
            for j in link_fp:
                interval.append([k for k in sub_graph.predecessors(j) if 'wave' not in k])
            interval = [k[0] for k in interval]
            strength_dict[i[0]] = interval
        return strength_dict
def add_same_source(Spectrum_graph,FG_graph):
        for name, spe_graph in tqdm.tqdm(Spectrum_graph.items()):
            test_class = Merge_PygDataset(FG_graph, spe_graph, Name=name)
            connected_dictionary = test_class.preprocess()      # 官能团信息：正相关
            all_Table, isexist, score_dict = test_class.Negative_relevant_information(is_print=False)   # 官能团信息：负相关
            OP = test_class.construct_graph(connected_dictionary, all_Table, is_print=False)        # 生成混合图谱
            fin = test_class.subgraph_extraction(hop=2)    # 生成邻近子图

            # generate edge sub_graph
            # Step1: get strength node
            pair_dict = {}
            molecule_name = name
            strength_node = [(node_name, attr) for node_name, attr in Spectrum_graph[molecule_name]['graph'].nodes(data=True) if 'strength' in node_name]

            for i in range(len(strength_node)):
                for j in range(i + 1, len(strength_node)):
                    pair_dict[(strength_node[i][0], strength_node[j][0])] = (strength_node[i][1], strength_node[j][1])

            # Step2: 生成样本配置
            sample_list = []
            for index, key in enumerate(pair_dict):
                sample_config = {'Molecule': molecule_name, 'Combination_Index': index, 'Peak': [key[0], key[1]], 'Functional_Group': [], 'Label': [], 'node_index': []}
                sample_list.append(sample_config)

            # Step3: 获取波峰节点的功能组信息
            strength_dict = get_related_fp_from_strength(strength_node, OP)

            # Step4: 比较波峰节点的功能组，并为同源节点之间添加边
            # 添加边的逻辑
            for i in sample_list:
                for sub_peak in i['Peak']:
                    i['Functional_Group'].append(strength_dict[sub_peak])
                # 比较两个功能组是否有交集
                if len(set(i['Functional_Group'][0]) & set(i['Functional_Group'][1])) > 0:

                    # 同源节点之间添加边
                    node1, node2 = i['Peak']
                    graph = Spectrum_graph[molecule_name]['graph']  # 获取图
                    
                    # 在图中为同源波峰节点之间添加边
                    # 假设你希望添加边属性，例如'weight'为1（可以根据需求调整）
                    if not graph.has_edge(node1, node2):  # 防止重复添加边
                        graph.add_edge(node1, node2, relation='feature edge.This two strength node of spectral peak come from the same source.', weight=1)  # 'same_origin'标签表示同源
        print('Add same source edge successfully!')
        return Spectrum_graph
