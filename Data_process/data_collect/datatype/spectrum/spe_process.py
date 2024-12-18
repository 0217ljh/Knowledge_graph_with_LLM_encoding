import re
import networkx as nx
from Data_process.base_process import GraphTextDataset

class Spectrum_GraphText(GraphTextDataset):
    def construct_graph(self, object,name):
        """
        Construct a graph object
        """
        #G = nx.Graph()
        # 创建有向图G
        G = nx.MultiDiGraph()
        # 添加节点
        # 给G添加节点，节点的名称为'wavelength{}'，{}为自然数列，从1开始，每个节点包含'raw_text'和'color'两个属性".
        for i in range(len(object)):
            G.add_node(f'wavelength{i+1}',raw_text = object.iloc[i]['wavelength'],color = 'green')
            G.add_node(f'strength{i+1}',raw_text = object.iloc[i]['strength'],color = 'blue')
            if object.iloc[i]['min_index'] != False:
                G.add_node(f'min_index{i+1}',raw_text = object.iloc[i]['min_index'],color = 'red')

        # 添加边
        # 检索所有波长节点，如果相邻的波长节点之间没有边，则添加边
        Feature = nx.get_node_attributes(G,'raw_text')
        for i in range(len(object)):
            if i == len(object)-1:
                pass
            else:
                G.add_edge(f'wavelength{i+1}',f'wavelength{i+2}')

            if eval(re.findall(r'<(.*?)>',Feature[f'strength{i+1}'])[0]) > 1 :
                G.add_edge(f'wavelength{i+1}',f'strength{i+1}')
            # 检测是否G中是否含有对应的min_index节点，如果有，则添加边
            if G.has_node(f'min_index{i+1}'):
                G.add_edge(f'wavelength{i+1}',f'min_index{i+1}')
                #if Feature[f'strength{i+1}'] == '1':
                #        G.add_edge(f'wavelength{i+1}',f'strength{i+1}')
                if not G.has_edge(f'wavelength{i+1}',f'strength{i+1}'):
                    G.add_edge(f'wavelength{i+1}',f'strength{i+1}')
        # 删除G中没有连线的点
        G.remove_nodes_from(list(nx.isolates(G)))
        self.graph = G
        self.graph_dataset[name] = G
        return G

# import re
# import networkx as nx
# from Data.base_process import GraphTextDataset

# class Spectrum_GraphText(GraphTextDataset):
#     def construct_graph(self, object,name):
#         """
#         Construct a graph object
#         """
#         #G = nx.Graph()
#         # 创建有向图G
#         G = nx.MultiDiGraph()
#         # 添加节点
#         # 给G添加节点，节点的名称为'wavelength{}'，{}为自然数列，从1开始，每个节点包含'raw_text'和'color'两个属性".
#         for i in range(len(object)):
#             G.add_node(f'wavelength{i+1}',raw_text = object.iloc[i]['wavelength'],color = 'green')
#             G.add_node(f'strength{i+1}',raw_text = object.iloc[i]['strength'],color = 'blue')
#             # if object.iloc[i]['min_index'] != False:
#             #     G.add_node(f'min_index{i+1}',raw_text = object.iloc[i]['min_index'],color = 'red')

#         # 添加边
#         # 检索所有波长节点，如果相邻的波长节点之间没有边，则添加边
#         Feature = nx.get_node_attributes(G,'raw_text')
#         for i in range(len(object)):
#             if i == len(object)-1:
#                 pass
#             else:
#                 G.add_edge(f'wavelength{i+1}',f'wavelength{i+2}')

#             if eval(re.findall(r'<(.*?)>',Feature[f'strength{i+1}'])[0]) > 1 :
#                 G.add_edge(f'wavelength{i+1}',f'strength{i+1}')
#             # 检测是否G中是否含有对应的min_index节点，如果有，则添加边
#             # if G.has_node(f'min_index{i+1}'):
#             #     G.add_edge(f'wavelength{i+1}',f'min_index{i+1}')
#                 #if Feature[f'strength{i+1}'] == '1':
#                 #        G.add_edge(f'wavelength{i+1}',f'strength{i+1}')
#             if not G.has_edge(f'wavelength{i+1}',f'strength{i+1}'):
#                 G.add_edge(f'wavelength{i+1}',f'strength{i+1}')
#         # 删除G中没有连线的点
#         G.remove_nodes_from(list(nx.isolates(G)))
#         self.graph = G
#         self.graph_dataset[name] = G
#         return G