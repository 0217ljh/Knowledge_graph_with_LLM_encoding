import networkx as nx
from Data.base_process import GraphTextDataset

class Function_group_GraphText(GraphTextDataset):
    def construct_graph(self, object):
        """
        Construct a graph object
        """
        G = nx.Graph()
        # 给G添加两种节点，节点的名称为'Functional group{}'，{}为自然数列，从1开始，每个节点包含'raw_text'和'color'两个属性".
        for i in range(len(object)):
            G.add_node(f'Functional group{i+1}',raw_text = object.iloc[i]['Functional group'],color = 'cyan')
            G.add_node(f'Range{i+1}',raw_text = object.iloc[i]['Range'],color = 'purple')
            G.add_node(f'S-Strength{i+1}',raw_text = object.iloc[i]['S-Strength'],color = 'yellow')
        # 这些有很多相同的部分
        G.add_node(f'Absorb{1}',raw_text = object.iloc[0]['Absorb'],color = 'yellow')
        G.add_node(f'Type{1}',raw_text = object.iloc[0]['Type'],color = 'yellow')
        G.add_node(f'Subtype{1}',raw_text = object.iloc[0]['Subtype'],color = 'yellow')
        
        # 添加边
        for i in range(len(object)):
            if i == len(object):
                break
            Type = object.iloc[i]['Type']
            SubType = object.iloc[i]['Subtype']
            if G.has_edge('Type1', 'Subtype1'): # 大类连线
                pass
            else:G.add_edge('Type1', 'Subtype1')
            if G.has_edge('Subtype1', f'Functional group{i+1}'):
                pass
            else:G.add_edge('Subtype1', f'Functional group{i+1}')
            if G.has_edge(f'Functional group{i+1}', f'Range{i+1}'):
                pass
            else:G.add_edge(f'Functional group{i+1}', f'Range{i+1}')
            if G.has_edge(f'Functional group{i+1}', f'S-Strength{i+1}'):
                pass
            else:G.add_edge(f'Functional group{i+1}', f'S-Strength{i+1}')
            if G.has_edge(f'Functional group{i+1}', f'Absorb{1}'):
                pass
            else:G.add_edge(f'Functional group{i+1}', f'Absorb{1}')
        self.graph = G
        return G