from Models.nn.layer.pyg import RGCNEdgeConv,RGATEdgeConv
from Models.nn.models.GNN import MultiLayerMessagePassing
from Models.nn.models.util_model import MLP
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.transforms.add_positional_encoding import AddRandomWalkPE
from torch_geometric.nn.conv import GINConv

LLM_DIM_DICT = {"ST": 768, "BERT": 768, "e5": 1024, "llama2_7b": 4096, "llama2_13b": 5120,'qwen2.5':896}

class SingleHeadAtt(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sqrt_dim = torch.sqrt(torch.tensor(dim))
        self.Wk = torch.nn.Parameter(torch.zeros((dim, dim)))
        torch.nn.init.xavier_uniform_(self.Wk)
        self.Wq = torch.nn.Parameter(torch.zeros((dim, dim)))
        torch.nn.init.xavier_uniform_(self.Wq)

    def forward(self, key, query, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = torch.nn.functional.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn

class PyGRGCNEdge(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers: int,
        num_rels: int,
        inp_dim: int,
        out_dim: int,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.num_rels = num_rels
        self.build_layers()

    def build_input_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_hidden_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_message_from_input(self, g):
        return {
            "g": g.edge_index,
            "h": g.x,
            "e": g.edge_type,
            "he": g.edge_attr,
        }

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": h, "e": g.edge_type, "he": g.edge_attr}

    def layer_forward(self, layer, message):
        return self.conv[layer](
            message["h"], message["he"], message["g"], message["e"]
        )
    

class PyGRGATEdge(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers: int,
        num_rels: int,
        inp_dim: int,
        out_dim: int,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
        heads=8,
        add_self_loops=False,
        share_att=False,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.num_rels = num_rels
        self.heads = heads
        self.add_self_loops = add_self_loops
        self.share_att = share_att
        self.build_layers()

    def build_input_layer(self):
        return RGATEdgeConv(
            self.inp_dim,
            self.out_dim,
            self.num_rels,
            heads=self.heads,
            add_self_loops=self.add_self_loops,
            share_att=self.share_att,
        )

    def build_hidden_layer(self):
        return RGATEdgeConv(
            self.inp_dim,
            self.out_dim,
            self.num_rels,
            heads=self.heads,
            add_self_loops=self.add_self_loops,
            share_att=self.share_att,
        )

    def build_message_from_input(self, g):
        return {
            "g": g.edge_index,
            "h": g.x,
            "e": g.edge_type,
            "he": g.edge_attr,
        }

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": h, "e": g.edge_type, "he": g.edge_attr}

    def layer_forward(self, layer, message):
        return self.conv[layer](
            message["h"], message["he"], message["g"], message["e"]
        )

class PyGGIN(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers,
        inp_dim,
        out_dim,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.build_layers()

    def build_input_layer(self):
        return GINConv(
            MLP(
                [self.inp_dim, 2 * self.inp_dim, self.out_dim],
                batch_norm=self.batch_norm is not None,
            ),
            train_eps=True,
        )

    def build_hidden_layer(self):
        return GINConv(
            MLP(
                [self.out_dim, 2 * self.out_dim, self.out_dim],
                batch_norm=self.batch_norm is not None,
            ),
            train_eps=True,
        )

    def build_message_from_input(self, g):
        return {"g": g.edge_index, "h": g.x}

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": h}

    def layer_forward(self, layer, message):
        return self.conv[layer](message["h"], message["g"])


class BinGraphModel(torch.nn.Module):
    def __init__(self, model, llm_name, outdim, task_dim, add_rwpe=None, dropout=0.0, **kwargs):
        super().__init__()
        assert llm_name in LLM_DIM_DICT.keys()
        self.model = model
        self.llm_name = llm_name
        self.outdim = outdim
        self.llm_proj = nn.Linear(LLM_DIM_DICT[llm_name], outdim)
        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0)
        
        if add_rwpe is not None:
            self.rwpe = AddRandomWalkPE(add_rwpe)
            self.edge_rwpe_prior = torch.nn.Parameter(
                torch.zeros((1, add_rwpe))
            )
            torch.nn.init.xavier_uniform_(self.edge_rwpe_prior)
            self.rwpe_normalization = torch.nn.BatchNorm1d(add_rwpe)
            self.walk_length = add_rwpe
        else:
            self.rwpe = None

    def initial_projection(self, g):
        g.x = self.llm_proj(g.x)
        g.edge_attr = self.llm_proj(g.edge_attr)
        return g

    def forward(self, g):
        g = self.initial_projection(g)

        if self.rwpe is not None:
            with torch.no_grad():
                rwpe_norm = self.rwpe_normalization(g.rwpe)
                g.x = torch.cat([g.x, rwpe_norm], dim=-1)
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        emb = torch.stack(self.model(g), dim=1)
        #print('emb.shape:',emb.shape)

        attention_mask = g.true_nodes_mask.unsqueeze(-1).expand(-1, 7)
        #print('attention_mask:',attention_mask.shape)

        emb=mean_pooling(emb,attention_mask)
        #print('emb-squ:',emb.shape)

        class_emb = emb[g.true_nodes_mask]
        #print('class_emb:',class_emb.shape)

        res = self.mlp(class_emb)
        #print('res:',res.shape)

        return res

    def freeze_gnn_parameters(self):
        for p in self.model.parameters():
           p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False
        for p in self.llm_proj.parameters():
            p.requires_grad = False

class BinGraphAttModel(torch.nn.Module):
    """
    GNN model that use a single layer attention to pool final node representation across
    layers.
    """
    def __init__(self, model, llm_name, outdim, task_dim, add_rwpe=None, dropout=0.0, **kwargs):
        super().__init__()
        assert llm_name in LLM_DIM_DICT.keys()
        self.model = model
        self.llm_name = llm_name
        self.outdim = outdim
        self.llm_proj = nn.Linear(LLM_DIM_DICT[llm_name], outdim)
        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0)
        self.att = SingleHeadAtt(outdim)
        if add_rwpe is not None:
            self.rwpe = AddRandomWalkPE(add_rwpe)
            self.edge_rwpe_prior = torch.nn.Parameter(
                torch.zeros((1, add_rwpe))
            )
            torch.nn.init.xavier_uniform_(self.edge_rwpe_prior)
            self.rwpe_normalization = torch.nn.BatchNorm1d(add_rwpe)
            self.walk_length = add_rwpe
        else:
            self.rwpe = None

    def initial_projection(self, g):
        g.x = self.llm_proj(g.x)
        g.edge_attr = self.llm_proj(g.edge_attr)
        return g

    def forward(self, g):
        g = self.initial_projection(g)
        if self.rwpe is not None:
            with torch.no_grad():
                rwpe_norm = self.rwpe_normalization(g.rwpe)
                g.x = torch.cat([g.x, rwpe_norm], dim=-1)
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        emb = torch.stack(self.model(g), dim=1)
        query = g.x.unsqueeze(1)
        emb = self.att(emb, query, emb)[0].squeeze()
        class_emb = emb[g.true_nodes_mask]
        res = self.mlp(class_emb)
        return res

    def freeze_gnn_parameters(self):
        for p in self.model.parameters():
           p.requires_grad = False
        for p in self.att.parameters():
            p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False
        for p in self.llm_proj.parameters():
            p.requires_grad = False

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-10)