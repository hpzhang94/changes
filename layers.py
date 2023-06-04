import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from labml_helpers.module import Module


from dataset import StepwiseSciPaperDataset

def normalize_adj(adj, beta=0.3):
    max_v = torch.max(torch.max(adj, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
    min_v = torch.min(torch.min(adj, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
    # print(max_v.shape)
    adj = adj - (min_v + beta * (max_v - min_v))
    adj = torch.maximum(adj, torch.tensor(0))
    # print(adj[0])
    return adj


def get_sen_sec_mask(sen_num, sen_sec_mask):
    sen_sec_mask_exped = []

    for i in range(0, sen_sec_mask.shape[1]):
        mask = sen_sec_mask[:, i, :].unsqueeze(-1)
        sen_sec_mask_exped.append(mask.expand(sen_sec_mask.shape[0], sen_num, sen_num))
    sen_sec_mask_expended = torch.stack([mask.permute(0, 2, 1) for mask in sen_sec_mask_exped],
                                        dim=1)  # (batch_size, sec_num, sen_num, sen_num)
    return sen_sec_mask_expended


# borrowed from labml.ai
class GraphAttentionLayer(Module):
    """
    ## Graph attention layer

    This is a single graph attention layer.
    A GAT is made up of multiple such layers.

    It takes
    $$\mathbf{h} = \{ \overrightarrow{h_1}, \overrightarrow{h_2}, \dots, \overrightarrow{h_N} \}$$,
    where $\overrightarrow{h_i} \in \mathbb{R}^F$ as input
    and outputs
    $$\mathbf{h'} = \{ \overrightarrow{h'_1}, \overrightarrow{h'_2}, \dots, \overrightarrow{h'_N} \}$$,
    where $\overrightarrow{h'_i} \in \mathbb{R}^{F'}$.
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial transformation;
        # i.e. to transform the node embeddings before self-attention
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.

        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """

        # Number of nodes
        n_nodes = h.shape[0]

        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)

        g_repeat = g.repeat(n_nodes, 1, 1)

        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)

        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)

        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)

        e = self.activation(self.attn(g_concat))
        # del g_concat, g_repeat, g_repeat_interleave
        # torch.cuda.empty_cache()
        # Remove the last dimension of size `1`
        e = e.squeeze(-1)

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        # Mask $e_{ij}$ based on adjacency matrix.
        # $e_{ij}$ is set to $- \infty$ if there is no edge from $i$ to $j$.
        e = e.masked_fill(adj_mat == 0, float(-1e9))

        a = self.softmax(e)

        a = self.dropout(a)

        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        # Concatenate the heads
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            return attn_res.mean(dim=1)


class GAT(Module):
    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):
        super().__init__()
        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)
        self.activation = nn.ELU()
        self.output = GraphAttentionLayer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        x = x.squeeze(0)
        adj_mat = adj_mat.squeeze(0)
        adj_mat = adj_mat.unsqueeze(-1).bool()
        x = self.dropout(x)
        x = self.layer1(x, adj_mat)
        x = self.activation(x)
        x = self.dropout(x)
        return self.output(x, adj_mat).unsqueeze(0)



class Attention(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(Attention, self).__init__()
        self.W_Q = nn.Parameter(torch.zeros(size=(in_dim, hid_dim)))
        self.W_K = nn.Parameter(torch.zeros(size=(in_dim, hid_dim)))
        nn.init.xavier_uniform_(self.W_Q.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_K.data, gain=1.414)


    def forward(self, Q, K, mask=None):
        KW_K = torch.matmul(K, self.W_K)
        QW_Q = torch.matmul(Q, self.W_Q)
        if len(KW_K.shape) == 3:
            KW_K = KW_K.permute(0, 2, 1)
        elif len(KW_K.shape) == 4:
            KW_K = KW_K.permute(0, 1, 3, 2)
        att_w = torch.matmul(QW_Q, KW_K).squeeze(1)
        if mask is not None:
            att_w = torch.where(mask == 1, att_w.double(), float(-1e10))

        att_w = F.softmax(att_w / torch.sqrt(torch.tensor(Q.shape[-1])), dim=-1)
        return att_w

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, layers=2, act=nn.LeakyReLU(), dropout_p=0.3, keep_last_layer=False):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.layers = layers
        self.act = act
        self.dropout = nn.Dropout(1 - dropout_p)
        self.keep_last = keep_last_layer

        self.mlp_layers = nn.ModuleList([])
        if layers == 1:
            self.mlp_layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            self.mlp_layers.append(nn.Linear(self.in_dim, self.hid_dim))
            for i in range(self.layers - 2):
                self.mlp_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            self.mlp_layers.append(nn.Linear(self.hid_dim, self.out_dim))

    def forward(self, x):
        for i in range(len(self.mlp_layers) - 1):
            x = self.dropout(self.act(self.mlp_layers[i](x)))
        if self.keep_last:
            x = self.mlp_layers[-1](x)
        else:
            x = self.act(self.mlp_layers[-1](x))
        return x



class StepWiseGraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, type_num, hid_dim=1024, dropout_p=0.3, act=nn.LeakyReLU(), fusion=False, nheads=6, graph=True, iter=1, final="att"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.type_num = type_num
        self.dropout_p = 1 - dropout_p
        self.act = act
        self.dropout = nn.Dropout(self.dropout_p)
        self.fusion = fusion
        self.graph = graph
        self.iter = iter

        self.gat = nn.ModuleList([GAT(in_features=self.in_dim, n_hidden=self.hid_dim, n_classes=self.in_dim,
                               dropout=self.dropout_p,  n_heads=nheads) for _ in range(iter)])

        self.gat2 = nn.ModuleList([GAT(in_features=self.in_dim, n_hidden=self.hid_dim, n_classes=self.in_dim,
                                      dropout=self.dropout_p, n_heads=nheads) for _ in range(iter)])


        self.feature_fusion_layer = nn.Linear(self.in_dim * 2, self.in_dim)
        self.feature_fusion_layer2 = nn.Linear(self.in_dim * 3, self.in_dim)

        self.ffn = MLP(self.in_dim, self.in_dim, 2048, dropout_p=dropout_p, layers=3)
        self.out_ffn = MLP(self.in_dim, self.in_dim, 2048, dropout_p=dropout_p)



    def forward(self, feature, adj):
        sen_adj = adj.clone()
        sen_adj[:, :, -6:] = 0
        sec_adj = adj.clone()
        sec_adj[:, :, :-6] = 0

        feature_sen = feature.clone()
        feature_sec = feature.clone()
        feature_resi = feature
        if self.graph:
            feature_sen_re = feature_sen
            feature_sec_re = feature_sec
            for i in range(0, self.iter):
                feature_sen = self.gat[i](feature_sen, sen_adj)
            feature_sen += feature_sen_re

            for i in range(0, self.iter):
                feature_sec= self.gat2[i](feature_sec, sec_adj)
            feature_sec += feature_sec_re

            feature = torch.concat([feature_sec, feature_sen], dim=-1)

            feature = self.dropout(F.leaky_relu(self.feature_fusion_layer(feature)))

            feature = self.ffn(feature)

        if self.fusion:
            sec_adj = adj[:, :, -self.type_num:-1].clone() #(batch_size, sen_num, sec_num)
            sec_feature = feature[:, -self.type_num:-1, :]
            sec_feature_exp = torch.matmul(sec_adj, sec_feature) #(batch_size, sen_num, feature_dim)
            doc_feature = feature[:, -1, :].unsqueeze(1).expand(sec_feature_exp.shape[0], sec_feature_exp.shape[1], -1)
            feature = torch.concat([feature, sec_feature_exp, doc_feature], dim=-1)
            feature = F.leaky_relu(self.feature_fusion_layer2(feature))

        feature = self.out_ffn(feature) + feature_resi
        return feature

