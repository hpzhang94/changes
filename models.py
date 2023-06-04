import torch
import torch.nn as nn
import torch.nn.functional as F
from GCL.losses import JSD
from GCL.models import SingleBranchContrast
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import DataLoader
import numpy as np

from dataset import StepwiseSciPaperDataset
from layers import StepWiseGraphConvLayer, MLP
from transformers import BertTokenizer, BertModel
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
class BERT_Encoder(nn.Module):
    def __init__(self, bert_model='bert-base-uncased'):
        super(BERT_Encoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.model = BertModel.from_pretrained(bert_model)

    def forward(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True).to(device)
        # print(encoded_input)
        output = self.model(**encoded_input)
        return output["pooler_output"]

class Contrast_Encoder(nn.Module):
    def __init__(self, graph_encoder, text_encoder, hidden_dim, bert_hidden=768, mode="train", in_dim=768, dropout_p=0.3):
        super(Contrast_Encoder, self).__init__()
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.bert_hidden = bert_hidden
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)
        self.common_proj_mlp = MLP(in_dim, in_dim, 2048, dropout_p=dropout_p, act=nn.LeakyReLU())

    def forward(self, p_gfeature, p_adj, n_gfeature=None, n_adj=None, text=None, ntext=None):
        pg = self.graph_encoder(p_gfeature, p_adj)
        pg = self.common_proj_mlp(pg)
        if n_gfeature is not None and n_adj is not None:
            ng = self.graph_encoder(self.dropout(n_gfeature), n_adj)
            ng = self.common_proj_mlp(ng)
        else:
            ng = None
        if text is not None:
            t = self.text_encoder(text)
            t = self.common_proj_mlp(t)
        else:
            t = None

        if ntext is not None:
            nt = self.text_encoder(ntext)
            nt = self.common_proj_mlp(nt)
        else:
            nt = None

        return pg, ng, t, nt

    def set_mode(self, mode):
        self.mode = mode



class End2End_Encoder(nn.Module):
    def __init__(self, graph_encoder, in_dim, hidden_dim, dropout_p):
        super(End2End_Encoder, self).__init__()
        self.graph_encoder = graph_encoder
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(1 - dropout_p)
        self.out_proj_layer_mlp = MLP(in_dim, in_dim, 2048, act=nn.LeakyReLU(), dropout_p=dropout_p, layers=2)
        self.final_layer = nn.Linear(in_dim, 1)
        self.sort_module = Sort_Module(hidden_dim, dropout_p=dropout_p)

    def forward(self, x, x_c, adj, max_sen_num=None, t_xmask=None, t_adjmask=None):
        xf, x_mask, adj_mask, selected_idx, score = self.sort_module(x_c, adj, max_sen_num)
        if t_xmask is not None and t_adjmask is not None:
            x_mask = t_xmask
            adj_mask = t_adjmask
        x = x_c * x_mask
        adj *= adj_mask

        x = self.graph_encoder(x, adj)
        x = x[:, :-6, :]

        x = self.out_proj_layer_mlp(x)
        x = self.final_layer(x)
        return x, selected_idx, score, x_mask


class Contrast_Filter(nn.Module):
    def __init__(self, contrast_encoder, in_dim, hidden_dim, dropout_p):
        super().__init__()
        # self.input_proj_layer = nn.Linear(in_dim, hidden_dim)
        self.contrast_encoder = contrast_encoder


    def forward(self, x, adj, n_gfeature, n_adj,  text, ntext):
        pg, ng, t, nt = self.contrast_encoder(x, adj, n_gfeature=None, n_adj=None, text=text, ntext=None)
        return pg, ng, t, nt



class Sort_Module(nn.Module):
    def __init__(self, hid_dim, dropout_p):
        super(Sort_Module, self).__init__()

    def forward(self, x, adj, max_sen_num=None, landmark=None):
        x_mask = torch.ones_like(x)
        adj_mask = torch.ones_like(adj)

        if landmark is None:
            x_doc = x[:, -1, :]
            landmark = x_doc
        landmark = landmark.unsqueeze(1)
        # x_sen = x
        x_m = x
        x_m_norm = F.normalize(x_m, p=2, dim=-1)
        landmark = F.normalize(landmark, p=2, dim=-1)

        sen_doc_sim_score = torch.matmul(x_m_norm, landmark.permute(0, 2, 1)).squeeze(-1)
        sen_doc_sim_score = torch.where(sen_doc_sim_score > 0, sen_doc_sim_score.double(), 0.).float()

        sen_sen_sim_score = torch.mean(torch.matmul(x_m_norm, x_m_norm.permute(0, 2, 1)), dim=-1)
        sen_sen_sim_score = torch.where(0 < sen_sen_sim_score, sen_sen_sim_score.double(), 1.).float()

        score = sen_doc_sim_score * 0.9 + (1 - sen_sen_sim_score) * 0.1
        score = score[:, :-6]
        if max_sen_num is None:
          sen_doc_sim_score_argsorted = torch.argsort(score, dim=-1, descending=True)[:, 0: 180]
        else:
            sec_adj = adj[:, -6:, :-6]
            sen_doc_sim_score_argsorted = filter_by_section(score, sec_adj, keep_num=max_sen_num)

        for i in range(0, x.shape[0]):
            for j in range(0, x[i].shape[0] - 6):
                if j not in sen_doc_sim_score_argsorted[i]:
                    x_mask[i, j, :] = 0
                    adj_mask[i, j, :] = 0
                    adj_mask[i, :, j] = 0
                    # adj_mask[i, j, j] = 1
        # x_sen = x_m

        return x_m, x_mask, adj_mask, sen_doc_sim_score_argsorted, score.unsqueeze(-1)


def filter_by_section(score, sec_adj, keep_num=None):
    if keep_num is None:
        keep_num = [10, 30, 80, 30, 10]
    sen_num = torch.sum(sec_adj, dim=-1)
    # print(sen_num)
    masked_score = score * sec_adj
    # print(masked_score.shape)
    seleced_idx = []

    for i in range(0, score.shape[0]):
        for j in range(0, sec_adj.shape[1] - 1):
            keep_idx = keep_num[j]
            if sen_num[i, j] == 0:
                continue
            elif sen_num[i, j] < keep_num[j]:
                keep_idx = sen_num[i, j].int()
            # print(keep_idx)
            seleced_idx.append(torch.argsort(masked_score[i, j], dim=-1, descending=True)[:keep_idx].unsqueeze(0))
    return torch.concat(seleced_idx, dim=-1)





def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

class InfoNCE(nn.Module):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def forward(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()






