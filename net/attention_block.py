import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MutualAttention(nn.Module):
    def __init__(self, dim_feature, dim_hidden, n_layers, n_heads, dropout=0.1, bool_sa=True, **kwargs):
        super().__init__()
        self.cross_attn_x = nn.ModuleList([
            CrossAttentionLayer(dim_feature, dim_hidden, n_heads, dropout, bool_sa) for _ in range(n_layers)
        ])
        self.cross_attn_y = nn.ModuleList([
            CrossAttentionLayer(dim_feature, dim_hidden, n_heads, dropout, bool_sa) for _ in range(n_layers)
        ])

    def forward(self, x_ft, y_ft, return_att_map=False):
        att_map_x = {}
        att_map_y = {}
        for i, (layer_x, layer_y) in enumerate(zip(self.cross_attn_x, self.cross_attn_y)):
            x_ft, att_map_x[i] = layer_x(y_ft, x_ft)
            y_ft, att_map_y[i] = layer_y(x_ft, y_ft)

        if return_att_map:
            return x_ft, y_ft, att_map_x, att_map_y
        return x_ft, y_ft


class CrossAttention(nn.Module):
    def __init__(self, dim_feature, dim_hidden, n_layers, n_heads, dropout=0.1, bool_sa=True, bool_res=False):
        super().__init__()
        self.cross_attn = nn.ModuleList([
            CrossAttentionLayer(dim_feature, dim_hidden, n_heads, dropout, bool_sa, bool_res) for _ in range(n_layers)
        ])

    def forward(self, cross_ft, ft, return_att_map=False):
        att_map = {}
        for i, layer in enumerate(self.cross_attn):
            ft, att_map[i] = layer(cross_ft, ft)

        if return_att_map:
            return ft, att_map
        return ft


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim_feature, dim_hidden, n_heads, dropout, bool_sa=True, bool_res=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim_feature, n_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_feature)

        self.ffn = nn.Sequential(
            nn.Linear(dim_feature, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_feature),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim_feature)

        self.bool_sa = bool_sa
        self.bool_res = bool_res
        if bool_sa:
            self.multihead_attn_sa = nn.MultiheadAttention(dim_feature, n_heads, batch_first=True)
            self.dropout1_sa = nn.Dropout(dropout)
            self.norm1_sa = nn.LayerNorm(dim_feature)

    def forward(self, cross_ft, ft):
        if self.bool_sa:
            residual = ft
            ft, _ = self.multihead_attn_sa(query=ft, key=ft, value=ft)
            ft = self.dropout1_sa(ft)
            ft = self.norm1_sa(ft + residual)

        residual1 = ft
        ft, att_map = self.multihead_attn(query=ft,
                                          key=cross_ft,
                                          value=cross_ft)
        ft = self.dropout1(ft)
        ft = self.norm1(ft + residual1) if self.bool_res else self.norm1(ft)

        residual2 = ft
        ft = self.ffn(ft)
        ft = self.dropout2(ft)
        ft = self.norm2(ft + residual2)

        return ft, att_map


class SelfAttention(nn.Module):
    def __init__(self, dim_feature, dim_hidden, n_layers, n_heads, dropout=0.1, **kwargs):
        super().__init__()
        self.cross_attn = nn.ModuleList([
            SelfAttentionLayer(dim_feature, dim_hidden, n_heads, dropout) for _ in range(n_layers)
        ])

    def forward(self, ft, return_att_map=False):
        # ft: N, L, C
        att_map = {}
        for i, layer in enumerate(self.cross_attn):
            ft, att_map[i] = layer(ft)

        if return_att_map:
            return ft, att_map
        return ft


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim_feature, dim_hidden, n_heads, dropout, **kwargs):
        super().__init__()
        self.multihead_attn_sa = nn.MultiheadAttention(dim_feature, n_heads, batch_first=True)
        self.dropout1_sa = nn.Dropout(dropout)
        self.norm1_sa = nn.LayerNorm(dim_feature)
        self.ffn_sa = nn.Sequential(
            nn.Linear(dim_feature, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_feature),
        )
        self.dropout2_sa = nn.Dropout(dropout)
        self.norm2_sa = nn.LayerNorm(dim_feature)

    def forward(self, ft):
        residual = ft
        ft, att_map = self.multihead_attn_sa(query=ft, key=ft, value=ft)
        ft = self.dropout1_sa(ft)
        ft = self.norm1_sa(ft + residual)
        residual = ft
        ft = self.ffn_sa(ft)
        ft = self.dropout2_sa(ft)
        ft = self.norm2_sa(ft + residual)

        return ft, att_map


class GateCrossAttention(nn.Module):
    def __init__(self, dim_feature, dim_hidden, n_layers, n_heads, dropout=0.1, bool_sa=True, **kwargs):
        super().__init__()
        self.cross_attn = nn.ModuleList([
            GateCrossAttentionLayer(dim_feature, dim_hidden, n_heads, dropout, bool_sa) for _ in range(n_layers)
        ])

    def forward(self, cross_ft, ft, return_att_map=False):
        att_map = {}
        for i, layer in enumerate(self.cross_attn):
            ft, att_map[i] = layer(cross_ft, ft)

        if return_att_map:
            return ft, att_map
        return ft


class GateCrossAttentionLayer(nn.Module):
    def __init__(self, dim_feature, dim_hidden, n_heads, dropout, bool_sa=True, **kwargs):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim_feature, n_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_feature)

        self.ffn = nn.Sequential(
            nn.Linear(dim_feature, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_feature),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim_feature)

        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(dim_feature)

        self.bool_sa = bool_sa
        if bool_sa:
            self.multihead_attn_sa = nn.MultiheadAttention(dim_feature, n_heads, batch_first=True)
            self.dropout1_sa = nn.Dropout(dropout)
            self.norm1_sa = nn.LayerNorm(dim_feature)

    def forward(self, cross_ft, ft):
        if self.bool_sa:
            residual = ft
            ft, _ = self.multihead_attn_sa(query=ft, key=ft, value=ft)
            ft = self.dropout1_sa(ft)
            ft = self.norm1_sa(ft + residual)

        original_ft = ft.clone()
        residual = ft
        ft, att_map = self.multihead_attn(query=ft,
                                          key=cross_ft,
                                          value=cross_ft)
        ft = self.dropout1(ft)
        ft = self.norm1(ft + residual)

        residual = ft
        ft = self.ffn(ft)
        ft = self.dropout2(ft)
        ft = self.norm2(ft + residual)

        residual = ft
        weights = torch.einsum('blc,blc->bl', ft, original_ft)  # N, L
        weights = torch.softmax(weights, dim=-1)
        weights = weights.unsqueeze(-1)  # N, L, 1
        ft = ft * weights  # N, L, C
        ft = self.dropout3(ft)
        ft = self.norm3(ft + residual)

        return ft, att_map


def cosine_position_embeddings(max_len, dim_model):
    pe = torch.zeros(max_len, dim_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


class GLAttention(nn.Module):
    def __init__(self, method, dim_feature):
        super().__init__()
        self.__method = method
        self.__dim_feature = dim_feature
        if self.__method == 'dp':
            pass
        elif self.__method == 'pc':
            self.__pc = nn.Linear(dim_feature, 1, bias=False)
        else:
            raise ('No such method', self.__method)

    def forward(self, ft, return_attention_map=False):
        # ft  (bs, l, c)
        # print('lcf.shape', lcf.shape)
        gbf_pj = ft.clone().mean(1)     # (bs, c)
        lcf_rs = ft  # (bs, l, c)
        if self.__method == 'dp':
            c = torch.matmul(lcf_rs, gbf_pj.unsqueeze(-1)).squeeze(-1)
        elif self.__method == 'pc':
            # print(lcf_rs.shape, gbf_pj.shape)
            add = torch.add(lcf_rs, gbf_pj.unsqueeze(-2))
            c = self.__pc(add).squeeze(-1)
        else:
            raise ('No such method', self.__method)
        a = torch.softmax(c, 1)     # (bs, h*w)
        ga = torch.matmul(lcf_rs.transpose(1, 2), a.unsqueeze(-1)).squeeze(-1)
        if return_attention_map:
            return ga, a
        return ga


class QueriesEmbedding(nn.Module):
    def __init__(self, n_queries, d_queries_in, d_queries_out, n_layers=1):
        super().__init__()
        # inputs (N, n_queries, d_queries_in) weight(n_queries, d_queries_out)
        self.n_layers = n_layers
        self.weight = nn.ParameterList()
        self.bias = nn.ParameterList()
        for i in range(n_layers):
            self.weight.append(nn.Parameter(torch.Tensor(n_queries, d_queries_in, d_queries_out)))
            self.bias.append(nn.Parameter(torch.Tensor(n_queries, d_queries_out)))
            self.weight[i].data.normal_(mean=0.0, std=0.01)
            self.bias[i].data.normal_(mean=0.0, std=0.01)

    def forward(self, x):
        # x: (N, n_queries, d_queries_in)
        # output: (N, n_queries, d_queries_out)
        for i in range(self.n_layers):
            x = torch.einsum('nqi,qio->nqo', x, self.weight[i]) + self.bias[i]
            x = F.relu(x)
        return x