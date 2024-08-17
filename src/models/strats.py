from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.abstract import TimeSeriesModel


@dataclass(frozen=True)
class StratsConfig:
    demographics_num: int
    features_num: int
    hid_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    attention_dropout: float
    head: Literal['forecast', 'binary', 'forecast_binary']
    
    def __post_init__(self):
        assert self.demographics_num > 0, 'Demographics number must be positive'
        assert self.features_num > 0, 'Features number must be positive'
        assert self.hid_dim > 0, 'Hidden dimension must be positive'
        assert self.num_layers > 0, 'Number of layers must be positive'
        assert self.num_heads > 0, 'Number of heads must be positive'
        assert 0 <= self.dropout < 1, 'Dropout must be in [0, 1)'
        assert 0 <= self.attention_dropout < 1, 'Attention dropout must be in [0, 1)'


class CVE(nn.Module):
    def __init__(self, hid_dim: int):
        super().__init__()
        int_dim = int(np.sqrt(hid_dim))
        self.W1 = nn.Parameter(torch.empty(1, int_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.W2 = nn.Parameter(torch.empty(int_dim, hid_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh
    
    def forward(self, x):
        # x: bsz, max_len
        x = torch.unsqueeze(x, -1)
        x = torch.matmul(x, self.W1) + self.b1[None, None, :]  # bsz,max_len,int_dim
        x = self.activation(x)
        x = torch.matmul(x, self.W2)  # bsz,max_len,hid_dim
        return x


class FusionAtt(nn.Module):
    def __init__(self, hid_dim: int):
        super().__init__()
        int_dim = hid_dim
        self.W = nn.Parameter(torch.empty(hid_dim, int_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.u = nn.Parameter(torch.empty(int_dim, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)
        self.activation = torch.tanh
    
    def forward(self, x, mask):
        # x: bsz, max_len, hid_dim
        att = torch.matmul(x, self.W) + self.b[None, None, :]  # bsz,max_len,int_dim
        att = self.activation(att)
        att = torch.matmul(att, self.u)[:, :, 0]  # bsz,max_len
        att = att + (~mask) * torch.finfo(x.dtype).min
        att = torch.softmax(att, dim=-1)  # bsz,max_len
        return att


# TODO: replace this with existing transformer implementation
class Transformer(nn.Module):
    def __init__(self, config: StratsConfig):
        super().__init__()
        self.N = config.num_layers
        self.d = config.hid_dim
        self.dff = self.d * 2
        self.attention_dropout = config.attention_dropout
        self.dropout = config.dropout
        self.h = config.num_heads
        self.dk = self.d // self.h
        self.all_head_size = self.dk * self.h
        
        self.Wq = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)),
                               requires_grad=True)
        self.Wk = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)),
                               requires_grad=True)
        self.Wv = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)),
                               requires_grad=True)
        self.Wo = nn.Parameter(self.init_proj((self.N, self.all_head_size, self.d)),
                               requires_grad=True)
        self.W1 = nn.Parameter(self.init_proj((self.N, self.d, self.dff)), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros((self.N, 1, 1, self.dff)), requires_grad=True)
        self.W2 = nn.Parameter(self.init_proj((self.N, self.dff, self.d)), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros((self.N, 1, 1, self.d)), requires_grad=True)
        
        self.norm1 = nn.LayerNorm(config.hid_dim, eps= 1e-5, bias=True)
        self.norm2 = nn.LayerNorm(config.hid_dim, eps= 1e-5, bias=True)
    
    def init_proj(self, shape, gain=1):
        x = torch.rand(shape)
        fan_in_out = shape[-1] + shape[-2]
        scale = gain * np.sqrt(6 / fan_in_out)
        x = x * 2 * scale - scale
        return x
    
    def forward(self, x, mask: torch.Tensor):
        # x: bsz, max_len, d
        # mask: bsz, max_len
        bsz, max_len, _ = x.size()
        mask = mask[:, :, None] * mask[:, None, :]
        mask = ~mask[:, None, :, :] * torch.finfo(x.dtype).min
        
        layer_mask = mask
        for i in range(self.N):
            # MHA
            q = torch.einsum('bld,hde->bhle', x, self.Wq[i])  # bsz,h,max_len,dk
            k = torch.einsum('bld,hde->bhle', x, self.Wk[i])  # bsz,h,max_len,dk
            v = torch.einsum('bld,hde->bhle', x, self.Wv[i])  # bsz,h,max_len,dk
            A = torch.einsum('bhle,bhke->bhlk', q, k)  # bsz,h,max_len,max_len
            if self.training:
                dropout_mask = (torch.rand_like(A) < self.attention_dropout).float() * torch.finfo(x.dtype).min
                layer_mask = mask + dropout_mask
            A = A + layer_mask
            A = torch.softmax(A, dim=-1)
            v = torch.einsum('bhkl,bhle->bkhe', A, v)  # bsz,max_len,h,dk
            all_head_op = v.reshape((bsz, max_len, -1))
            all_head_op = torch.matmul(all_head_op, self.Wo[i])
            all_head_op = F.dropout(all_head_op, self.dropout, self.training, inplace=True)
            # Add+layernorm
            x = self.norm1(all_head_op + x)
            # FFN
            ffn_op = torch.matmul(x, self.W1[i]) + self.b1[i]
            ffn_op = F.gelu(ffn_op)
            ffn_op = torch.matmul(ffn_op, self.W2[i]) + self.b2[i]
            ffn_op = F.dropout(ffn_op, self.dropout, self.training, inplace=True)
            # Add+layernorm
            x = self.norm2(ffn_op + x)
        return x


class NoOp(nn.Module):
    def forward(self, *args, **kwargs):
        return None


class Strats(TimeSeriesModel):
    def __init__(self, config: StratsConfig):
        super().__init__()
        
        self.cve_time = CVE(config.hid_dim)
        self.cve_value = CVE(config.hid_dim)
        self.variable_emb = nn.Embedding(config.features_num, config.hid_dim)
        self.transformer = Transformer(config)
        self.fusion_att = FusionAtt(config.hid_dim)
        self.dropout = config.dropout
        self.features_num = config.features_num
        
        self.head_type = config.head
        self.demo_emb = nn.Sequential(
            nn.Linear(config.demographics_num, config.hid_dim * 2),
            nn.Tanh(),
            nn.Linear(config.hid_dim * 2, config.hid_dim)
        )
        ts_demo_emb_size = config.hid_dim * 2
        
        if self.head_type == 'forecast':
            self.forecast_fc = nn.Linear(ts_demo_emb_size, config.features_num)
            self.binary_head = nn.Identity()
        elif self.head_type == 'binary':
            self.binary_head = nn.Sequential(
                nn.Linear(ts_demo_emb_size, 1),
                nn.Sigmoid()
            )
            self.forecast_fc = nn.Identity()
        elif self.head_type == 'forecast_binary':
            self.forecast_fc = nn.Linear(ts_demo_emb_size, config.features_num)
            self.binary_head = nn.Sequential(
                nn.Linear(config.features_num, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f'Invalid head type: {config.head}')
    
    def forward(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        variables: torch.Tensor,
        input_mask: torch.Tensor,
        demographics: torch.Tensor,
    ):
        # demographics embedding
        demo_emb = self.demo_emb(demographics)
        
        # initial triplet embedding
        time_emb = self.cve_time(times)
        value_emb = self.cve_value(values)
        variable_emb = self.variable_emb(variables)
        triplet_emb = time_emb + value_emb + variable_emb
        # TODO: does inplace dropout improve performance?
        triplet_emb = F.dropout(triplet_emb, self.dropout, self.training, inplace=True)
        # contextual triplet emb
        contextual_emb = self.transformer(triplet_emb, input_mask)
        
        # fusion attention
        attention_weights = self.fusion_att(contextual_emb, input_mask)[:, :, None]
        ts_emb = (contextual_emb * attention_weights).sum(dim=1)
        
        # concat demographics and ts_emb
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        
        if self.head_type == 'forecast':
            return self.forecast_fc(ts_demo_emb)
        elif self.head_type == 'binary':
            return self.binary_head(ts_demo_emb)
        elif self.head_type == 'forecast_binary':
            ts_demo_emb = self.forecast_fc(ts_demo_emb)
            return self.binary_head(ts_demo_emb)
