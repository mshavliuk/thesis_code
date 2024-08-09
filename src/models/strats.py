from argparse import Namespace
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
        att = att + (~mask) * torch.finfo(att.dtype).min
        att = torch.softmax(att, dim=-1)  # bsz,max_len
        return att


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
        # self.layer_norm1 = nn.ModuleList([nn.LayerNorm(self.d) for i in range(self.N)])
        # self.layer_norm2 = nn.ModuleList([nn.LayerNorm(self.d) for i in range(self.N)])
    
    def init_proj(self, shape, gain=1):
        x = torch.rand(shape)
        fan_in_out = shape[-1] + shape[-2]
        scale = gain * np.sqrt(6 / fan_in_out)
        x = x * 2 * scale - scale
        return x
    
    def forward(self, x, mask):
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
            # x_for_qkv = x[:,None,:,:,None]
            # q = (x_for_qkv*self.Wq[i][None,:,None,:,:]).sum(dim=-2) # bsz,h,max_len,dk
            # k = (x_for_qkv*self.Wk[i][None,:,None,:,:]).sum(dim=-2) # bsz,h,max_len,dk
            # v = (x_for_qkv*self.Wv[i][None,:,None,:,:]).sum(dim=-2) # bsz,h,max_len,dk
            A = torch.einsum('bhle,bhke->bhlk', q, k)  # bsz,h,max_len,max_len
            # A = (q[:,:,:,None,:]*k[:,:,None,:,:]).sum(dim=-1) # bsz,h,max_len,max_len
            if self.training:
                dropout_mask = (torch.rand_like(A) < self.attention_dropout
                                ).float() * torch.finfo(x.dtype).min
                layer_mask = mask + dropout_mask
            A = A + layer_mask
            A = torch.softmax(A, dim=-1)
            v = torch.einsum('bhkl,bhle->bkhe', A, v)  # bsz,max_len,h,dk
            # v = (A[:,:,:,:,None]*v[:,:,None,:,:]).sum(dim=-2).transpose(1,2) # bsz,max_len,h,dk
            all_head_op = v.reshape((bsz, max_len, -1))
            all_head_op = torch.matmul(all_head_op, self.Wo[i])
            all_head_op = F.dropout(all_head_op, self.dropout, self.training)
            # Add+layernorm
            # x = self.layer_norm1[i](all_head_op+x) # bsz,max_len,d
            x = (all_head_op + x) / 2
            # FFN
            ffn_op = torch.matmul(x, self.W1[i]) + self.b1[i]
            ffn_op = F.gelu(ffn_op)
            ffn_op = torch.matmul(ffn_op, self.W2[i]) + self.b2[i]
            ffn_op = F.dropout(ffn_op, self.dropout, self.training)
            # Add+layernorm
            # x = self.layer_norm2[i](ffn_op+x)
            x = (ffn_op + x) / 2
        return x


class Strats(TimeSeriesModel):
    def __init__(self, config: StratsConfig):
        super().__init__()
        self.args = config
        self.demo_emb = nn.Sequential(
            nn.Linear(config.demographics_num, config.hid_dim * 2),
            nn.Tanh(),
            nn.Linear(config.hid_dim * 2, config.hid_dim)
        )
        ts_demo_emb_size = config.hid_dim * 2
        
        if config.head == 'forecast':
            self.forecast_fc = nn.Linear(ts_demo_emb_size, config.features_num)
            self.head = self.forecast_fc
        elif config.head == 'binary':
            self.head = nn.Sequential(
                nn.Linear(ts_demo_emb_size, 1),
                nn.Sigmoid()
            )
        elif config.head == 'forecast_binary':
            self.forecast_fc = nn.Linear(ts_demo_emb_size, config.features_num)
            self.head = nn.Sequential(
                self.forecast_fc,
                nn.Linear(config.features_num, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f'Invalid head type: {config.head}')
        
        self.cve_time = CVE(config.hid_dim)
        self.cve_value = CVE(config.hid_dim)
        self.variable_emb = nn.Embedding(config.features_num, config.hid_dim)
        self.transformer = Transformer(config)
        self.fusion_att = FusionAtt(config.hid_dim)
        self.dropout = config.dropout
        self.features_num = config.features_num
    
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
        triplet_emb = F.dropout(triplet_emb,self.dropout,self.training)
        # contextual triplet emb
        contextual_emb = self.transformer(triplet_emb, input_mask)
        
        # fusion attention
        attention_weights = self.fusion_att(contextual_emb, input_mask)[:, :, None]
        ts_emb = (contextual_emb * attention_weights).sum(dim=1)
        
        # concat demographics and ts_emb
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        
        return self.head(ts_demo_emb)

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def params_number(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return f'Strats({self.args})\nNumber of parameters: {self.params_number:,d}'
