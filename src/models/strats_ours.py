import numpy as np
import torch
import torch.nn as nn

from .configs import (
    FeaturesInfo,
    StratsConfig,
)


class CVE(nn.Module):
    def __init__(self, hid_dim: int):
        super().__init__()
        int_dim = int(np.sqrt(hid_dim))
        self.fnn = nn.Sequential(
            nn.Linear(1, int_dim),
            nn.Tanh(),
            nn.Linear(int_dim, hid_dim, bias=False)
        )
    
    def forward(self, x):
        # x: bsz, max_len
        x = torch.unsqueeze(x, -1)
        x = self.fnn(x)
        return x


class FusionAtt(nn.Module):
    def __init__(self, hid_dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1, bias=False)
        )
    
    def forward(self, x, mask):
        # x: bsz, max_len, hid_dim
        att = self.ffn(x).squeeze(-1)  # bsz,max_len
        
        att = att + (~mask) * torch.finfo(x.dtype).min
        att = torch.softmax(att, dim=-1)  # bsz,max_len
        return att


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention
    """
    
    def __init__(self, config: StratsConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.dk = config.hid_dim // config.num_heads
        self.dropout_p = config.attention_dropout
        self.qkv_proj = nn.Linear(config.hid_dim, config.hid_dim * 3, bias=False)
        self.out_proj = nn.Linear(self.dk * self.num_heads, config.hid_dim, bias=False)
    
    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (bsz, seq_len, E_in)
            mask (torch.Tensor): Mask tensor of shape (bsz, seq_len)
        Returns:
            torch.Tensor: Output tensor of shape (bsz, seq_len, E_out)
        """
        bsz, seq_len, _ = x.size()
        # Combined projection for q, k, v
        qkv: torch.Tensor = self.qkv_proj(x)  # (bsz, seq_len, 3 * num_heads * dk)
        
        # (bsz, num_heads, seq_len, dk, 3)
        qkv = qkv.view(bsz, seq_len, self.num_heads, 3, self.dk).permute(0, 2, 1, 4, 3)
        
        q, k, v = qkv.unbind(dim=-1)
        A = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p * self.training,
            attn_mask=mask
        )
        A = A.transpose(1, 2).reshape(bsz, seq_len, -1)  # equivalent to concatenating all mha
        return self.out_proj(A)


class TransformerLayer(nn.Module):
    def __init__(self, config: StratsConfig):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(config.hid_dim, config.hid_dim * 2),
            nn.GELU(),
            nn.Linear(config.hid_dim * 2, config.hid_dim),
            nn.Dropout(config.dropout)
        )
        self.norm1 = nn.LayerNorm(config.hid_dim, eps=1e-5, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(config.hid_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(self, x, mask):
        a = self.mha(x, mask)
        a = self.dropout(a)
        x = self.norm1(x + a)
        f = self.ffn(x)
        x = self.norm2(x + f)
        return x


class Transformer(nn.Module):
    def __init__(self, config: StratsConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(config)
            for _ in range(config.num_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask[:, None, None, :]
        for layer in self.layers:
            x = layer(x, mask)
        return x


class StratsOurs(nn.Module):
    def __init__(
        self,
        config: StratsConfig,
        features_info: FeaturesInfo,
    ):
        super().__init__()
        
        self.cve_time = CVE(config.hid_dim)
        self.cve_value = CVE(config.hid_dim)
        self.variable_emb = nn.Embedding(features_info.features_num + 1,
                                         config.hid_dim,
                                         padding_idx=features_info.features_num)
        self.transformer = Transformer(config)
        self.fusion_att = FusionAtt(config.hid_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.demo_emb = nn.Sequential(
            nn.Linear(features_info.demographics_num, config.hid_dim * 2),
            nn.Tanh(),
            nn.Linear(config.hid_dim * 2, config.hid_dim)
        )
        
        head_layers = {
            'forecast_fc': lambda inp_dim: nn.Linear(inp_dim, features_info.features_num),
            'sigmoid': lambda _: nn.Sigmoid(),
            'binary_fc': lambda inp_dim: nn.Linear(inp_dim, 1),
            'dropout': lambda _: nn.Dropout(config.dropout),
        }
        
        self.head = nn.Sequential()
        last_layer_dim = config.hid_dim * 2  # concat two embeddings with hid_dim width
        for layer_name in config.head_layers:
            layer = head_layers[layer_name](last_layer_dim)
            self.head.add_module(layer_name, layer)
            if hasattr(layer, 'out_features'):
                last_layer_dim = layer.out_features
    
    def forward(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        variables: torch.Tensor,
        input_mask: torch.Tensor,
        demographics: torch.Tensor,
        **kwargs,
    ):
        # demographics embedding
        demo_emb = self.demo_emb(demographics)
        
        # initial triplet embedding
        time_emb = self.cve_time(times)
        value_emb = self.cve_value(values)
        variable_emb = self.variable_emb(variables)
        triplet_emb = time_emb + value_emb + variable_emb
        triplet_emb = self.dropout(triplet_emb)
        # contextual triplet emb
        contextual_emb = self.transformer(triplet_emb, input_mask)
        
        # fusion attention
        attention_weights = self.fusion_att(contextual_emb, input_mask)[:, :, None]
        ts_emb = (contextual_emb * attention_weights).sum(dim=1)
        
        # concat demographics and ts_emb
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        
        return self.head(ts_demo_emb)
