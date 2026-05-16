from torch.nn import TransformerEncoderLayer
from torch import Tensor
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpretableTransformerEncoder(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.3, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.attention_weights: Optional[Tensor] = None

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                  is_causal: bool = False) -> Tensor:
        x, weights = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    is_causal=is_causal,
                                    need_weights=True,
                                    average_attn_weights=False)
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self) -> Optional[Tensor]:
        return self.attention_weights


class TopologyAwareTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.3, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 device=None, dtype=None) -> None:
        super().__init__()
        if not batch_first:
            raise ValueError("TopologyAwareTransformerEncoder expects batch_first=True")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.activation = activation
        self.norm_first = norm_first

        self.q_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.attention_weights: Optional[Tensor] = None

    def _normalize_adjacency(self, adjacency: Tensor) -> Tensor:
        adjacency = torch.abs(adjacency)
        degree = adjacency.sum(dim=-1)
        degree_inv_sqrt = degree.clamp(min=1e-6).pow(-0.5)
        return adjacency * degree_inv_sqrt.unsqueeze(-1) * degree_inv_sqrt.unsqueeze(-2)

    def _graph_project_input(self, x: Tensor, adjacency: Optional[Tensor]) -> Tensor:
        if adjacency is None:
            return x

        norm_adj = self._normalize_adjacency(adjacency.to(dtype=x.dtype, device=x.device))
        if x.size(1) == norm_adj.size(1) + 1:
            token, nodes = x[:, :1, :], x[:, 1:, :]
            nodes = torch.bmm(norm_adj, nodes)
            return torch.cat([token, nodes], dim=1)

        return torch.bmm(norm_adj, x)

    def _shape_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.nhead, self.head_dim)
        return x.transpose(1, 2)

    def _topology_attention(self, x: Tensor, adjacency: Optional[Tensor]) -> Tensor:
        graph_x = self._graph_project_input(x, adjacency)
        q = self._shape_heads(F.relu(self.q_proj(graph_x)))
        k = self._shape_heads(F.relu(self.k_proj(graph_x)))
        v = self._shape_heads(F.relu(self.v_proj(graph_x)))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        self.attention_weights = weights
        weights = self.dropout(weights)

        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(
            x.size(0), x.size(1), self.d_model)
        return self.dropout1(self.out_proj(context))

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, x: Tensor, adjacency: Optional[Tensor] = None) -> Tensor:
        if self.norm_first:
            x = x + self._topology_attention(self.norm1(x), adjacency)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._topology_attention(x, adjacency))
            x = self.norm2(x + self._ff_block(x))
        return x

    def get_attention_weights(self) -> Optional[Tensor]:
        return self.attention_weights
