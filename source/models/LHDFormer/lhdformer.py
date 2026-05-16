import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import InterpretableTransformerEncoder, TopologyAwareTransformerEncoder
from omegaconf import DictConfig
from ..base import BaseModel
import pickle
from ...utils.device import device_set

device = device_set()

import torch
import torch.nn.functional as F


def neuro_walk_embedding(data, walk_length=8, beta=0.1):
    """
    NeuroWalk-based biased random walk embedding.
    """
    device = data.device
    num_nodes = data.shape[0]

    degree_inv = data.sum(dim=1).clamp(min=1e-6).reciprocal()
    fro_norm = torch.norm(data, p='fro')
    scale = torch.sigmoid(
        fro_norm / torch.sqrt(torch.tensor(num_nodes, dtype=torch.float32, device=device))
    )

    pe_list = [torch.eye(num_nodes, device=device)]
    current = torch.eye(num_nodes, device=device)

    for k in range(1, walk_length):
        factor_k = torch.exp(
            torch.tensor(beta * (1 - k), dtype=torch.float32, device=device)
        ) * scale
        transition_k = (factor_k * data) * degree_inv.view(-1, 1)

        current = current @ transition_k
        pe_list.append(current)

    pe = torch.stack(pe_list, dim=-1)
    abs_pe = pe.diagonal().transpose(0, 1)
    return abs_pe



def add_full_rrwp(data, walk_length, thresholded=False, thresh=0.3, beta=0.1, add_identity=True):
    """
    Batch wrapper. `data` expected shape: [batch, N, N] (same as original usage in your forward).
    Returns: tensor of shape [batch, N, walk_length]
    """
    pes = []
    batch = data.shape[0]
    for idx in range(batch):
        dt = data[idx].squeeze()
        dt = dt.to(torch.float32)
        pe = add_every_rrwp(dt, walk_length=walk_length, thresholded=thresholded, thresh=thresh,
                             beta=beta, add_identity=add_identity)
        pes.append(pe)
    return torch.stack(pes, dim=0)


def add_every_rrwp(data,
                   walk_length=8,
                   add_identity=True,
                   spd=False,
                   thresholded=False,
                   thresh=0.3,
                   beta=0.1,
                   **kwargs):

    device = data.device
    data = data.to(torch.float32).to(device)

    if thresholded:
        mask = (data > thresh).to(data.dtype)
        data = data * mask

    data = torch.abs(data)
    data = data.fill_diagonal_(0)

    pe = neuro_walk_embedding(data, walk_length=walk_length, beta=beta)

    if not add_identity:
        if pe.shape[1] > 0:
            pe = pe[:, 1:]
        else:
            pass

    return pe


class TransPoolingEncoder(nn.Module):

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True,
                 freeze_center=False, project_assignment=True, nHead=4, local_transformer=False,
                 topology_aware=False, num_tokens=5):
        super().__init__()
        self.topology_aware = topology_aware
        transformer_cls = TopologyAwareTransformerEncoder if topology_aware else InterpretableTransformerEncoder
        self.transformer = transformer_cls(d_model=input_feature_size, nhead=nHead,
                                           dim_feedforward=hidden_size,
                                           batch_first=True)

        self.local_transformer = local_transformer
        if local_transformer:
            self.pooling = False
        else:
            self.pooling = pooling
        if self.pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

        if local_transformer:
            self.class_token = nn.ParameterList()
            for _ in range(num_tokens):
                self.class_token.append(
                    nn.Parameter(torch.Tensor(1, input_feature_size), requires_grad=True).to(device))

        self.reset_parameters(local_transformer)

    def reset_parameters(self, local_transformer=False):
        if local_transformer:
            for i in range(len(self.class_token)):
                nn.init.xavier_normal_(self.class_token[i])

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self,
                x: torch.tensor, cluster_num=-1, adjacency=None):
        bz, node_num, dim = x.shape
        if self.local_transformer:
            class_token = self.class_token[cluster_num]
            class_token = class_token.repeat(bz, 1, 1)
            x = torch.cat((class_token, x), dim=1)
        if self.topology_aware:
            x = self.transformer(x, adjacency=adjacency)
        else:
            x = self.transformer(x)
        if self.local_transformer:
            cls_token = x[:, 0, :]
            x = x[:, 1:, :]
            return x, None, cls_token.reshape(x.shape[0], 1, -1)
        else:
            cls_token = x[:, 0, :]
            x = x[:, 1:, :]
            if self.pooling:
                x, assignment = self.dec(x)
                return x, assignment, cls_token.reshape(x.shape[0], 1, -1)
            else:
                return x, None, cls_token.reshape(x.shape[0], 1, -1)

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class LHDFormer(BaseModel):

    def __init__(self, config: DictConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        self.pos_encoding = config.model.pos_encoding
        self.pos_embed_dim = config.model.pos_embed_dim
        forward_dim = config.dataset.node_sz

        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)
        if self.pos_encoding == 'rrwp':
            forward_dim = config.model.pos_embed_dim

        self.node_sz = config.dataset.node_sz
        self.num_windows = int(config.model.get("num_windows", 5))
        self.total_window_nodes = self.node_sz * self.num_windows
        self.forward_dim = forward_dim
        self.window_threshold = float(config.model.get("window_threshold", 0.3))
        self.thresholded = bool(config.model.get("thresholded", True))
        self.neurowalk_beta = float(config.model.get("beta", 0.1))
        hidden_size = int(config.model.get("hidden_size", 1024))

        self.local_transformer = TransPoolingEncoder(input_feature_size=forward_dim,
                                                     input_node_num=self.node_sz,
                                                     hidden_size=hidden_size,
                                                     output_node_num=self.node_sz,
                                                     pooling=False,
                                                     orthogonal=config.model.orthogonal,
                                                     freeze_center=config.model.freeze_center,
                                                     project_assignment=config.model.project_assignment,
                                                     nHead=config.model.nhead,
                                                     local_transformer=True,
                                                     topology_aware=True,
                                                     num_tokens=self.num_windows)

        self.temporal_transformer = InterpretableTransformerEncoder(d_model=forward_dim,
                                                                    nhead=config.model.nhead,
                                                                    dim_feedforward=hidden_size,
                                                                    batch_first=True)

        readout_dim = (self.num_windows + 2) * forward_dim
        self.fc = nn.Sequential(
            nn.Linear(readout_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

        self.assignMat = None

    @staticmethod
    def _build_window_slices(timepoints, num_windows):
        if timepoints < num_windows:
            raise ValueError(
                f"Expected at least {num_windows} time points, got {timepoints}")
        window_length = timepoints // num_windows
        if window_length <= 1:
            raise ValueError(
                f"Expected each window to contain more than 1 time point, got {window_length}")
        window_slices = []
        for idx in range(num_windows):
            start = idx * window_length
            end = start + window_length
            window_slices.append((start, end))
        return window_slices

    def sliding_windows_pearson(self, tensor):
        batch_size, num_nodes, time_series_length = tensor.shape
        correlation_matrices = []
        for i in range(batch_size):
            sample = tensor[i]
            corr_matrix = torch.corrcoef(sample)
            nan_mask = torch.isnan(corr_matrix)
            if nan_mask.any():
                corr_matrix[nan_mask] = 0.000001
            corr_matrix = torch.abs(corr_matrix)
            corr_matrix = corr_matrix.fill_diagonal_(0)
            if self.thresholded:
                corr_matrix = torch.where(
                    corr_matrix >= self.window_threshold,
                    corr_matrix,
                    torch.zeros_like(corr_matrix)
                )
            corr_matrix = torch.round(corr_matrix * 1000) / 1000
            correlation_matrices.append(corr_matrix)
        return torch.stack(correlation_matrices)

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):

        bz, node_num, _ = node_feature.shape
        if node_num != self.node_sz:
            raise ValueError(f"Expected {self.node_sz} nodes, got {node_num}")
        if time_seires.shape[1] != self.node_sz:
            raise ValueError(
                f"Expected time series with {self.node_sz} nodes, got {time_seires.shape[1]}"
            )
        window_slices = self._build_window_slices(
            time_seires.shape[2], self.num_windows)
        local_spatial_features = []
        local_class_tokens = []

        for idx, (start_time, end_time) in enumerate(window_slices):
            window_adj = self.sliding_windows_pearson(
                time_seires[:, :, start_time:end_time])

            if self.pos_encoding == 'identity':
                pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
                window_input = torch.cat([window_adj, pos_emb], dim=-1)
            elif self.pos_encoding == 'rrwp':
                window_input = add_full_rrwp(
                    window_adj, self.pos_embed_dim, beta=self.neurowalk_beta)
            else:
                window_input = window_adj

            window_spatial, _, local_class_token = self.local_transformer(
                window_input, cluster_num=idx, adjacency=window_adj
            )
            local_spatial_features.append(window_spatial)
            local_class_tokens.append(local_class_token)

        spatial_tokens = torch.cat(local_class_tokens, dim=1)
        spatial_features = torch.cat(local_spatial_features, dim=1)
        temporal_input = torch.cat([spatial_tokens, spatial_features], dim=1)
        temporal_output = self.temporal_transformer(temporal_input)

        token_output = temporal_output[:, :self.num_windows, :].reshape(bz, -1)
        node_output = temporal_output[:, self.num_windows:, :]
        mean_pool = node_output.mean(dim=1)
        max_pool = node_output.max(dim=1).values
        readout = torch.cat([token_output, mean_pool, max_pool], dim=-1)

        self.assignMat = None
        return self.fc(readout), None

    def get_assign_mat(self):
        return self.assignMat

    def get_attention_weights(self):
        return self.temporal_transformer.get_attention_weights()

    def get_local_attention_weights(self):
        return self.local_transformer.get_attention_weights()

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return None

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        return None
