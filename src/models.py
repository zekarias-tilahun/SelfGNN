from torch_geometric.nn import GCNConv, GATConv, SAGEConv

import torch.nn.functional as F
import torch.nn as nn
import torch

from functools import wraps
import copy

torch.manual_seed(0)

"""
The following code is borrowed from BYOL

=====================Start=================
"""


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


"""
=====================End====================
"""


class Encoder(nn.Module):

    def __init__(self, layer_config, gnn_type, dropout=None, project=False, **kwargs):
        super().__init__()
        rep_dim = layer_config[-1]
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.project = project
        self.stacked_gnn = get_encoder(layer_config=layer_config, gnn_type=gnn_type, **kwargs)
        if self.project:
            self.projection_head = nn.Sequential(nn.Linear(rep_dim, rep_dim), nn.BatchNorm1d(rep_dim),
                                                 nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        for i, gnn in enumerate(self.stacked_gnn):
            if self.gnn_type == "gat" or self.gnn_type == "sage":
                x = gnn(x, edge_index)
            else:
                x = gnn(x, edge_index, edge_weight=edge_weight)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x, (self.projection_head(x) if self.project else None)


class SelfGNN(nn.Module):

    def __init__(self, layer_config, dropout=0.0, moving_average_decay=0.99, gnn_type='gcn', **kwargs):
        super().__init__()
        self.student_encoder = Encoder(layer_config=layer_config, gnn_type=gnn_type, dropout=dropout, **kwargs)
        self.teacher_encoder = None
        self.teacher_ema_updater = EMA(moving_average_decay)
        rep_dim = layer_config[-1]
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, rep_dim), nn.BatchNorm1d(rep_dim),
                                               nn.ReLU(inplace=True), nn.Dropout(dropout))

    @singleton('teacher_encoder')
    def _get_teacher_encoder(self):
        teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(teacher_encoder, False)
        return teacher_encoder

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def encode(self, x, edge_index, edge_weight=None, encoder=None):
        encoder = self.student_encoder if encoder is None else encoder
        encoder.train(self.training)
        return encoder(x, edge_index, edge_weight)

    def forward(self, x1, x2, edge_index_v1, edge_index_v2, edge_weight_v1=None, edge_weight_v2=None):
        v1_enc = self.encode(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1)
        v1_rep, v1_student = v1_enc if v1_enc[1] is not None else (v1_enc[0], v1_enc[0])
        v2_enc = self.encode(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2)
        v2_rep, v2_student = v2_enc if v2_enc[1] is not None else (v2_enc[0], v2_enc[0])

        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)
        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            v1_enc = self.encode(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1, encoder=teacher_encoder)
            v1_teacher = v1_enc[1] if v1_enc[1] is not None else v1_enc[0]
            v2_enc = self.encode(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2, encoder=teacher_encoder)
            v2_teacher = v2_enc[1] if v2_enc[1] is not None else v2_enc[0]

        loss1 = loss_fn(v1_pred, v2_teacher.detach())
        loss2 = loss_fn(v2_pred, v1_teacher.detach())

        loss = loss1 + loss2
        return v1_student, v2_student, loss.mean()


def get_encoder(layer_config, gnn_type, **kwargs):
    if gnn_type == "gcn":
        return nn.ModuleList([GCNConv(layer_config[i-1], layer_config[i]) for i in range(1, len(layer_config))])
    elif gnn_type == "sage":
        return nn.ModuleList([SAGEConv(layer_config[i-1], layer_config[i]) for i in range(1, len(layer_config))])
    elif gnn_type == "gat":
        heads = kwargs['heads'] if 'heads' in kwargs else [8] * len(layer_config)
        concat = kwargs['concat'] if 'concat' in kwargs else True
        return nn.ModuleList([GATConv(layer_config[i-1], layer_config[i] // heads[i-1], heads=heads[i-1], concat=concat)
                              for i in range(1, len(layer_config))])
