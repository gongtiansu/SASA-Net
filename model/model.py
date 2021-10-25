import math

import torch
from torch import nn
import torch.nn.functional as F


class AAEmbeddings(nn.Module):
    def __init__(self, embed_dim):
        super(AAEmbeddings, self).__init__()
        self.embed_dim = embed_dim
        self.onehot = nn.Embedding(21, 21)
        self.onehot.weight.data = torch.eye(21)
        self.onehot.weight.requires_grad = False
        self.aa_embeddings = nn.Linear(21, embed_dim)

    def forward(self, seq_ids):
        """
        seq_ids: (B, L)
        return: (B, L, C)
        """
        x = self.onehot(seq_ids)
        x = self.aa_embeddings(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, ninp, dim_feedforward, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(ninp, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, ninp)
        self.norm1 = nn.LayerNorm(ninp)
        self.norm2 = nn.LayerNorm(ninp)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward_fn(self, x, branch):
        x = x + self.dropout1(branch)
        x = self.norm1(x)
        branch = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(branch)
        x = self.norm2(x)
        return x

    def forward(self, x, branch):
        return self.forward_fn(x, branch)


class X3DAttention(nn.Module):
    def __init__(self, ninp, nhead, dim2d, dropout):
        super(X3DAttention, self).__init__()
        if ninp % nhead != 0:
            raise ValueError(
                "The hidden size is not a multiple of the number of attention heads"
            )
        self.nhead = nhead
        self.ninp = ninp

        self.fc_query = nn.Linear(ninp, ninp)
        self.fc_key = nn.Linear(ninp, ninp)
        self.fc_value = nn.Linear(ninp, ninp)
        self.pre_proj = nn.Sequential(
            nn.Linear(dim2d, ninp),
            nn.LayerNorm(ninp),
            nn.ReLU(),
        )
        self.efc_q = nn.Linear(ninp, ninp)
        self.efc_k = nn.Linear(ninp, ninp)
        self.fc = nn.Linear(ninp, ninp)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x, nhead):
        """
        x has shape (*, L, C)
        return shape (*, nhead, L, C/nhead)
        """
        new_shape = x.shape[:-1] + (nhead, -1)
        x = x.view(*new_shape)
        return x.transpose(-3, -2)

    def forward(self, x2d, x3d):
        """
        x2d has shape (B, L, L, C)
        x3d has shape (B, L, C)
        return shape (B, L, C)
        """
        query = self.transpose_for_scores(self.fc_query(x3d), self.nhead)
        key = self.transpose_for_scores(self.fc_key(x3d), self.nhead)
        value = self.transpose_for_scores(self.fc_value(x3d), self.nhead)

        edge = self.pre_proj(x2d)
        # (B, L, nhead, L, C)
        efq = self.transpose_for_scores(self.efc_q(edge), self.nhead)
        efk = self.transpose_for_scores(self.efc_k(edge), self.nhead)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        ex = torch.sum(efk * query.unsqueeze(1), dim=-1).permute(0, 2, 1, 3)
        attention_scores += ex

        attention_scores = attention_scores / math.sqrt(self.ninp / self.nhead)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        x1 = torch.matmul(attention_weights, value)
        x2 = torch.sum(attention_scores.unsqueeze(2) * efq.permute(0, 2, 4, 1, 3), dim=-1)
        x = x1 + x2.transpose(-1, -2)
        x = x.transpose(-3, -2)
        x = x.reshape(*x.shape[:-2], -1)
        x = self.dropout(self.fc(x))
        return x


class X3DTransformerLayer(nn.Module):
    def __init__(self, ninp, nhead, dim_feedforward, dropout, dim2d):
        super(X3DTransformerLayer, self).__init__()
        self.attention = X3DAttention(
            ninp=ninp, nhead=nhead, dim2d=dim2d + 6, dropout=dropout
        )
        self.feed_forward = FeedForward(
            ninp=ninp, dim_feedforward=dim_feedforward, dropout=dropout
        )

    def forward(self, x2d, x3d, pt, pr):
        pose_feat = torch.matmul(
            pr.unsqueeze(2),
            (pt.unsqueeze(1) - pt.unsqueeze(2)).unsqueeze(-1),
        ).squeeze(-1)
        pose_feat = pose_feat / 10
        new_x2d = torch.cat([x2d, pose_feat, pose_feat.transpose(-2, -3)], dim=-1)
        branch = self.attention(new_x2d, x3d)
        x3d = self.feed_forward(x3d, branch)

        return x2d, x3d


class X3DTransformer(nn.Module):
    def __init__(self, n_layer, **kwargs):
        super(X3DTransformer, self).__init__()
        self.layers = nn.ModuleList(
            [X3DTransformerLayer(**kwargs) for _ in range(n_layer)]
        )
        ninp = kwargs["ninp"]
        self.fc = nn.Sequential(nn.Linear(ninp, ninp), nn.ReLU(), nn.Linear(ninp, 9))

    def forward(self, node_feat, x2d, pose, prefix=""):
        x3d = node_feat
        extra = {}
        pt, pr = pose
        for i, model_fn in enumerate(self.layers):
            x2d, x3d = model_fn(x2d, x3d, pt, pr)
        x = self.fc(x3d)
        x = x.reshape(*x.shape[:-1], 3, 3)

        nt = pt + torch.matmul(pr.transpose(-1, -2), x[..., :1]).squeeze(-1)
        v1, v2 = x[..., 1], x[..., 2]
        v1 = v1 / torch.norm(v1, dim=-1).unsqueeze(-1)
        v2 = v2 / torch.norm(v2, dim=-1).unsqueeze(-1)
        e1 = v1
        e2 = torch.cross(e1, v2)
        e3 = torch.cross(e1, e2)
        rot_inv = torch.cat([e1, e2, e3], dim=-1).reshape(*e3.shape[:-1], 3, 3)
        nr = torch.matmul(rot_inv, pr)
        return x2d, (nt, nr)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.node_dim = 256
        self.edge_dim = 128 + 37 + 25 + 25 + 13
        self.max_pos = 128

        self.aa_embeddings = nn.Sequential(
            AAEmbeddings(self.node_dim), nn.LayerNorm(self.node_dim)
        )
        self.pos_embeddings = nn.Sequential(
            nn.Embedding(self.max_pos * 2 + 1, 128), nn.LayerNorm(128)
        )

        self.x3d_transformer = X3DTransformer(
            n_layer=4,
            ninp=self.node_dim,
            nhead=self.node_dim // 16,
            dim_feedforward=self.node_dim * 4,
            dropout=0.1,
            dim2d=self.edge_dim,
        )

    def forward(self, data):
        node_feat = self.aa_embeddings(data["seq"])
        seq = data["seq"]
        index = torch.arange(seq.shape[1], device=seq.device)[None]
        rp = index.unsqueeze(2) - index.unsqueeze(1) + self.max_pos
        rp = torch.clamp(rp, 0, self.max_pos * 2)
        edge_feat = torch.cat([self.pos_embeddings(rp), data["pwt_feat"]], dim=-1)

        B, L = node_feat.shape[:2]
        init_T = torch.zeros(B, L, 3).to(node_feat)
        init_R = torch.eye(3).to(node_feat)[None, None].repeat(B, L, 1, 1)

        ret = {}
        pose = (init_T, init_R)
        n_iter = 6 if self.training else 20
        for i in range(n_iter):
            _, pose = self.x3d_transformer(node_feat, edge_feat, pose)
            pose = (pose[0].detach(), pose[1].detach())

        return pose
