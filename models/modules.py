import torch
import torch.nn as nn

import math


class Embedder(nn.Module):
    def __init__(self, dim, dim_embed, bias=True):
        super(Embedder, self).__init__()
        self.embed = nn.Conv1d(dim, dim_embed, 1, bias=bias)

    def forward(self, x):
        """
        Args:
            x: [B, dim, N]
        Returns:
            embedding: [B, dim_embed, N]
        """
        return self.embed(x)


def simpleMHA(q, k, v, num_heads=8):
    """
    MHA without embedding
    Args:
        q: [B, dim_embed, L]
        k, v : [B, dim_embed, S]
    Returns:
        MHA(q, k, v): [B, dim_embed, L]
    """
    batch, dim_embed, _ = q.shape
    dim_part = int(dim_embed / num_heads)
    q_ = q.view(batch, num_heads, dim_part, -1).permute(1, 0, 3, 2)  # [heads, B, L, dim_part]
    k_ = k.view(batch, num_heads, dim_part, -1).permute(1, 0, 2, 3)  # [heads, B, dim_part, S]
    v_ = v.view(batch, num_heads, dim_part, -1).permute(1, 0, 3, 2)  # [heads, B, S, dim_part]

    # [heads, B, L, dim_part] * [heads, B, dim_part, S] = (heads, B, L, S]
    compatibility = torch.matmul(q_, k_) / math.sqrt(q_.size(-1))

    # [heads, B, L, S] * [heads, B, S, dim_part] = [heads, B, L, dim_part]
    partial_out = torch.matmul(torch.softmax(compatibility, dim=-1), v_)

    # [B, dim_embed, L]
    out = partial_out.permute(1, 0, 3, 2).contiguous().view(batch, dim_embed, -1)

    return out


class MHA(nn.Module):
    """
    MultiHeadAttention(q, k, v) = qk^T/(d_k)^0.5 * v
    """

    def __init__(self, dim_embed, n_heads, norm, dropout):
        """
        Args:
            dim_embed
            norm: Normalization to use(batch, instance or None)
            dropout: droput ratio
        """
        super(MHA, self).__init__()
        # batch_frist=True → input=[B, seq, feature]
        self.mha = nn.MultiheadAttention(dim_embed, n_heads, batch_first=True, dropout=dropout)
        if norm == "batch":
            self.bn = nn.BatchNorm1d(dim_embed, affine=True, track_running_stats=True)
        elif norm == "instance":
            self.bn = nn.InstanceNorm1d(dim_embed, affine=True, track_running_stats=False)
        else:
            self.bn = None

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        """
        Args:
            q: [B, dim_embed, L]
            k, v : [B, dim_embed, S]
        Returns:
            BN(q + Attention(q, k, v)): [B, dim_embed, L]
        """
        # atten_output: [B, L, dim_embed], attn_output_weight: [B, L, S]
        y, _ = self.mha(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        if self.bn is None:
            return q + y.transpose(1, 2)
        return self.bn(q + y.transpose(1, 2))


class FF(nn.Module):
    def __init__(self, dim_embed, norm, hidden_dim):
        """
        Args:
            dim_embed:
            norm: Normalization to use(batch, instance or None)
            hidden_dim 
        """
        super(FF, self).__init__()
        self.affine1 = Embedder(dim_embed, hidden_dim)  # 32 is arbitrary
        self.relu = nn.ReLU()
        self.affine2 = Embedder(hidden_dim, dim_embed, bias=False)  # BatchNormにbias項があるので
        if norm == "batch":
            self.bn = nn.BatchNorm1d(dim_embed, affine=True, track_running_stats=True)
        elif norm == "instance":
            self.bn = nn.InstanceNorm1d(dim_embed, affine=True, track_running_stats=False)
        else:
            self.bn = None

    def forward(self, x):
        """
        Args:
            x: [B, dim_embed, N]
        Returns:
            BN(x + FF(x)): [B, dim_embed, N]
        """
        y = self.affine2(self.relu(self.affine1(x)))
        if self.bn is None:
            return x + y
        return self.bn(x + y)


class SelfAttentionLayer(nn.Module):
    """
    Attention + FF
    h_mha = BN(q + Attention(q, k, v))
    h_ff = BN(h_mha + FN(h_mha))
    """

    def __init__(self, dim_embed, n_heads, norm, dropout, hidden_dim):
        """
        Args:
            dim_embed:
            n_heads:
            norm: Normalization to use(batch, instance or None)
            dropout: droput ratio
            hidden_dim:
        """
        super(SelfAttentionLayer, self).__init__()
        self.mha = MHA(dim_embed, n_heads, norm, dropout)
        self.ff = FF(dim_embed, norm, hidden_dim)

    def forward(self, q):
        """
        Args:
            q: [B, dim_embed, L]
        Returns:
            h_mha: [B, dim_embed, L]
        """
        h_mha = self.mha(q, q, q)
        h_ff = self.ff(h_mha)
        return h_ff


class SelfAttention(nn.Module):
    """
    SelfAttention with mutiple layers
    """

    def __init__(self, dim_embed, n_heads, n_layers, norm, dropout, hidden_dim):
        """
        Args:
            dim_embed:
            n_heads:
            n_layers: number of layers
            norm: Normalization to use(batch, instance or None)
            dropout: droput ratio
            hidden_dim:
        """
        super(SelfAttention, self).__init__()
        layers = [SelfAttentionLayer(dim_embed, n_heads, norm, dropout, hidden_dim) for _ in range(n_layers)]
        self.layers = nn.Sequential(*layers)

    def forward(self, q):
        # nn.Sequential accepts only one argument
        q = self.layers(q)
        return q

