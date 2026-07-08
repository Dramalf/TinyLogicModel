"""TinyGPT: a minimal decoder-only Transformer for TinyLogicLM V1.

刻意做小 (默认 d_model=64, 2 层, 2 头) -- 越小越容易做 circuit 级别的可解释性.
forward(return_activations=True) 会把每层残差流吐出来, 供探针 / logit lens 使用.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        y = self.dropout(attn) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out(y)
        return (y, attn) if return_attn else y


class TransformerBlock(nn.Module):
    """一个 block = 自注意力 + 前馈网络, 每个子层后面接 "Add & Normalize".

    完全对应 Illustrated Transformer 里的结构图:
      - 前馈网络是 Linear -> ReLU -> Linear
      - 残差 + LayerNorm 放在子层"之后" (Post-LN, 即文章画的 Add&Normalize)
    """
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, dropout=0.0):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, return_attn=False):
        a = self.attn(x, return_attn=return_attn)
        if return_attn:
            a, attn = a
        x = self.ln1(x + a)              # Add & Normalize
        x = self.ln2(x + self.mlp(x))    # Add & Normalize
        return (x, attn) if return_attn else x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size=200, max_seq_len=32, d_model=64,
                 n_layers=2, n_heads=2, d_ff=256, dropout=0.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, return_activations=False):
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"T={T} > max_seq_len={self.max_seq_len}"
        pos = torch.arange(T, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None]

        acts = {"embedding": x, "blocks": [], "attn": []} if return_activations else None
        for block in self.blocks:
            if return_activations:
                x, attn = block(x, return_attn=True)
                acts["blocks"].append(x)      # 每层输出的残差流 [B,T,d]
                acts["attn"].append(attn)      # 注意力权重 [B,heads,T,T]
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if return_activations:
            acts["final"] = x
            return logits, acts
        return logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
