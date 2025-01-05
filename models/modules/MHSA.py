import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Query, Key, Value projection layers
        self.qkv = nn.Linear(dim, dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Scaled Dot-Product Attention
        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, N, N)
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.attention_dropout(attn_probs)

        # Attention Output
        attn_output = (attn_probs @ V).transpose(1, 2).reshape(B, N, D)  # (B, N, dim)
        return self.out_proj(attn_output)
