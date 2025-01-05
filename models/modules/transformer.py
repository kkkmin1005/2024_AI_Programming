import torch
import torch.nn as nn
from models.modules.MHSA import *
from models.modules.FFN import *

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.feed_forward = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Self-Attention
        attn_output = self.attention(self.norm1(x))
        x = x + attn_output

        # Feed Forward
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output

        return x
