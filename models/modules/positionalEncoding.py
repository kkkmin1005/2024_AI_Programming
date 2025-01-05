import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, dim):
        super(PositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches+1, dim))

    def forward(self, x):
        return x + self.position_embeddings
