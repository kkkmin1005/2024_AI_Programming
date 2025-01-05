import torch
import torch.nn as nn
from models.modules.patchEmbedding import PatchEmbedding
from models.modules.positionalEncoding import *
from models.modules.transformer import *

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, dim)
        self.positional_encoding = PositionalEncoding(self.patch_embedding.num_patches, dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B, _, _, _ = x.shape

        # Patch Embedding + Positional Encoding
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.positional_encoding(x)

        # Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Classification Head
        cls_output = x[:, 0]  # CLS token만 사용
        return self.mlp_head(cls_output)
