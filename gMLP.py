"""
gMLP is a MLP with gating architecture. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_projection = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_projection.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_projection(v)
        out  = u * v
        return out

class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_projection_1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_projection_2 = nn.Linear(d_ffn, d_model)
        self.SGU = SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_projection_1(x))
        x = self.SGU(x)
        x = self.channel_projection_2(x)
        out = x + residual
        return out

class gMLP(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, num_layers=6) -> None:
        super().__init__()
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)

def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "'image_size' should be divisible by patch size"
    num_patches = sqrt_num_patches ** 2
    return num_patches

class gMLPForImageClassification(gMLP):
    def __init__( self, in_channels, n_classes, image_size, patch_size, 
        d_model=256, d_ffn=512, seq_len=256, num_layers=6):

        num_patches = check_sizes(image_size, patch_size)
        super().__init__(d_model, d_ffn, seq_len, num_layers)
        self.patcher = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)
        embedding = embedding.mean(dim=1)
        out = self.classifier(embedding)
        return out