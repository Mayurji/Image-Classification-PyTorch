"""
This particular network doesn't come under convolutional networks as the key idea is to use simple MLP architecture.

MLP-Mixer is a multi-layer perceptron based model, it uses common techniques like non-linearites, matrix multiplication,
normalization, skip connections etc. This paper is very interesting to the fact that when MLP was introduced, it was 
particular made upfront that the MLP architectures cannot capture translation invariance in an image. 

Let's see how things have changed. The Network uses a block of MLP Block with two linear layers and 1 activation function
GELU unit. Along with MLPBlock, there are two simple small block called as token mixer and channel mixer.

* First, the image is converted into patches
* These patches are also called as tokens.
* In Token Mixer, we mix these tokens using MLP.
* In Channel Mixer, we mix the channels using MLP.
* The we combine of channel mixer and token mixer.
* It passed into Global Average Pooling and then 
into Fully connected layer.


Best tutorial to learn about einops: https://github.com/arogozhnikov/einops/blob/master/docs

"""
import torch
import torch.nn as nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()
        # Transpose (p x c) into (c x p)
        # In token_mix, each channel learns patches.
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b p c -> b c p'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b c p -> b p c')
        )

        # In channel_mix, each patch's channels will communicate between them.
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        px = x + self.token_mix(x)
        cx = px + self.channel_mix(px)
        
        return cx
        
class MLPMixer(nn.Module):
    def __init__(self, input_channels, dim, n_classes, patch_size, image_size, depth, token_dim, channel_dim):
        
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size// patch_size) ** 2
        
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(input_channels, dim, patch_size, patch_size), #Creating a patches of the images
            Rearrange('b c h w -> b (h w) c'), #combining the patches, we get patches*channel(p x c)
        )

        self.mixer_blocks = nn.ModuleList([])
        for i in range(depth):
            cx = MixerBlock(dim, self.num_patch, token_dim, channel_dim) #Check MixerBlock
            self.mixer_blocks.append(cx)

        
        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(nn.Linear(dim, n_classes))

    def forward(self, x):
        x = self.to_patch_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)      
    
