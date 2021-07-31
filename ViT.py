"""
Transformers are the backbone architecture for many of the NLP architectures like BERT etc. Though, it
started with focus on NLP tasks, the transformer is used in computer vision space.

Vision Transformer aka ViT

We'll discuss about Transformer architecture separately except the notion on data, we'll see how the 
image is processed in transformer, which was primarily built for sentence tokens. There are series of 
steps followed to convert image into sequence of token and passed into transformer encoder with MLP.

* Convert Image into Patches of fixed size.
* Flatten those patches into sequence of embedding
* Add positional embeddings
* Feed the sequence into transformer encoder
* And predict using MLP block at last.

I've omitted few notions from transformer architecture like residual connections, multi-head attention
etc. Each of these concept requires separate blog post.

Note: ViT was trained on large image dataset with 14M images, and the pretrained model is fine tuned to 
work with our custom dataset.

Citation: https://github.com/lucidrains/vit-pytorch/tree/main/vit_pytorch
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, func):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.func = func

    def forward(self, x, **kwargs):
        return self.func(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, input_channel, output_channel, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_channel, output_channel),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_channel, input_channel),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
    Attention (Q, K, V) = softmax( Q . (K.T) / (d_m ** 0.5) ) . V
    """

    def __init__(self, input_channel, heads=8, dimension_head=64, dropout=0.):
        super().__init__()
        inner_dim = dimension_head * heads #inner_dim=16*64=512 if heads=16
        project_out = not(heads==1 and dimension_head==input_channel)

        self.heads = heads
        self.scale = dimension_head ** -0.5 # scaling factor

        self.attend = nn.Softmax(dim=-1)
        self.to_QKV = nn.Linear(input_channel, inner_dim*3, bias=False)

        self.out = nn.Sequential(
            nn.Linear(inner_dim, input_channel),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        QKV = self.to_QKV(x).chunk(3, dim=-1)
        Q, K, V = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), QKV)

        dots = einsum('b h i d, b h j d -> b h i j', Q, K) * self.scale

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, V)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.out(out)

class Transformers(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads, dimension_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, n_classes, dim, depth, heads, mlp_dim, pool='cls', 
                input_channel=1, dim_head=64, dropout=0.,emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, "Image dimension must be divisible by patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = input_channel * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformers(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes)
        )

    def forward(self, input):
        """
        to_patch_embedding:
            input image: (batch x channel x height x width) => (32 x 3 x 224 x 224)
            Using rearrange: 
                p1, p2 = patch_size
                * (b x c x h x w) => (b x (h x p1) x (w x p2) x c)
                * (b x (h x p1) x (w x p2) x c) => (b x (h w) x (p1 x p2 x c))
                * (32 x (7 x 32) x (7 x 32) x 3) => (32 x 49 x 3072)
            
            Passing through Linear Layer:
                * (32 x 49 x 3072) => (32 x 49 x 1024)
            
        Adding Positional Embedding 
            * (32 x 49 x 1024) + (32 x 1 x 1024) => (32 x 50 x 1024)
        
        Transformers
        Input is (32 x 50 x 1024)
            * LayerNorm -> Attention => output1
                * (32 x 50 x 1024) ---LN---> (32 x 50 x 1024)
                * (32 x 50 x 1024) ---Attn--> 
                    * Using Linear Layer, we generate Q, K, V matrices by splitting 
                        * (32 x 50 x 1024) => (32 x 50 x 3072) using chunks
                        * (32 x 50 x 3072) => Q - (32 x 50 x 1024) 
                                              K - (32 x 50 x 1024) 
                                              V - (32 x 50 x 1024)
                        * Using Rearrange, 
                            Q - (32 x 50 x 1024) ==(1024 = 16 x 64)==> (32 x 16 x 50 x 64)
                            K - (32 x 50 x 1024) ==(1024 = 16 x 64)==> (32 x 16 x 50 x 64)
                            V - (32 x 50 x 1024) ==(1024 = 16 x 64)==> (32 x 16 x 50 x 64)
                        * Attention Weights, Q and K, dots operation
                            einsum(Q, K), 
                            (32 x 16 x 50 x 64) . (32 x 16 x 64 x 50) => (32 x 16 x 50 x 50)
                        * Attention Weights and Value
                            Attn . Value
                            (32 x 16 x 50 x 50) . (32 x 16 x 50 x 64) ==(16 x 64 = 1024)==> (32 x 50 x 1024)
                        * Attn.Value -> Linear
                            (32 x 50 x 1024) --Linear--> (32 x 50 x 1024)
                Attention output => (32 x 50 x 1024)
            
            * output => LayerNorm --> FeedForward
                * To LayerNorm, (32 x 50 x 1024) --LN--> (32 x 50 x 1024) from output of attention.
                * To Linear, (32 x 50 x 1024) --Linear--> (32 x 50 x 1024) from output of LayerNorm
                * (32 x 50 x 1024) <---Residual--> output(attention output)
                * (32 x 50 x 1024)
      MLP Head
      * Linear Layer to n_classes
          * (32 x 50 x 1024) => * (32 x 10)

        """

        x = self.to_patch_embedding(input)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout(x)
        
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        
        return self.mlp_head(x)
