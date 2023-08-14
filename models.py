import pos_embed as pos
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from timm.layers import Mlp, DropPath, use_fused_attn

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, embed_dim=256, grid_size = 3, num_heads=16, mlp_ratio=4.,norm_layer=nn.LayerNorm, depth = 4):
        super(TransformerModel, self).__init__()

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([grid_size**2, embed_dim]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.blocks = nn.ModuleList([
            Block(embed_dim*2, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])

        self.norm = norm_layer(embed_dim*2)

        # self.flatten = nn.Flatten()
        #
        # self.fc1 = nn.Linear(256*9, 256*7)
        #
        # self.fc2 = nn.Linear(256*7, 256*5)
        #
        # self.fc3 = nn.Linear(256*5, 256*3)
        #
        # self.fc4 = nn.Linear(256*3, 256)

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size from the first dimension of x
        x = torch.cat([x, self.pos_embed.unsqueeze(0).expand(batch_size, -1, -1)], dim=2)  # add positional embeddings

        for blk in self.blocks: # multi-headed self-attention layer
            x = blk(x)
        x = self.norm(x)
        # x = self.flatten(x) # flatten
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)

        return x

""" Modification of "Vision Transformer (ViT) in PyTorch"
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
A PyTorch implement of Vision Transformers as described in:
Hacked together by / Copyright 2020, Ross Wightman
"""

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            q_bias=False,
            k_bias=False,
            v_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.w_qs = nn.Linear(dim, dim, bias=q_bias)
        self.w_ks = nn.Linear(dim, dim, bias=k_bias)
        self.w_vs = nn.Linear(dim, dim, bias=v_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):

        batch_size, len_q, len_kv, C = x_q.size(0), x_q.size(1), x_kv.size(1), self.dim

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(x_q).view(batch_size, self.num_heads, len_q, self.head_dim)
        k = self.w_ks(x_kv).view(batch_size, self.num_heads, len_kv, self.head_dim)
        v = self.w_vs(x_kv).view(batch_size, self.num_heads, len_kv, self.head_dim)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(batch_size, len_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            q_bias=False,
            k_bias=False,
            v_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv, mlp_layer=True): # modify to take two inputs and whether or not to do MLP
        x = x_q + self.drop_path1(self.ls1(self.attn(self.norm1(x_q), self.norm1(x_kv))))
        if mlp_layer:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        else:
            x = self.norm2(x)

        return x