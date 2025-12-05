import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple
import math


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = nn.Parameter(torch.ones(dim))

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.ls2 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]

        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size

        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.interpolate_pos_encoding(x, W, H)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x


class DinoBloomFeatureExtractor(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        resolution_strategy: Literal["resize", "interpolate"] = "resize",
        freeze: bool = True
    ):
        super().__init__()
        self.resolution_strategy = resolution_strategy

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.backbone = VisionTransformer(
            img_size=224,
            patch_size=14,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0
        )

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value

        missing_keys, unexpected_keys = self.backbone.load_state_dict(new_state_dict, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys[:5]}...")

        self.embed_dim = 768

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.resolution_strategy == "resize" and x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        features = self.backbone(x)

        cls_token = features[:, 0]
        patch_tokens = features[:, 1:]
        avg_patch = patch_tokens.mean(dim=1)

        concat_features = torch.cat([cls_token, avg_patch], dim=1)

        return cls_token, avg_patch, concat_features


def load_dinobloom(
    checkpoint_path: str,
    resolution_strategy: Literal["resize", "interpolate" , ] = "resize",
    freeze: bool = True,
    device: str = "mps"
) -> DinoBloomFeatureExtractor:
    model = DinoBloomFeatureExtractor(
        checkpoint_path=checkpoint_path,
        resolution_strategy=resolution_strategy,
        freeze=freeze
    )
    model = model.to(device)
    model.eval()
    return model
