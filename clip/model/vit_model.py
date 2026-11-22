import torch
import torch.nn as nn
import torchvision


class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = self.cfg.MODEL.VIT.embed_dim
        self.n_blocks = self.cfg.MODEL.VIT.n_blocks

        self.image_patcher = ImagePatcher(self.cfg)
        self.encoders = nn.ModuleList([ViTEncoder(self.cfg) for _ in range(self.n_blocks)])
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x = self.image_patcher(x)
        for encoder in self.encoders:
            x = encoder(x)

        x = self.layer_norm(x)
        x = x[:, 0, :]

        return x


class ImagePatcher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = self.cfg.MODEL.VIT.patch_size
        self.n_patches = (self.cfg.DATA.img_size // self.patch_size) ** 2
        self.lin_in = (self.patch_size**2) * 3
        self.lin_out = self.cfg.MODEL.VIT.embed_dim

        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.linear_proj = nn.Linear(self.lin_in, self.lin_out)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.lin_out) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, self.lin_out) * 0.02)

    def forward(self, x):
        x = self.unfold(x).transpose(-1, -2)
        x = self.linear_proj(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=-2)
        x = x + self.pos_embedding

        return x


class ViTEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_heads = self.cfg.MODEL.VIT.n_heads
        self.embed_dim = self.cfg.MODEL.VIT.embed_dim

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.attn_heads = nn.ModuleList([AttentionHead(self.cfg) for _ in range(self.n_heads)])
        self.attn_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4), nn.GELU(), nn.Linear(self.embed_dim * 4, self.embed_dim)
        )

    def forward(self, x):
        x_norm = self.layer_norm1(x)
        attn_out = torch.cat([head(x_norm) for head in self.attn_heads], dim=-1)
        x = self.attn_proj(attn_out) + x
        x_norm = self.layer_norm2(x)
        x = self.mlp(x_norm) + x

        return x


class AttentionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = self.cfg.MODEL.VIT.embed_dim
        self.n_heads = self.cfg.MODEL.VIT.n_heads
        self.out_dim = self.embed_dim // self.n_heads

        self.query = nn.Linear(self.embed_dim, self.out_dim)
        self.key = nn.Linear(self.embed_dim, self.out_dim)
        self.value = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn = (Q @ torch.transpose(K, dim0=-2, dim1=-1)) / (self.out_dim**0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = attn @ V

        return attn


class PretrainedViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.freeze = self.cfg.MODEL.VIT.PRETRAINED.freeze

        self.backbone = torchvision.models.get_model(self.cfg.MODEL.VIT.PRETRAINED.name, weights="DEFAULT")

        self.backbone.heads = nn.Identity()
        self.out_dim = self.backbone.hidden_dim

        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)

        if self.freeze:
            self.backbone.eval()

        return self

    def forward(self, x):
        x = self.backbone(x)

        return x
