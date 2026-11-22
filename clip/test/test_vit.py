import pytest
import torch

from clip.model.vit_model import AttentionHead, ImagePatcher, ViT, ViTEncoder
from clip.utils.misc import load_config


@pytest.fixture
def cfg():
    return load_config("clip/config/clip_config.yaml")


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_image_patcher(cfg, bs):
    img_size = cfg.DATA.img_size
    patch_size = cfg.MODEL.VIT.patch_size
    n_patches = (img_size // patch_size) ** 2
    embed_dim = cfg.MODEL.VIT.embed_dim

    x = torch.rand(bs, 3, img_size, img_size)
    image_patcher = ImagePatcher(cfg)
    out = image_patcher(x)

    assert out.shape == (bs, n_patches + 1, embed_dim)


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_attn_head(cfg, bs):
    img_size = cfg.DATA.img_size
    patch_size = cfg.MODEL.VIT.patch_size
    n_patches = (img_size // patch_size) ** 2
    embed_dim = cfg.MODEL.VIT.embed_dim
    n_heads = cfg.MODEL.VIT.n_heads

    x = torch.rand(bs, n_patches + 1, embed_dim)
    attn_head = AttentionHead(cfg)
    out = attn_head(x)

    assert out.shape == (bs, n_patches + 1, embed_dim // n_heads)


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_vit_encoder(cfg, bs):
    img_size = cfg.DATA.img_size
    patch_size = cfg.MODEL.VIT.patch_size
    n_patches = (img_size // patch_size) ** 2
    embed_dim = cfg.MODEL.VIT.embed_dim

    x = torch.rand(bs, n_patches + 1, embed_dim)
    vit_encoder = ViTEncoder(cfg)
    out = vit_encoder(x)

    assert out.shape == x.shape


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_vit(cfg, bs):
    img_size = cfg.DATA.img_size
    embed_dim = cfg.MODEL.VIT.embed_dim

    x = torch.rand(bs, 3, img_size, img_size)
    vit = ViT(cfg)
    out = vit(x)

    assert out.shape == (bs, embed_dim)
