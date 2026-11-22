import pytest
import torch

from clip.model.gpt_model import GPT, AttentionHead, DecoderBlock, FeedForward, MultiHeadAttention
from clip.utils.misc import load_config


@pytest.fixture
def cfg():
    return load_config("clip/config/clip_config.yaml")


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_attn_head(cfg, bs):
    context_size = cfg.MODEL.GPT.context_size
    embed_dim = cfg.MODEL.GPT.embed_dim
    n_heads = cfg.MODEL.GPT.n_heads

    x = torch.rand(bs, context_size, embed_dim)
    attn_head = AttentionHead(cfg)
    out = attn_head(x)

    assert out.shape == (bs, context_size, embed_dim // n_heads)


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_multihead_attn(cfg, bs):
    context_size = cfg.MODEL.GPT.context_size
    embed_dim = cfg.MODEL.GPT.embed_dim

    x = torch.randn(bs, context_size, embed_dim)
    multihead_attn = MultiHeadAttention(cfg)
    out = multihead_attn(x)

    assert out.shape == x.shape


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_decoder_block(cfg, bs):
    context_size = cfg.MODEL.GPT.context_size
    embed_dim = cfg.MODEL.GPT.embed_dim

    x = torch.randn(bs, context_size, embed_dim)
    decoder_block = DecoderBlock(cfg)
    out = decoder_block(x)

    assert out.shape == x.shape


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_feedforward(cfg, bs):
    context_size = cfg.MODEL.GPT.context_size
    embed_dim = cfg.MODEL.GPT.embed_dim

    x = torch.randn(bs, context_size, embed_dim)
    feedforward = FeedForward(cfg)
    out = feedforward(x)

    assert out.shape == x.shape


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_gpt(cfg, bs):
    context_size = cfg.MODEL.GPT.context_size
    vocab_size = cfg.TOKENIZER.vocab_size + 4

    x = torch.randint(0, vocab_size, (bs, context_size), dtype=torch.long)
    gpt = GPT(cfg)
    out = gpt(x)

    assert out.shape == x.shape
