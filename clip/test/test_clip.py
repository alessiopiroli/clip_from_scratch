import pytest
import torch

from clip.model.clip_model import CLIP
from clip.utils.misc import load_config


@pytest.fixture
def cfg():
    return load_config("clip/config/clip_config.yaml")


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_clip(cfg, bs):
    img_size = cfg.DATA.img_size
    context_size = cfg.MODEL.GPT.context_size
    vocab_size = cfg.TOKENIZER.vocab_size + 4
    join_dim = cfg.MODEL.joint_dim

    visual_input = torch.rand(bs, 3, img_size, img_size)
    text_input = torch.randint(0, vocab_size, (bs, context_size), dtype=torch.long)
    clip = CLIP(cfg)
    visual_out, text_out, _ = clip(visual_input, text_input)

    assert visual_out.shape == (bs, join_dim)
    assert text_out.shape == (bs, join_dim)
