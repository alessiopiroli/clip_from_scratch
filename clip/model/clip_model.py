import math

import torch
import torch.nn as nn

from clip.model.gpt_model import GPT
from clip.model.vit_model import PretrainedViT, ViT


class CLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = GPT(self.cfg)

        if self.cfg.MODEL.VIT.PRETRAINED.pretrained:
            self.vision_encoder = PretrainedViT(self.cfg)
            vision_dim = self.vision_encoder.out_dim
        else:
            self.vision_encoder = ViT(self.cfg)
            vision_dim = self.cfg.MODEL.VIT.embed_dim

        self.image_proj = nn.Linear(vision_dim, self.cfg.MODEL.joint_dim)
        self.text_proj = nn.Linear(self.cfg.MODEL.GPT.embed_dim, self.cfg.MODEL.joint_dim)
        self.temperature = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, imgs, text):
        imgs = self.vision_encoder(imgs)
        text = self.text_encoder(text)

        imgs_proj = self.image_proj(imgs)
        text_proj = self.text_proj(text)

        image_norm = torch.linalg.vector_norm(imgs_proj, dim=-1, keepdim=True)
        text_norm = torch.linalg.vector_norm(text_proj, dim=-1, keepdim=True)

        imgs_proj = imgs_proj / image_norm
        text_proj = text_proj / text_norm

        return imgs_proj, text_proj, self.temperature
