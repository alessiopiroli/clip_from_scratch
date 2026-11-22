import torch
import torch.nn as nn


class CLIPLoss:
    def __init__(self):
        pass

    def get_loss(self, image_proj, text_proj, temp):
        assert image_proj.shape == text_proj.shape

        logits = image_proj @ torch.transpose(text_proj, dim0=0, dim1=-1)
        logits = logits * torch.exp(temp)

        labels = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)

        image_loss = nn.functional.cross_entropy(logits, labels)
        text_loss = nn.functional.cross_entropy(torch.transpose(logits, dim0=0, dim1=-1), labels)

        loss = (image_loss + text_loss) / 2

        return loss
