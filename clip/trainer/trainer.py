import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip.dataset.flicker8k_dataset import Flicker8kDataset, collate_fn
from clip.loss.clip_loss import CLIPLoss
from clip.model.clip_model import CLIP
from clip.utils.misc import get_device, log_epoch, log_model, setup_logging


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_epochs = self.cfg.TRAINER.n_epochs
        self.batch_size = self.cfg.TRAINER.batch_size
        self.lr = float(self.cfg.TRAINER.lr)
        self.logger, self.writer, self.experiment_dir = setup_logging(self.cfg)
        self.build_model()
        trainable_params = [p for p in self.clip.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)
        self.loss_fn = CLIPLoss()
        self.build_dataloaders()
        self.training_steps = 1
        self.validation_steps = 1

    def build_model(self):
        self.device = get_device(self.logger)
        self.clip = CLIP(self.cfg).to(self.device)
        log_model(self.cfg, self.logger, self.clip, self.batch_size, self.device)

    def build_dataloaders(self):
        if self.cfg.DATA.dataset == "flickr":
            self.train_dataset = Flicker8kDataset(self.cfg, split="train")
            self.val_dataset = Flicker8kDataset(self.cfg, split="dev")

        self.train_loader = DataLoader(
            self.train_dataset, self.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, self.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False
        )

    def train(self):
        for epoch in range(self.n_epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)

            log_epoch(
                self.logger, self.writer, epoch, self.n_epochs, train_loss, val_loss, self.experiment_dir, self.clip
            )

    def train_one_epoch(self, epoch):
        self.clip.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Training epoch {epoch+1}")

        for images, captions in pbar:
            self.optimizer.zero_grad()
            images, captions = images.to(self.device), captions.to(self.device)
            img_proj, text_proj, temp = self.clip(images, captions)
            loss = self.loss_fn.get_loss(img_proj, text_proj, temp)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self.writer.add_scalar("Training-Loss-Step", loss.item(), self.training_steps)
            self.training_steps += 1

        return total_loss / len(self.train_loader)

    def validate_one_epoch(self, epoch):
        self.clip.eval()
        total_loss = 0.0
        pbar = tqdm(self.val_loader, f"Validating epoch {epoch+1}")

        with torch.no_grad():
            for images, captions in pbar:
                images, captions = images.to(self.device), captions.to(self.device)
                img_proj, text_proj, temp = self.clip(images, captions)
                loss = self.loss_fn.get_loss(img_proj, text_proj, temp)
                total_loss += loss.item()
                self.writer.add_scalar("Validation-Loss-Step", loss.item(), self.validation_steps)
                self.validation_steps += 1

        return total_loss / len(self.val_loader)
