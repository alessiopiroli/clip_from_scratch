import logging
import os
import time

import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return edict(config)


def setup_logging(config):
    base_dir = config.LOGGING.logging_dir
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(base_dir, f"exp_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    log_file = os.path.join(experiment_dir, "log.log")
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    tb_dir = os.path.join(experiment_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    logger.info(f"Experiment logs saved to: {experiment_dir}")

    return logger, writer, experiment_dir


def get_device(logger):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Device: {device}")

    return device


def log_model(cfg, logger, model, batch_size, device):
    imgs = torch.rand(batch_size, 3, cfg.DATA.img_size, cfg.DATA.img_size, device=device)
    captions = torch.randint(0, cfg.TOKENIZER.vocab_size + 4, (batch_size, cfg.MODEL.GPT.context_size), device=device)

    model_summary = summary(model, input_data=[imgs, captions], verbose=0)

    logger.info(f"Model summary:\n{model_summary}")


def log_epoch(logger, writer, epoch, n_epochs, train_loss, val_loss, experiment_dir, model):
    logger.info(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    writer.add_scalars("loss", {"train": train_loss, "validation": val_loss}, epoch + 1)

    save_path = os.path.join(experiment_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)


def denormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    image = image.cpu() * std + mean
    image = image.clamp(0, 1)

    return image.permute(1, 2, 0).numpy()


def decode_caption(token_ids, itos, special_tokens=("<BOS>", "<EOS>", "<PAD>")):
    tokens = [itos[int(t)] for t in token_ids]
    tokens = [t for t in tokens if t not in special_tokens]

    return "".join(tokens).strip()


def visualize_top_k(dataset, similarities, caption, experiment_dir, logger, k=5, file_name="top_k.png"):
    import matplotlib.pyplot as plt

    top_indices = sorted(similarities, key=similarities.get, reverse=True)[:k]

    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3.5), squeeze=False)
    fig.suptitle(f"Caption: '{caption}'")

    for ax, idx in zip(axes[0], top_indices):
        image, _ = dataset[idx]
        ax.imshow(denormalize(image))
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(experiment_dir, file_name)
    fig.savefig(save_path)
    plt.close(fig)

    logger.info(f"Saved top-{k} visualization to {save_path}")
