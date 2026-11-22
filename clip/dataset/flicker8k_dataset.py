import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Flicker8kDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.root_dir = Path(self.cfg.DATA.root_dir)
        self.img_dir = Path(self.cfg.DATA.img_dir)
        self.img_size = self.cfg.DATA.img_size

        if split not in ["train", "dev", "test"]:
            raise ValueError(f"Split is not correct.")

        self.captions_file = self.load_captions_file()
        self.image_names = self.load_image_names()
        self.vocabulary, self.merges = self.load_tokenizer_data()

        self.initialize_stoi()
        self.initialize_itos()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if split == "train":
            self.transform_image = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.img_size, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform_image = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.CenterCrop(self.img_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    def initialize_itos(self):
        self.itos = {}
        self.itos[0] = "<BOS>"
        self.itos[1] = "<EOS>"
        self.itos[2] = "<PAD>"
        self.itos[3] = "<UNK>"

        for i, tok in enumerate(self.vocabulary):
            self.itos[i + 4] = tok

    def initialize_stoi(self):
        self.stoi = {}
        self.stoi["<BOS>"] = 0
        self.stoi["<EOS>"] = 1
        self.stoi["<PAD>"] = 2
        self.stoi["<UNK>"] = 3

        for i, tok in enumerate(self.vocabulary):
            self.stoi[tok] = i + 4

    def load_tokenizer_data(self):
        tokenizer_data_file = Path(self.cfg.TOKENIZER.tokenizer_data_file)
        tokenizer_data_file = Path.joinpath(self.root_dir, tokenizer_data_file)

        if not tokenizer_data_file.is_file():
            raise FileNotFoundError(f"Could not find {tokenizer_data_file}")

        with open(tokenizer_data_file, "r") as f:
            tokenizer_data = json.load(f)

        vocabulary, merges = tokenizer_data["vocabulary"], tokenizer_data["merges"]

        return vocabulary, merges

    def load_captions_file(self):
        captions_file_name = Path(f"{self.split}_data.json")
        file_path = Path.joinpath(self.root_dir, captions_file_name)

        if not file_path.is_file():
            raise ValueError(f"{file_path} does not contain the data file")

        with open(file_path, "r") as f:
            data_file = json.load(f)

        return data_file

    def load_image_names(self):
        images_file_name = Path(f"Flickr_8k.{self.split}Images.txt")
        file_path = Path.joinpath(self.root_dir, images_file_name)

        if not file_path.is_file():
            raise ValueError(f"{file_path} does not contain the data file")

        with open(file_path, "r") as f:
            image_names = f.read().split()

        return image_names

    def preprocess_caption(self, caption):
        new_caption = []
        new_caption.append("<BOS>")

        for token in caption:
            new_caption.append(token)

        new_caption.append("<EOS>")
        new_caption = self.encode(new_caption)

        return new_caption

    def encode(self, caption):
        encoded_caption = []
        unk_id = self.stoi["<UNK>"]

        for token in caption:
            encoded_caption.append(self.stoi.get(token, unk_id))

        return encoded_caption

    def decode(self, caption, cleanup=False):
        s = "".join([self.itos[tok] for tok in caption])
        if cleanup:
            s = s.replace("<BOS>", "")
            s = s.replace("<EOS>", "")

        return s

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        captions = self.captions_file[image_name]
        caption = captions[random.randint(0, len(captions) - 1)]
        caption = self.preprocess_caption(caption)

        image_path = Path.joinpath(self.root_dir, self.img_dir, image_name)
        if not image_path.exists():
            raise FileNotFoundError(f"Could not find image {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform_image(image)

        return image, caption


def collate_fn(batch):
    images = [sample[0] for sample in batch]
    captions = [sample[1] for sample in batch]

    images = torch.stack(images)

    max_caption_length = max(len(c) for c in captions)
    B = len(captions)

    padded = torch.full((B, max_caption_length), fill_value=2, dtype=torch.long)

    for i, c in enumerate(captions):
        padded[i, : len(c)] = torch.tensor(c, dtype=torch.long)

    return images, padded
