import torch
from tqdm import tqdm

from clip.dataset.flicker8k_dataset import Flicker8kDataset
from clip.model.clip_model import CLIP
from clip.utils.misc import decode_caption, get_device, setup_logging, visualize_top_k


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state_dict_path = self.cfg.LOGGING.trained_model
        self.logger, self.writer, self.experiment_dir = setup_logging(self.cfg)
        self.build_model()
        self.dataset = Flicker8kDataset(self.cfg, split="test")

    def build_model(self):
        self.device = get_device(self.logger)
        self.clip = CLIP(self.cfg).to(self.device)
        state_dict = torch.load(self.state_dict_path, map_location=self.device)
        self.clip.load_state_dict(state_dict)
        self.clip.eval()

        self.vision_encoder = self.clip.vision_encoder
        self.text_encoder = self.clip.text_encoder
        self.image_proj = self.clip.image_proj
        self.text_proj = self.clip.text_proj

    @torch.no_grad()
    def evaluate(self):
        image_encodings = {}

        for i, (image, _) in tqdm(enumerate(self.dataset), total=len(self.dataset), desc="Encoding images"):
            image = image.to(self.device).unsqueeze(0)
            image_encoding = self.image_proj(self.vision_encoder(image))
            image_encoding = image_encoding / image_encoding.norm(dim=-1, keepdim=True)
            image_encodings[i] = image_encoding

        for j, (_, caption_ids) in tqdm(enumerate(self.dataset), total=len(self.dataset), desc="Evaluating captions"):
            caption_text = decode_caption(caption_ids, self.dataset.itos)
            caption = torch.tensor(caption_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            caption_encoding = self.text_proj(self.text_encoder(caption))
            caption_encoding = caption_encoding / caption_encoding.norm(dim=-1, keepdim=True)

            similarities = {}

            for i, image_encoding in image_encodings.items():
                similarity = caption_encoding @ image_encoding.T
                similarities[i] = similarity.item()

            visualize_top_k(
                self.dataset,
                similarities,
                caption_text,
                self.experiment_dir,
                self.logger,
                k=5,
                file_name=f"top_k_{j}.png",
            )
