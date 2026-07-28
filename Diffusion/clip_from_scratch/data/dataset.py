import os
import glob
import json
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset


class WebImgDataset(Dataset):
    def __init__(self, data_root: str, transform=None, tokenizer=None):
        super().__init__()
        self.image_paths = glob.glob(os.path.join(data_root, "images", "*.jpg"))
        self.caption_paths = glob.glob(os.path.join(data_root, "captions", "*.txt"))
        self.metadata_paths = glob.glob(os.path.join(data_root, "metadata", "*.json"))

        assert len(self.image_paths) == len(self.caption_paths) == len(self.metadata_paths)

        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(os.path.join(image_path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def _load_caption(self, idx):
        caption_path = self.caption_paths[idx]
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.readline()
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)
        if len(caption.shape) == 2:
            caption = caption.squeeze(0)
        return caption

    def _load_metadata(self, idx):
        metadata_path = self.metadata_paths[idx]
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata

    def __getitem__(self, idx):
        image = self._load_image(idx)
        caption = self._load_caption(idx)
        metadata = self._load_metadata(idx)

        return image, caption, metadata

    @staticmethod
    def collate_fn(batch):
        images, captions, metadata = zip(*batch)
        images = torch.stack(images, dim=0)
        captions = torch.stack(captions, dim=0)
        metadata = [m for m in metadata]
        return {
            "images": images,
            "captions": captions,
            "metadata": metadata,
        }
