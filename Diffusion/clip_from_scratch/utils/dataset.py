import pandas as pd
from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import torch
import numpy as np


class TsvDataset(Dataset):
    def __init__(self, tsv_file: str, transform=None, tokenizer=None):
        super().__init__()
        self.data = pd.read_csv(tsv_file, sep='\t', names=["caption", "path"])
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.data["caption"][idx]
        path = self.data["path"][idx]
        image = Image.open(os.path.join(path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)
        return image, caption