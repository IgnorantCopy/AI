import random
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision.transforms import transforms


def resize(image: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)


class COCO(Dataset):
    def __init__(self, image_root: str, label_root: str, list_file: str, image_size=416, multiscale=True, scale_freq=10, transform=None):
        assert os.path.basename(image_root) == os.path.basename(label_root), \
            f"{os.path.basename(image_root)} != {os.path.basename(label_root)}"

        with open(list_file, "r") as file:
            lines = file.readlines()
            self.image_path = [os.path.join(image_root, line) for line in lines]

        self.label_path = []
        for path in self.image_path:
            name = os.path.basename(path).split(".")[0] + ".txt"
            name = os.path.join(label_root, name)
            if not os.path.exists(name):
                raise FileNotFoundError(name)
            self.label_path.append(name)

        self.image_size = image_size
        self.multiscale = multiscale
        self.scale_freq = scale_freq
        self.transform = transform
        self.batch_count = 0
        self.min_size = self.image_size - 3 * 32
        self.max_size = self.image_size + 3 * 32


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        image_path = self.image_path[item]
        image = Image.open(image_path).convert("RGB")
        image = transforms.ToTensor()(image)

        if len(image.shape) != 3:
            image = image.unsqueeze(0).expand((3, image.shape[1:]))

        label_path = self.label_path[item]
        boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

        if self.transform:
            image, boxes = self.transform((image, boxes))

        return image, boxes

    def collate_fn(self, batch):
        self.batch_count += 1

        images, boxes = list(zip(*batch))
        if self.multiscale and self.batch_count % self.scale_freq == 0:
            self.image_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # resize image
        images = torch.stack([resize(image, self.image_size) for image in images])
        # add index to boxes
        for i, box in enumerate(boxes):
            box[:, 0] = i
        boxes = torch.cat(boxes, dim=0)

        return images, boxes