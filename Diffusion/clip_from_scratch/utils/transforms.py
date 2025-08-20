from typing import Tuple, Union
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image


class ResizeKeepRatio(nn.Module):
    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 interpolation: InterpolationMode = InterpolationMode.BICUBIC,
                 longest: int = 0,
                 ):
        """
        Resize the image to the given size while keeping the aspect ratio.
        :param size: target size of the image
        :param interpolation: interpolation mode
        :param longest: whether to use the longer side of the image to calculate the ratio
        """
        super().__init__()
        self.size = size if isinstance(size, tuple) else (size, size)
        self.interpolation = interpolation
        self.longest = longest

    def forward(self, img: Union[Image.Image, torch.Tensor]):
        _, h, w = F.get_dimensions(img)
        target_h, target_w = self.size
        ratio_h = target_h / h
        ratio_w = target_w / w
        ratio = max(ratio_h, ratio_w) * self.longest + min(ratio_h, ratio_w) * (1 - self.longest)
        size = [int(w * ratio), int(h * ratio)]
        return F.resize(img, size, self.interpolation)


class CropPad(nn.Module):
    def __init__(self, size: Union[int, Tuple[int, int]], fill=0):
        """
        Crop or pad the image to the given size.
        :param size: target size of the image
        :param fill: if padding, fill the image with this value
        """
        super().__init__()
        self.size = size if isinstance(size, tuple) else (size, size)
        self.fill = fill

    def forward(self, img: Union[Image.Image, torch.Tensor]):
        _, h, w = F.get_dimensions(img)
        target_h, target_w = self.size

        if h < target_h or w < target_w:
            padding_ltrb = [
                (target_w - w) // 2 if w < target_w else 0,
                (target_h - h) // 2 if h < target_h else 0,
                (target_w - w + 1) // 2 if w < target_w else 0,
                (target_h - h + 1) // 2 if h < target_h else 0,
            ]
            img = F.pad(img, padding_ltrb, fill=self.fill)
            _, h, w = F.get_dimensions(img)
            if target_h == h and target_w == w:
                return img

        top = int(round((h - target_h) / 2.0))
        left = int(round((w - target_w) / 2.0))
        return F.crop(img, top, left, target_h, target_w)
