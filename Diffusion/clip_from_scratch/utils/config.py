import os
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from typing import Tuple

from ..models import clip
from ..models.configs import VisualConfig, TextConfig
from .transforms import ResizeKeepRatio, CropPad
from .tokenizer import SimpleTokenizer
from Diffusion.clip_from_scratch.data.dataset import WebImgDataset
from .loss import CLIPLoss


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_optimizer(model: nn.Module, config) -> optim.Optimizer:
    train_config = config['train']
    lr = train_config['lr']
    optimizer_config = train_config['optimizer']
    name = optimizer_config["name"]
    if name == "Adam":
        weight_decay = getattr(optimizer_config, "weight_decay", 5e-4)
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "SGD":
        momentum = getattr(optimizer_config, "momentum", 0.9)
        weight_decay = getattr(optimizer_config, "weight_decay", 5e-4)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def get_scheduler(optimizer: optim.Optimizer, config) -> optim.lr_scheduler:
    train_config = config['train']
    scheduler_config = train_config['scheduler']
    name = scheduler_config["name"]
    if name == "ReduceLROnPlateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        factor = getattr(scheduler_config, "factor", 0.5)
        patience = getattr(scheduler_config, "patience", 10)
        min_lr = getattr(scheduler_config, "min_lr", 1e-6)
        mode = getattr(scheduler_config, "mode", "max")
        return ReduceLROnPlateau(optimizer, factor=factor, patience=patience, mode=mode, min_lr=min_lr)
    else:
        raise ValueError(f"Unsupported scheduler: {name}")


def get_loss(config) -> nn.Module:
    train_config = config['train']
    loss_config = train_config['loss']
    name = loss_config['loss']
    if name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif name == "BCELoss":
        return nn.BCELoss()
    elif name == "CLIPLoss":
        return CLIPLoss()
    else:
        raise ValueError(f"Unsupported loss: {name}")


def get_model(config) -> clip.CLIP:
    name = config["model_name"]
    if name == "ViT_S_16":
        return clip.ViT_S_16()
    elif name == "ViT_S_32":
        return clip.ViT_S_32()
    elif name == "ViT_M_16":
        return clip.ViT_M_16()
    elif name == "ViT_M_32":
        return clip.ViT_M_32()
    elif name == "ViT_B_16":
        return clip.ViT_B_16()
    elif name == "ViT_B_32":
        return clip.ViT_B_32()
    elif name == "ViT_L_14":
        return clip.ViT_L_14()
    elif name == "ViT_L_16":
        return clip.ViT_L_16()
    elif name == "ViT_H_14":
        return clip.ViT_H_14()
    elif name == "ViT_H_16":
        return clip.ViT_H_16()
    elif name == "ResNet_50":
        return clip.ResNet_50()
    elif name == "ResNet_101":
        return clip.ResNet_101()
    else:
        raise ValueError(f"Unsupported model: {name}")


def get_transform(config: VisualConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    image_size = config.image_size
    train_transform = transforms.Compose([
        ResizeKeepRatio(image_size),
        CropPad(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    val_transform = transforms.Compose([
        ResizeKeepRatio(image_size),
        CropPad(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
        ])
    return train_transform, val_transform


def get_tokenizer(config: TextConfig):
    seq_len = config.seq_len
    return SimpleTokenizer(context_length=seq_len)


def get_data(config, model: clip.CLIP) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_root = config["data_root"]
    imagenet_root = config["imagenet_root"]
    train_config = config["train"]
    batch_size = train_config["batch_size"]
    num_workers = train_config["num_workers"]

    train_transform, val_transform = get_transform(model.visual_config)
    tokenizer = get_tokenizer(model.text_config)

    train_dataset = WebImgDataset(os.path.join(data_root, "train/filtered_captions.tsv"), train_transform, tokenizer)
    val_dataset = WebImgDataset(os.path.join(data_root, "val/filtered_captions.tsv"), val_transform, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    imagenet_val_dataset = ImageNet(imagenet_root, split='val', transform=val_transform)
    imagenet_val_dataloader = DataLoader(imagenet_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, imagenet_val_dataloader