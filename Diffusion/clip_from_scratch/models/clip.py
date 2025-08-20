import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from configs import VisualConfig, TextConfig


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        nn.GELU()(x)
        return x * torch.sigmoid(1.702 * x)


class CLIP(nn.Module):
    def __init__(self, embed_dim: int, visual_config: VisualConfig, text_config: TextConfig,
                 quick_gelu: bool = True, logit_scale: bool = False, init_logit_scale: float = np.log(1 / 0.07),
                 init_logit_bias: Optional[float] = None, output_dict: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.quick_gelu = quick_gelu
        self.visual_config = visual_config
        self.text_config = text_config
        self.visual_model = self._build_visual_model(visual_config)
        self.text_model = self._build_text_model(text_config)

        logit_shape = [1] if logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(logit_shape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(logit_shape) * init_logit_bias)
        else:
            self.logit_bias = None
        self.output_dict = output_dict

    def _build_visual_model(self, visual_config: VisualConfig):
        if isinstance(visual_config.layers, (tuple, list)):
            from resnet import ResNet
            vision_heads = visual_config.dim * 32 // visual_config.head_width
            return ResNet(
                layers=visual_config.layers,
                output_dim=self.embed_dim,
                heads=vision_heads,
                image_size=visual_config.image_size,
                width=visual_config.dim,
            )
        else:
            from transformers import ViT
            vision_heads = visual_config.dim // visual_config.head_width
            return ViT(
                image_size=visual_config.image_size,
                patch_size=visual_config.patch_size,
                dim=visual_config.dim,
                layers=visual_config.layers,
                heads=vision_heads,
                fc_ratio=visual_config.fc_ratio,
                scale=visual_config.scale_factor,
                output_dim=self.embed_dim,
                patch_dropout=visual_config.patch_dropout,
                ln_pre=visual_config.ln_pre,
                ls_init_value=visual_config.ls_init_value,
                attn_pool=visual_config.attn_pool,
                attn_pool_heads=visual_config.attn_pool_heads,
                attn_pool_queries=visual_config.attn_pool_queries,
                global_pool=visual_config.global_pool,
                return_tokens=visual_config.return_tokens,
            )

    def _build_text_model(self, text_config: TextConfig):
        from .transformers import TextTransformer
        activation = QuickGELU if self.quick_gelu else nn.GELU
        return TextTransformer(
            seq_len=text_config.seq_len,
            vocab_size=text_config.vocab_size,
            dim=text_config.dim,
            heads=text_config.heads,
            layers=text_config.layers,
            fc_ratio=text_config.fc_ratio,
            output_dim=self.embed_dim,
            ls_init_value=text_config.ls_init_value,
            pad_id=text_config.pad_id,
            eos_id=text_config.eos_id,
            pool_type=text_config.pool_type,
            proj_type=text_config.proj_type,
            proj_bias=text_config.proj_bias,
            activation=activation,
            embed_cls=text_config.embed_cls,
            casual_mask=text_config.casual_mask,
            use_pad_mask=text_config.use_pad_mask,
            return_tokens=text_config.return_tokens,
        )

    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = self.visual_model(image)
        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = self.text_model(text)
        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None
    ) -> Optional[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            output_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
            }
            if self.logit_bias is not None:
                output_dict["logit_bias"] = self.logit_bias
            return output_dict
        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        else:
            return image_features, text_features, self.logit_scale.exp()


def ViT_S_16() -> CLIP:
    embed_dim = 384
    visual_config = VisualConfig(
        image_size=224,
        layers=12,
        dim=384,
        patch_size=16,
    )
    text_config = TextConfig(
        seq_len=77,
        vocab_size=49408,
        dim=384,
        heads=6,
        layers=12,
    )
    return CLIP(embed_dim, visual_config, text_config)


def ViT_S_32() -> CLIP:
    embed_dim = 384
    visual_config = VisualConfig(
        image_size=224,
        layers=12,
        dim=384,
        patch_size=32,
    )
    text_config = TextConfig(
        seq_len=77,
        vocab_size=49408,
        dim=384,
        heads=6,
        layers=12,
    )
    return CLIP(embed_dim, visual_config, text_config)


def ViT_M_16() -> CLIP:
    embed_dim = 512
    visual_config = VisualConfig(
        image_size=224,
        layers=12,
        dim=512,
        patch_size=16,
    )
    text_config = TextConfig(
        seq_len=77,
        vocab_size=49408,
        dim=512,
        heads=8,
        layers=12,
    )
    return CLIP(embed_dim, visual_config, text_config)


def ViT_M_32() -> CLIP:
    embed_dim = 512
    visual_config = VisualConfig(
        image_size=224,
        layers=12,
        dim=512,
        patch_size=32,
    )
    text_config = TextConfig(
        seq_len=77,
        vocab_size=49408,
        dim=512,
        heads=8,
        layers=12,
    )
    return CLIP(embed_dim, visual_config, text_config)


def ViT_B_16() -> CLIP:
    embed_dim = 512
    visual_config = VisualConfig(
        image_size=224,
        layers=12,
        dim=768,
        patch_size=16,
    )
    text_config = TextConfig(
        seq_len=77,
        vocab_size=49408,
        dim=512,
        heads=8,
        layers=12,
    )
    return CLIP(embed_dim, visual_config, text_config)


def ViT_B_32() -> CLIP:
    embed_dim = 512
    visual_config = VisualConfig(
        image_size=224,
        layers=12,
        dim=768,
        patch_size=32,
    )
    text_config = TextConfig(
        seq_len=77,
        vocab_size=49408,
        dim=512,
        heads=8,
        layers=12,
    )
    return CLIP(embed_dim, visual_config, text_config)


def ViT_L_14() -> CLIP:
    embed_dim = 768
    visual_config = VisualConfig(
        image_size=224,
        layers=24,
        dim=1024,
        patch_size=14,
    )
    text_config = TextConfig(
        seq_len=77,
        vocab_size=49408,
        dim=768,
        heads=12,
        layers=12,
    )
    return CLIP(embed_dim, visual_config, text_config)


def ViT_L_16() -> CLIP:
    embed_dim = 768
    visual_config = VisualConfig(
        image_size=224,
        layers=24,
        dim=1024,
        patch_size=16,
    )
    text_config = TextConfig(
        seq_len=77,
        vocab_size=49408,
        dim=768,
        heads=12,
        layers=12,
    )
    return CLIP(embed_dim, visual_config, text_config)


def ViT_H_14() -> CLIP:
    embed_dim = 1024
    visual_config = VisualConfig(
        image_size=224,
        layers=32,
        dim=1280,
        head_width=80,
        patch_size=14,
    )
    text_config = TextConfig(
        seq_len=77,
        vocab_size=49408,
        dim=1024,
        heads=16,
        layers=24,
    )
    return CLIP(embed_dim, visual_config, text_config)


def ViT_H_16() -> CLIP:
    embed_dim = 1024
    visual_config = VisualConfig(
        image_size=224,
        layers=32,
        dim=1280,
        head_width=80,
        patch_size=16,
    )
    text_config = TextConfig(
        seq_len=77,
        vocab_size=49408,
        dim=1024,
        heads=16,
        layers=24,
    )
    return CLIP(embed_dim, visual_config, text_config)


def ResNet_50() -> CLIP:
    embed_dim = 1024
    visual_config = VisualConfig(
        image_size=224,
        layers=[3, 4, 6, 3],
        dim=64,
    )
    text_config = TextConfig(
        seq_len=77,
        vocab_size=49408,
        dim=512,
        heads=8,
        layers=12,
    )
    return CLIP(embed_dim, visual_config, text_config)


def ResNet_101() -> CLIP:
    embed_dim = 512
    visual_config = VisualConfig(
        image_size=224,
        layers=[3, 4, 23, 3],
        dim=64,
    )
    text_config = TextConfig(
        seq_len=77,
        vocab_size=49408,
        dim=512,
        heads=8,
        layers=12,
    )
    return CLIP(embed_dim, visual_config, text_config)
