import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List
from collections import OrderedDict


class Bottleneck(nn.Module):
    expansion = 4   # channel expansion factor

    def __init__(self, in_channels: int, hidden_channels: int, stride: int = 1):
        """
        Residual block with bottleneck design
        :param in_channels: input channels
        :param hidden_channels: hidden channels
        :param stride: stride
        """
        super().__init__()

        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.cbr2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv = nn.Conv2d(hidden_channels, hidden_channels * self.expansion, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride > 1 or in_channels != hidden_channels * self.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ('-1', nn.AvgPool2d(stride)),
                ('0', nn.Conv2d(in_channels, hidden_channels * self.expansion, kernel_size=1, stride=1, bias=False)),
                ('1', nn.BatchNorm2d(hidden_channels * self.expansion))
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.avg_pool(x)
        x = self.bn(self.conv(x))

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class AttentionPool2d(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int, heads: int, output_dim: int = None):
        """
        Attention pooling layer
        :param num_patches: image size // patch size
        :param embed_dim: image channels
        :param heads: number of heads
        :param output_dim: output dimension
        """
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.pos_embedding = nn.Parameter(torch.randn(num_patches ** 2 + 1, 1, embed_dim) * self.scale)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, output_dim or embed_dim)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b c h w -> (h w) b c')
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # [h * w + 1, b, c]
        x = x + self.pos_embedding.to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.fc.weight,
            out_proj_bias=self.fc.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ResNet(nn.Module):
    def __init__(self, layers: List[int], output_dim: int, heads: int,
                 in_channels: int = 3, image_size: int = 224, width: int = 64):
        """
        Modified ResNet model
        :param layers: depth of each residual block
        :param output_dim: output dimension for attention pooling
        :param heads: number of heads for attention pooling
        :param in_channels: channels of input image
        :param image_size: resolution of input image
        :param width: base dim of the network
        """
        super().__init__()
        self.output_dim = output_dim
        self.heads = heads
        self.in_channels = in_channels
        self.image_size = image_size
        self.width = width

        # stem
        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels, width // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width // 2),
            nn.ReLU(inplace=True),
        )
        self.cbr2 = nn.Sequential(
            nn.Conv2d(width // 2, width // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.cbr3 = nn.Sequential(
            nn.Conv2d(width // 2, width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AvgPool2d(2)

        # residual blocks
        self._in_channels = width
        self.layers = nn.ModuleList([self._make_layer(width, layers[0])])
        for i in range(1, len(layers)):
            self.layers.append(self._make_layer(width * 2 ** i, layers[i], stride=2))

        # head
        patch_size = 32
        embed_dim = width * 32
        self.attn_pool = AttentionPool2d(image_size // patch_size, embed_dim, heads, output_dim)

        self._init_weights()

    def _make_layer(self, hidden_channels: int, depth: int, stride: int = 1):
        layers = [Bottleneck(self._in_channels, hidden_channels, stride)]

        self._in_channels = hidden_channels * Bottleneck.expansion
        for _ in range(1, depth):
            layers.append(Bottleneck(self._in_channels, hidden_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        std = self.attn_pool.scale
        nn.init.normal_(self.attn_pool.q_proj.weight, std=std)
        nn.init.normal_(self.attn_pool.k_proj.weight, std=std)
        nn.init.normal_(self.attn_pool.v_proj.weight, std=std)
        nn.init.normal_(self.attn_pool.fc.weight, std=std)

        for layer in self.layers:
            for name, param in layer.named_parameters():
                if name.endswith("bn.weight"):
                    nn.init.zeros_(param)

    def stem(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        x = self.avg_pool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.attn_pool(x)

        return x