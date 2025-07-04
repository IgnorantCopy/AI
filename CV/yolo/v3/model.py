import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, stride: int, mode: str = "nearest"):
        super().__init__()
        self.stride = stride
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.stride, mode=self.mode)


class YOLO(nn.Module):
    def __init__(self, anchors: list, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.output_per_anchor = self.num_classes + 5
        self.grid = torch.zeros(1)

        anchors = torch.tensor(anchors, dtype=torch.float32)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid',
            anchors.clone().view(1, -1, 1, 1, 2)    # [1, num_anchors, 1, 1, 2]
        )
        self.stride = None

    def forward(self, x: torch.Tensor, image_size: int) -> torch.Tensor:
        b, _, ny, nx = x.shape
        self.stride = image_size // ny
        # [b, 255, 20, 20] -> [b, 3, 20, 20, 85]
        x = x.view(b, self.num_anchors, self.output_per_anchor, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)
            x[..., 0:2] = (x[..., 0:2] + self.grid) * self.stride   # x, y
            x[..., 2:4] = x[..., 2:4] ** 2 * (4 * self.anchor_grid) # w, h
            x[..., 4:]  = x[..., 4:].sigmoid()                      # confidence, cls
            x = x.view(b, -1, self.output_per_anchor)

        return x

    @staticmethod
    def _make_grid(nx: int, ny: int) -> torch.Tensor:
        y, x = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack([y, x], dim=2).view((1, 1, ny, nx, 2)).float()


class DarkNet(nn.Module):
    def __init__(self, in_channels: int, model_config: list):
        super().__init__()
        self.in_channels = in_channels
        self.model_config = model_config
        self.module_list = self._make_layer()
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], YOLO)]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_size = x.shape[2]
        layer_outputs, yolo_outputs = [], []
        for i, (layer_config, module) in enumerate(zip(self.model_config, self.module_list)):
            if layer_config["name"] in ["conv", "upsample"]:
                x = module(x)
            elif layer_config["name"] == "route":


    def _make_layer(self):
        channels = [self.in_channels]
        module_list = nn.ModuleList()
        for i, layer_config in enumerate(self.model_config):
            module = nn.Sequential()
            if layer_config["name"] == "conv":
                batch_norm      = layer_config["batch_norm"]
                out_channels    = layer_config["out_channels"]
                kernel_size     = layer_config["kernel_size"]
                stride          = layer_config["stride"]
                padding         = layer_config["padding"]
                activation      = layer_config["activation"]
                assert padding == (kernel_size - 1) // 2, "Padding should be same as kernel_size // 2"
                module.add_module(
                    f"conv{i+1}",
                    nn.Conv2d(
                        in_channels=channels[-1],
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=not batch_norm,
                    )
                )
                if batch_norm:
                    module.add_module(
                        f"bn{i+1}",
                        nn.BatchNorm2d(out_channels)
                    )
                if activation == "leaky":
                    module.add_module(f"leaky_{i+1}", nn.LeakyReLU(0.1))
                elif activation == "logistic":
                    module.add_module(f"sigmoid_{i+1}", nn.Sigmoid())
            elif layer_config["name"] == "upsample":
                stride = int(layer_config["stride"])
                module.add_module(f"upsample_{i+1}", Upsample(stride=stride))
            elif layer_config["name"] == "route":
                layers = layer_config["layers"]
                out_channels = sum([channels[1:][i] for i in layers])
                module.add_module(f"route_{i+1}", nn.Sequential())  # placeholder
            elif layer_config["name"] == "shortcut":
                from_layer = layer_config["from"]
                out_channels = channels[1:][from_layer]
                module.add_module(f"shortcut_{i+1}", nn.Sequential())   # placeholder
            elif layer_config["name"] == "yolo":
                anchors = layer_config["anchors"]
                num_classes = layer_config["classes"]
                module.add_module( f"yolo_{i+1}", YOLO(anchors, num_classes))
            module_list.append(module)
            channels.append(out_channels)

        return module_list


