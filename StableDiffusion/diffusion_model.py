import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.context_dim = context_dim if context_dim is not None else embed_dim
        self.self_attention = context_dim is None

        self.Q = nn.Linear(self.hidden_dim, self.embed_dim, bias=False)
        self.K = nn.Linear(self.context_dim, self.embed_dim, bias=False)
        self.V = nn.Linear(self.context_dim, self.embed_dim, bias=False)
        self.out = nn.Linear(self.embed_dim, self.hidden_dim)

    def forward(self, tokens, t=None, context=None):
        B, T, _ = tokens.shape
        H = self.num_heads
        Q = self.Q(tokens).view(B, T, H, self.head_dim).transpose(1, 2)

        if self.self_attention:
            K = self.K(tokens).view(B, T, H, self.head_dim).transpose(1, 2)
            V = self.V(tokens).view(B, T, H, self.head_dim).transpose(1, 2)
        else:
            _, context_len, context_C = context.shape
            if context_C != self.context_dim:
                context = nn.Linear(context_C, self.context_dim).to(context.device)(context)
                context_C = self.context_dim
            K = self.K(context).view(B, context_len, H, self.head_dim).transpose(1, 2)
            V = self.V(context).view(B, context_len, H, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_scores = F.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention_scores, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        out = self.out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_heads, use_self_attention=False, use_cross_attention=False):
        super().__init__()
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.self_attention = Attention(hidden_dim, hidden_dim, num_heads=num_heads)
        self.cross_attention = Attention(hidden_dim, hidden_dim, context_dim=context_dim, num_heads=num_heads)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)

        self.feed_forward1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1),
        )
        self.feed_forward2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x, t=None, context=None):
        if self.use_self_attention:
            x = self.self_attention(self.norm1(x)) + x
            x = self.feed_forward1(self.norm2(x)) + x
        if self.use_cross_attention:
            x = self.cross_attention(self.norm3(x), context=context) + x
            x = self.feed_forward2(self.norm4(x)) + x
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, hidden_dim, context_dim=512, num_heads=4, use_self_attention=False, use_cross_attention=False):
        super().__init__()
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.transformer = TransformerBlock(hidden_dim, context_dim, num_heads, use_self_attention, use_cross_attention)
        self.context_proj = nn.Linear(context_dim, hidden_dim) if context_dim != hidden_dim else nn.Identity()

    def forward(self, x, t=None, context=None):
        _, _, h, w = x.shape
        residual = x
        x = rearrange(x, "b c h w -> b (h w) c")

        if context is not None:
            context = self.context_proj(context)

        x = self.transformer(x, t, context)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x + residual


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.activation = nn.SiLU()
        self.layer1 = nn.Sequential(
            nn.GroupNorm(4, in_channels, eps=1e-6),
            self.activation,
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.GroupNorm(4, out_channels, eps=1e-6),
            self.activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(0.1)
        self.time_proj = nn.Linear(time_dim, out_channels)

    def forward(self, x, t):
        residual = self.residual(x)
        x = self.layer1(x)
        x = self.time_proj(self.activation(t))[:, :, None, None] + x
        x = self.dropout(x)
        x = self.layer2(x)
        return x + residual


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, use_self_attention=False, use_cross_attention=False, num_heads=1, context_dim=512):
        super().__init__()
        self.resnet1 = ResNet(in_channels, out_channels, time_dim)
        self.resnet2 = ResNet(out_channels, out_channels, time_dim)
        self.transformer1 = SpatialTransformer(out_channels, context_dim, num_heads, use_self_attention=True) if use_self_attention else None
        self.transformer2 = SpatialTransformer(out_channels, context_dim, num_heads, use_cross_attention=True) if use_cross_attention else None
        self.down_sample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t, y):
        x = self.resnet1(x, t)
        if self.transformer1:
            x = self.transformer1(x, t, y)
        x = self.resnet2(x, t)
        if self.transformer2:
            x = self.transformer2(x, t, y)
        x = self.down_sample(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, use_self_attention=False, use_cross_attention=False, num_heads=1, context_dim=512):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.resnet1 = ResNet(out_channels, out_channels, time_dim)
        self.resnet2 = ResNet(out_channels, out_channels, time_dim)
        self.transformer1 = SpatialTransformer(out_channels, context_dim, num_heads, use_self_attention=True) if use_self_attention else None
        self.transformer2 = SpatialTransformer(out_channels, context_dim, num_heads, use_cross_attention=True) if use_cross_attention else None

    def forward(self, x, t, y):
        x = self.up_sample(x)
        x = self.resnet1(x, t)
        if self.transformer1:
            x = self.transformer1(x, t, y)
        x = self.resnet2(x, t)
        if self.transformer2:
            x = self.transformer2(x, t, y)
        return x


class Middle(nn.Module):
    def __init__(self, channels, time_dim, context_dim):
        super().__init__()
        self.resnet1 = ResNet(channels, channels, time_dim)
        self.resnet2 = ResNet(channels, channels, time_dim)
        self.transformer = SpatialTransformer(channels, context_dim, num_heads=channels // 64, use_self_attention=True, use_cross_attention=True)

    def forward(self, x, t, context):
        x = self.resnet1(x, t)
        x = self.transformer(x, t, context)
        x = self.resnet2(x, t)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, time_dim=256, context_dim=512):
        super().__init__()
        self.time_dim = time_dim
        self.context_dim = context_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.down1 = self._down(64, 128, time_dim)
        self.down2 = self._down(128, 256, time_dim, use_self_attention=True, use_cross_attention=False, num_heads=4, context_dim=context_dim)
        self.down3 = self._down(256, 512, time_dim, use_self_attention=True, use_cross_attention=False, num_heads=8, context_dim=context_dim)
        self.middle = Middle(512, time_dim, context_dim)
        self.up1 = self._up(512, 256, time_dim, use_self_attention=True, use_cross_attention=True, num_heads=8, context_dim=context_dim)
        self.up2 = self._up(256 + 256, 128, time_dim, use_self_attention=True, use_cross_attention=True, num_heads=4, context_dim=context_dim)
        self.up3 = self._up(128 + 128, 64, time_dim)
        self.out = nn.Sequential(
            ResNet(64 * 2, 64, time_dim),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
        )

    def get_sin_position_embedding(self, t, embedding_dim):
        half = embedding_dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb

    def _down(self, in_channels, out_channels, time_dim, use_self_attention=False, use_cross_attention=False, num_heads=1, context_dim=None):
        return DownSample(in_channels, out_channels, time_dim, use_self_attention, use_cross_attention, num_heads, context_dim or self.context_dim)

    def _up(self, in_channels, out_channels, time_dim, use_self_attention=False, use_cross_attention=False, num_heads=1, context_dim=None):
        return UpSample(in_channels, out_channels, time_dim, use_self_attention, use_cross_attention, num_heads, context_dim or self.context_dim)

    def forward(self, x, t, y):
        residual = x
        if y.dim() == 2:
            y.unsqueeze(1)
        t = self.get_sin_position_embedding(t, self.time_dim)
        t = self.time_mlp(t)

        x1 = self.conv(x)
        x2 = self.down1(x1, t, y)
        x3 = self.down2(x2, t, y)
        x4 = self.down3(x3, t, y)
        x4 = self.middle(x4, t, y)
        x = self.up1(x4, t, y)
        x = torch.cat([x, x3], dim=1)   # skip connection
        x = self.up2(x, t, y)
        x = torch.cat([x, x2], dim=1)   # skip connection
        x = self.up3(x, t, y)
        x = torch.cat([x, x1], dim=1)   # skip connection
        x = self.out[0](x, t)
        for layer in self.out[1:]:
            x = layer(x)
        return x + residual


class NoiseScheduler(nn.Module):
    def __init__(self, T, device):
        super().__init__()
        self.T = T
        self.device = device
        self.beta = NoiseScheduler.cos_beta_schedule(T).to(self.device)
        self.alpha = (1. - self.beta).to(self.device)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(self.device)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar).to(self.device)
        self.sqrt_1_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar).to(self.device)

    @staticmethod
    def cos_beta_schedule(t, s=8e-3):
        steps = t + 1
        x = torch.linspace(0, t, steps)
        alpha_bar = torch.cos(((x / t) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clip(beta, 0.0001, 0.9999)

    def add_noise(self, x0, t):
        t = t.clone().detach().long().to(self.sqrt_alpha_bar.device)
        epsilon = torch.randn_like(x0)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_1_minus_alpha_bar_t = self.sqrt_1_minus_alpha_bar[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar_t * x0 + sqrt_1_minus_alpha_bar_t * epsilon
        return x_t, epsilon


@torch.no_grad()
def sample(model, x_t, noise_scheduler, t, text_embedding):
    t = t.to(x_t.device)
    beta_t = noise_scheduler.beta[t]
    alpha_t = noise_scheduler.alpha[t]
    alpha_bar_t = noise_scheduler.alpha_bar[t]
    noise_pred = model(x_t, t, text_embedding)
    noise = torch.randn_like(x_t).to(x_t.device) if t > 0 else torch.zeros_like(x_t).to(x_t.device)
    return (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / (torch.sqrt(1- alpha_bar_t))) * noise_pred) + torch.sqrt(beta_t) * noise