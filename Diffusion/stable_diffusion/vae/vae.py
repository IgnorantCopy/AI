import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 4, image_size: int = 512):
        """
        输入：[in_channels, image_size, image_size]
        :param in_channels: 通道数
        :param latent_dim: 潜在空间维度
        :param image_size: 图像大小
        """
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Encoder: [3, 512, 512] -> [4, 64, 64]
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, 64),  # [64, 256, 256]
            self._conv_block(64, 128),  # [128, 128, 128]
            self._conv_block(128, 256),  # [256, 64, 64]
        )
        self.mu = nn.Conv2d(256, latent_dim, kernel_size=1)
        self.log_var = nn.Conv2d(256, latent_dim, kernel_size=1)

        # Decoder: [4, 64, 64] -> [3, 512, 512]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=1),  # [256, 64, 64]
            self._conv_transpose_block(256, 128),  # [128, 128, 128]
            self._conv_transpose_block(128, 64),  # [64, 256, 256]
            self._conv_transpose_block(64, in_channels)  # [3, 512, 512]
        )

    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            # nn.GroupNorm(num_groups=1, num_channels=out_channels),  # layer norm < group norm < batch norm
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    @staticmethod
    def _conv_transpose_block(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def encode(self, x):
        result = self.encoder(x)
        return self.mu(result), self.log_var(result)

    def decode(self, z):
        result = self.decoder(z)
        result = nn.Tanh()(result)
        # return result.view(-1, self.in_channels, self.image_size, self.image_size)
        return result

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        :param x: input image
        :return: reconstruction, x, mu, log_var
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)  # 重参数化，得到潜在空间的表示 [4, 64, 64]
        return self.decoder(z), x, mu, log_var