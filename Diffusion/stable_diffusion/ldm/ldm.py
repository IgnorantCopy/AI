import torch
import torch.nn as nn


from Diffusion.stable_diffusion.vae.vae import VAE
from Diffusion.stable_diffusion.ddpm.ddpm import UNet, NoiseScheduler


class StableDiffusion(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 4,
                 image_size: int = 512, timesteps: int = 1000, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.vae = VAE(in_channels=in_channels, latent_dim=latent_dim, image_size=image_size)
        self.unet = UNet(in_channels=latent_dim)
        self.noise_scheduler = NoiseScheduler(T=timesteps, device=device)

    def encode(self, x):
        return self.vae.encode(x)[0]

    def decode(self, z):
        return self.vae.decode(z)

    def diffuse(self, latents, t, context):
        return self.unet(latents, t, context)

    def forward(self, latents, t, context):
        return self.diffuse(latents, t, context)

    def load_vae(self, path):
        self.vae.load_state_dict(torch.load(path))

    def load_diffusion(self, path):
        self.unet.load_state_dict(torch.load(path))

    def freeze_vae(self):
        for param in self.vae.parameters():
            param.requires_grad = False

    def sample(self, context, latent_size=64, batch_size=1, guidance_scale=3.0):
        latents = torch.randn(batch_size, self.vae.latent_dim, latent_size, latent_size).to(self.device)
        unconditional_embeddings = torch.zeros_like(context)    # for classifier-free guidance

        for t in reversed(range(self.noise_scheduler.T)):
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=self.device)

            noise_pred_unconditional = self.diffuse(latents, t_batch, unconditional_embeddings)
            noise_pred_conditional = self.diffuse(latents, t_batch, context)
            noise_pred = noise_pred_unconditional + guidance_scale * (noise_pred_conditional - noise_pred_unconditional)

            alpha_t = self.noise_scheduler.alpha[t]
            alpha_t_bar = self.noise_scheduler.alpha_bar[t]
            beta_t = self.noise_scheduler.beta[t]

            noise = torch.randn_like(latents) if t > 0 else torch.zeros_like(latents)

            latents = (1 / torch.sqrt(alpha_t)) * (latents - ((1 - alpha_t) / (torch.sqrt(1 - alpha_t_bar))) * noise_pred) + torch.sqrt(beta_t) * noise
            latents = torch.clamp(latents, -1, 1)

        return latents


class DDIMSampler:
    def __init__(self, model: StableDiffusion, n_steps: int = 50, device: str = 'cuda'):
        self.model = model
        self.n_steps = n_steps
        self.device = device

    @torch.no_grad()
    def sample(self, noise, context, guidance_scale=3.0):
        scheduler = self.model.noise_scheduler
        x = noise
        for i in reversed(range(0, scheduler.T, scheduler.T // self.n_steps)):
            t = torch.full((noise.shape[0],), i, dtype=torch.long, device=self.device)

            noise_pred_unconditional = self.model.unet(x, t, y=torch.zeros_like(context))
            noise_pred_conditional = self.model.unet(x, t, y=context)
            noise_pred = noise_pred_unconditional + guidance_scale * (noise_pred_conditional - noise_pred_unconditional)

            alpha_bar_t = scheduler.alpha_bar[t]
            beta_t = scheduler.beta[t]
            alpha_bar_t_prev = scheduler.alpha_bar[t - 1] if t > 0 else torch.ones_like(alpha_bar_t)

            x_0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            x_t = torch.sqrt(1 - alpha_bar_t_prev - beta_t) * noise_pred
            x = torch.sqrt(alpha_bar_t_prev) * x_0 + x_t

        return x
