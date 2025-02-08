import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
import os
from tqdm.auto import tqdm
from diffusion_model import UNet, NoiseScheduler, sample

# hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 64
in_channels = 3
epochs = 1000
batch_size = 32
lr = 1e-4
T = 1000
save_checkpoint = 100

# CLIP model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# dataset
dataset = load_dataset("svjack/pokemon-blip-captions-en-zh", split="train")
train_dataset = dataset.select(range(600))
val_dataset = dataset.select(range(600, 800))

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# models
diffusion_model = UNet(in_channels=in_channels).to(device)
noise_scheduler = NoiseScheduler(T, device)

optimizer = Adam(diffusion_model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=5e-5)

os.makedirs("./models/results", exist_ok=True)

for epoch in range(epochs):
    diffusion_model.train()
    bar = tqdm(total=len(train_dataset), desc=f"Epoch {epoch + 1}/{epochs}")
    train_loss = 0.
    for batch in train_dataset:
        images = batch['image'].to(device)
        text = batch['text']
        text_input = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_input["input_ids"].to(device)).last_hidden_state

        t = torch.randint(0, T, (images.shape[0],), device=device).long()
        noisy_images, noise = noise_scheduler(images, t)
        noise_pred = diffusion_model(noisy_images, t, text_embeddings)
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        bar.update(1)
        bar.set_postfix({"loss": loss.item()})

    diffusion_model.eval()
    val_loss = 0.
    with torch.no_grad():
        for batch in val_dataset:
            images = batch['image'].to(device)
            text = batch['text']
            text_input = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input["input_ids"].to(device)).last_hidden_state

            t = torch.randint(0, T, (images.shape[0],), device=device).long()
            noisy_images, noise = noise_scheduler(images, t)
            noise_pred = diffusion_model(noisy_images, t, text_embeddings)
            loss = F.mse_loss(noise_pred, noise)

            val_loss += loss.item()
    scheduler.step(val_loss)

    print(f"Epoch {epoch + 1}/{epochs}:\n\tTrain Loss: {train_loss / len(train_dataset)}\n\tVal Loss: {val_loss / len(val_dataset)}")

    if (epoch + 1) % save_checkpoint == 0:
        # save model
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": diffusion_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, f"./models/diffusion_{epoch + 1}.pth")
        print(f"Model saved at epoch {epoch + 1}...")
        # save images
        diffusion_model.eval()
        with torch.no_grad():
            sample_text = ["a red pokémon with a red fire tail"]
            text_input = tokenizer(sample_text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input["input_ids"].to(device)).last_hidden_state
            sampled_images = sample(diffusion_model, noise_scheduler, len(sample_text), in_channels, text_embeddings, image_size=image_size, device=device)
            for i, image in enumerate(sampled_images):
                image = image * 0.5 + 0.5
                image = image.detach().cpu().permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)
                image_pil = Image.fromarray(image)
                image_pil.save(f"./models/results/image_epoch{epoch + 1}_sample_{i}.png")
torch.save({
    "model": diffusion_model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
}, "./models/diffusion_final.pth")