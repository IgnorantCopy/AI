import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
import os
from tqdm.auto import tqdm
from ddpm import UNet, NoiseScheduler, sample

# hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 64
in_channels = 3
epochs = 1000
batch_size = 32
lr = 1e-4
T = 1000
save_checkpoint = 100

def transform_all(data):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    images = [transform(image.convert('RGB')) for image in data["image"]]
    en_texts = data["en_text"]
    return {"image": images, "en_text": en_texts}
# load dataset
dataset = load_dataset("svjack/pokemon-blip-captions-en-zh", split="train", cache_dir=r"D:\HuggingFace\cache")
dataset.set_transform(transform_all)
train_dataset = dataset.select(range(600))
val_dataset = dataset.select(range(600, 800))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

# load clip model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# log
save_dir = "../logs/ddpm"
os.makedirs(save_dir, exist_ok=True)

# define model
diffusion_model = UNet(in_channels=in_channels).to(device)
noise_scheduler = NoiseScheduler(T, device)

optimizer = Adam(diffusion_model.parameters(), lr=lr, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=5e-5)

# train
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    diffusion_model.train()
    train_bar = tqdm(total=len(train_loader), desc="Training")
    train_loss = 0.
    for batch in train_loader:
        images = batch['image'].to(device)
        text = batch['en_text']
        text_input = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_input["input_ids"].to(device)).last_hidden_state

        t = torch.randint(0, T, (images.shape[0],), device=device).long()
        noisy_images, noise = noise_scheduler.add_noise(images, t)
        noise_pred = diffusion_model(noisy_images, t, text_embeddings)
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_bar.update(1)
        train_bar.set_postfix({"loss": loss.item()})

    diffusion_model.eval()
    val_bar = tqdm(total=len(val_loader), desc="Validation")
    val_loss = 0.
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            text = batch['en_text']
            text_input = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input["input_ids"].to(device)).last_hidden_state

            t = torch.randint(0, T, (images.shape[0],), device=device).long()
            noisy_images, noise = noise_scheduler.add_noise(images, t)
            noise_pred = diffusion_model(noisy_images, t, text_embeddings)
            loss = F.mse_loss(noise_pred, noise)

            val_loss += loss.item()
            val_bar.update(1)
            val_bar.set_postfix({"loss": loss.item()})
    scheduler.step(val_loss)

    print(f"Epoch {epoch + 1}/{epochs}:\n\tTrain Loss: {train_loss / len(train_loader)}\n\tVal Loss: {val_loss / len(val_loader)}")

    if (epoch + 1) % save_checkpoint == 0:
        # save model
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": diffusion_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, os.path.join(save_dir, f"diffusion_{epoch + 1}.pth"))
        print(f"Model saved at epoch {epoch + 1}...")
        # save images
        diffusion_model.eval()
        with torch.no_grad():
            sample_text = ["a cartoon pikachu with big eyes and big ears"]
            text_input = tokenizer(sample_text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input["input_ids"].to(device)).last_hidden_state
            epsilon = torch.randn(len(sample_text), in_channels, image_size, image_size).to(device)
            sampled_images = sample(diffusion_model, epsilon, noise_scheduler,text_embeddings, guidance_scale=3.0)
            for i, image in enumerate(sampled_images):
                image = (image + 1) / 2  # denormalize
                image = torch.clip(image, 0, 1)
                image = image.detach().cpu().permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)
                image_pil = Image.fromarray(image)
                image_pil.save(os.path.join(save_dir, f"image_epoch{epoch + 1}_sample_{i}.png"))
torch.save({
    "model": diffusion_model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
}, os.path.join(save_dir, "diffusion_final.pth"))