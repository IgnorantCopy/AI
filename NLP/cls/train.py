import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from logger import default_logger


class Trainer:
    def __init__(self, model, data_module, config, ckpt=None, device="auto"):
        self.model = model
        self.data_module = data_module
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['epochs'], eta_min=config['learning_rate'] * 0.01)
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        self.criterion.to(self.device)

        log_dir = self.config['log_dir']
        tensorboard_dir = os.path.join(log_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=tensorboard_dir)

        self.start_epoch = 0
        if ckpt is not None:
            self.load_model(ckpt)

    def train_epoch(self):
        self.model.train()
        dataloader = self.data_module.train_dataloader
        total_loss = 0
        accuracy = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            accuracy += (outputs.argmax(dim=1) == labels).float().sum().item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        total_loss /= len(dataloader)
        accuracy /= len(dataloader.dataset)
        self.logger.add_scalar('train/loss', total_loss)
        self.logger.add_scalar('train/accuracy', accuracy)
        default_logger.info(f"Train Loss: {total_loss:.4f}, Train Accuracy: {accuracy:.4f}")

    def validate_epoch(self):
        self.model.eval()
        dataloader = self.data_module.val_dataloader
        total_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                accuracy += (outputs.argmax(dim=1) == labels).float().sum().item()
                total_loss += loss.item()
        total_loss /= len(dataloader)
        accuracy /= len(dataloader.dataset)
        self.logger.add_scalar('val/loss', total_loss)
        self.logger.add_scalar('val/accuracy', accuracy)
        default_logger.info(f"Validation Loss: {total_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

    def test(self):
        self.model.eval()
        dataloader = self.data_module.test_dataloader
        total_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                accuracy += (outputs.argmax(dim=1) == labels).float().sum().item()
                total_loss += loss.item()
        total_loss /= len(dataloader)
        accuracy /= len(dataloader.dataset)
        default_logger.info(f"Test Loss: {total_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    def fit(self):
        self.validate_epoch()
        epochs = self.config['epochs']
        save_freq = self.config['save_freq']
        for epoch in range(self.start_epoch, epochs):
            default_logger.info(f'Epoch {epoch+1}/{epochs}')
            self.train_epoch()
            self.validate_epoch()
            self.scheduler.step()
            if (epoch + 1) % save_freq == 0:
                self.save_model(epoch + 1)
        self.save_model("last")

    def save_model(self, epoch):
        ckpt_dir = os.path.join(self.config['log_dir'], 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

    def load_model(self, ckpt):
        checkpoint = torch.load(ckpt, weights_only=False)
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])