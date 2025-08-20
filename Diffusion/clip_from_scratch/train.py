import argparse
import os
import time
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.logger import Logger, AverageMeter
from utils import config



def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="./configs/config.yaml", help="config file path")
    parser.add_argument("--log-path", type=str, default="./logs",                help="path to log file")
    parser.add_argument("--resume",   type=str, default=None,                    help="path to checkpoint file")
    parser.add_argument("--device",   type=str, default="cuda",                  help="device to use", choices=["cuda", "cpu"])
    args = parser.parse_args()
    print(args)
    return args


def save_model(model, optimizer, lr_scheduler, epoch, best_loss, filename):
    torch.save({
        'epoch': epoch + 1,
        'best_loss': best_loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }, filename)


def train(model: nn.Module,
          train_loader: DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          device: str):
    model.train()
    loss_meter = AverageMeter()
    start_time = time.time()
    with torch.autocast(device):
        for i, (images, captions) in enumerate(tqdm(train_loader)):
            output = model(images, captions)
            optimizer.zero_grad()
            loss = criterion(*output)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), images.shape[0])
    return loss_meter.avg, time.time() - start_time


def val(model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: str):
    model.eval()
    loss_meter = AverageMeter()
    start_time = time.time()
    with torch.no_grad(), torch.autocast(device):
        for i, (images, captions) in enumerate(tqdm(val_loader)):
            output = model(images, captions)
            loss = criterion(*output)
            loss_meter.update(loss.item(), images.shape[0])
    return loss_meter.avg, time.time() - start_time


def main():
    args = config_parser()
    config_path = args.config
    log_path    = args.log_path
    resume      = args.resume
    device      = args.device

    log_path = os.path.join(log_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)
    logger = Logger(os.path.join(log_path, "train.txt"))

    configure = config.load_config(config_path)
    train_config = configure['train']
    epochs = train_config['epochs']

    model = config.get_model(configure)
    optimizer = config.get_optimizer(model, configure)
    lr_scheduler = config.get_scheduler(optimizer, configure)
    criterion = config.get_loss(configure)

    start_epoch = 0
    best_loss = 0.
    if resume:
        params = torch.load(resume, weights_only=False)
        model.load_state_dict(params['state_dict'])
        lr_scheduler.load_state_dict(params['lr_scheduler'])
        optimizer.load_state_dict(params['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and device == "cuda":
                    state[k] = v.cuda()
        start_epoch = params['epoch']
        best_loss = params['best_loss']
        logger.log(f"Loaded checkpoint from {resume}")
    model.to(device)

    train_loader, val_loader = config.get_data(configure, model)

    for epoch in range(start_epoch, epochs):
        logger.log(f"--------------- Epoch [{epoch+1}/{epochs}] ---------------")
        train_loss, train_time = train(model, train_loader, optimizer, criterion, device)
        writer.add_scalar('train/loss', train_loss, epoch)
        logger.log(f"Train Loss: {train_loss:.4f}\n"
                   f"Train Time: {train_time:.2f}s\n")

        val_loss, val_time = val(model, val_loader, criterion, device)
        writer.add_scalar('val/loss', val_loss, epoch)
        logger.log(f"Val Loss: {val_loss:.4f}\n"
                   f"Val Time: {val_time:.2f}s\n")

        lr_scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model, optimizer, lr_scheduler, epoch, best_loss, os.path.join(log_path, "best.pth"))
            torch.save(model.state_dict(), os.path.join(log_path, "model_state_dict.pth"))
            logger.log(f"Best model saved at epoch {epoch+1} with val_loss {best_loss:.4f}")
        save_model(model, optimizer, lr_scheduler, epoch, best_loss, os.path.join(log_path, f"latest.pth"))

    logger.close()
    writer.close()