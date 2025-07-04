import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import ViT


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir",    type=str,   default="./logs", help="log dir")
    parser.add_argument("--data-root",  type=str,   default="/home/nju-student/mkh/datasets/CIFAR10", help="data root")
    parser.add_argument("--resume",     type=str,   default=None, help="resume path")
    parser.add_argument("--epochs",     type=int,   default=100, help="number of epochs")
    parser.add_argument("--image-size", type=int,   default=224, help="image size")
    parser.add_argument("--batch-size", type=int,   default=16, help="batch size")
    parser.add_argument("--lr",         type=float, default=0.03, help="initial learning rate")
    parser.add_argument("--device",     type=str,   default="cuda", help="device")
    return parser.parse_args()


def main():
    args = config()
    log_dir     = args.log_dir
    data_root   = args.data_root
    resume      = args.resume
    epochs      = args.epochs
    image_size  = args.image_size
    batch_size  = args.batch_size
    lr          = args.lr
    device      = args.device

    log_dir = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S"))
    check_dir(log_dir)

    model = ViT(num_classes=10, image_size=image_size, dim=768, depth=6, heads=12, mlp_dim=768)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    start_epoch = 0
    best_acc = 0.
    logger = open(os.path.join(log_dir, "train.txt"), "w")
    if resume:
        params = torch.load(resume)
        model.load_state_dict(params["state_dict"])
        optimizer.load_state_dict(params["optimizer"])
        lr_scheduler.load_state_dict(params["lr_scheduler"])
        start_epoch = params["epoch"]
        best_acc = params["best_acc"]
        log_dir = os.path.dirname(resume)
        print(f"resume from epoch {start_epoch} with best acc {best_acc:.4f}")
        logger.write(f"resume from epoch {start_epoch} with best acc {best_acc:.4f}")
        logger.flush()
    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)

    print(f"Model has {count_parameters(model) / 1024 ** 2:,}MB trainable parameters.")
    logger.write(f"Model has {count_parameters(model) / 1024 ** 2:,}MB trainable parameters.\n")
    logger.flush()

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = CIFAR10(root=data_root, train=True, transform=train_transforms, download=True)
    test_dataset = CIFAR10(root=data_root, train=False, transform=test_transforms, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("------------------- Start Training ------------------")
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.
        train_acc = 0.
        train_time = time.time()
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * image.size(0)
            train_acc += (output.argmax(dim=1) == label).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        print(f"Epoch {epoch+1}:\n\tTrain Loss: {train_loss:.4f}\n\tTrain Acc: {train_acc:.4f}\n\tTime: {time.time()-train_time:.2f}s")
        logger.write(f"Epoch {epoch+1}:\n\tTrain Loss: {train_loss:.4f}\n\tTrain Acc: {train_acc:.4f}\n\tTime: {time.time()-train_time:.2f}s\n")

        model.eval()
        test_loss = 0.
        test_acc = 0.
        test_time = time.time()
        with torch.no_grad():
            for image, label in test_loader:
                image = image.to(device)
                label = label.to(device)
                output = model(image)
                loss = criterion(output, label)
                test_loss += loss.item() * image.size(0)
                test_acc += (output.argmax(dim=1) == label).sum().item()
        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/acc", test_acc, epoch)
        print(f"\n\tTest Loss: {test_loss:.4f}\n\tTest Acc: {test_acc:.4f}\n\tTime: {time.time()-test_time:.2f}s")
        logger.write(f"\n\tTest Loss: {test_loss:.4f}\n\tTest Acc: {test_acc:.4f}\n\tTime: {time.time()-test_time:.2f}s\n")

        lr_scheduler.step(test_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }, os.path.join(log_dir, "best.pth"))
            print(f"Best model saved at epoch {epoch+1}.")
            logger.write(f"Best model saved at epoch {epoch+1}.\n")
        print("-" * 50)
        logger.write("-" * 50 + "\n")
        logger.flush()
    print("------------------- End Training ------------------")
    logger.write("------------------- End Training ------------------\n")
    print(f"Max GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")
    logger.write(f"Max GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB\n")
    writer.close()
    logger.close()


if __name__ == '__main__':
    main()