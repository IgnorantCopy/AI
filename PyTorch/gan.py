import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# 输入：长度为 100 的噪声向量(正态分布随机数)
# 输出：(1, 28, 28)的图片
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),  # Linear1: 100 --> 256
            nn.ReLU(),
            nn.Linear(256, 512),  # Linear2: 256 --> 512
            nn.ReLU(),
            nn.Linear(512, 28 * 28),  # Linear3: 512 --> 28 * 28
            nn.Tanh()
        )

    def forward(self, x):
        """
        :param x: 长度为 100 的噪声向量
        :return: 输出的图片
        """
        image = self.main(x)
        image = image.view(-1, 28, 28, 1)  # 转换为 (batch_size, 28, 28, 1) ；-1 表示 batch_size 由上层自动计算
        return image


# 输入：(1, 28, 28)的图片
# 输出：二分类的概率值(使用sigmoid激活函数) ==> 使用 BCELoss 作为损失函数计算交叉熵损失
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 512),  # Linear1: 28 * 28 --> 512
            nn.LeakyReLU(),
            nn.Linear(512, 256),  # Linear2: 512 --> 256
            nn.LeakyReLU(),
            nn.Linear(256, 1),  # Linear3: 256 --> 1
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.main(x)
        return x


def plot_generator_image(model, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    figure = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow((prediction[i] + 1) / 2)
        plt.axis('off')
    plt.show()


def train():
    # 归一化(因为生成模型会用 tanh 进行激活，所以数据要在 -1 到 1 之间)
    transform = transforms.Compose([
        transforms.ToTensor(),  # 作用：① 将图像转化为tensor( channel x width x height )，② 归一化到[0,1]
        transforms.Normalize(0.5, 0.5)  # 均值, 方差；减去均值，除以方差，使得数据分布在[-1,1]之间
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 定义模型
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    # 定义优化器
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    # 定义损失函数
    loss_fn = nn.BCELoss()

    D_losses = []
    G_losses = []
    epochs = 500
    test_input = torch.randn(16, 100, device=device)
    for epoch in range(epochs):
        d_epoch_loss = 0
        g_epoch_loss = 0
        count = len(dataloader)  # 返回批次数；len(dataset)会返回样本数
        for i, (image, _) in enumerate(dataloader):
            image = image.to(device)
            size = image.size(0)
            noise = torch.randn(size, 100, device=device)

            # 判别器的梯度清零
            optimizer_D.zero_grad()
            # 判别器对真实图片的预测
            real_output = discriminator(image)
            d_real_loss = loss_fn(real_output, torch.ones_like(real_output))
            d_real_loss.backward()
            # 判别器对生成图片的预测
            fake_image = generator(noise)
            fake_output = discriminator(fake_image.detach())
            d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))
            d_fake_loss.backward()
            # 判别器的损失
            d_loss = d_real_loss + d_fake_loss
            optimizer_D.step()

            # 生成器的梯度清零
            optimizer_G.zero_grad()
            # 生成器对生成图片的预测
            fake_output = discriminator(fake_image)
            g_loss = loss_fn(fake_output, torch.ones_like(fake_output))
            g_loss.backward()
            optimizer_G.step()

            with torch.no_grad():
                d_epoch_loss += d_loss
                g_epoch_loss += g_loss
        with torch.no_grad():
            d_epoch_loss /= count
            g_epoch_loss /= count
            D_losses.append(d_epoch_loss.item())
            G_losses.append(g_epoch_loss.item())
            print(f"Epoch {epoch + 1}/{epochs}, D_loss: {d_epoch_loss.item():.4f}, G_loss: {g_epoch_loss.item():.4f}")
            plot_generator_image(generator, test_input)


if __name__ == '__main__':
    train()