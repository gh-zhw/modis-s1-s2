import time
import torch
import torchvision
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import *
from load_dataset import SatelliteImageDataset, group_image_paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODIS_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(mean=1291.39, std=734.66)])
S1_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean=-12.77, std=5.45)])
S2_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean=1595.74, std=838.34)])
transforms = {"MODIS": MODIS_transform, "S1": S1_transform, "S2": S2_transform}

dataset = SatelliteImageDataset(group_image_paths, transform=transforms)

batch_size = 64
dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

generator = Generator()
discriminator = Discriminator()
generator = generator.to(device)
discriminator = discriminator.to(device)

g_lr = 1e-3
d_lr = 2e-4
# g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=g_lr)
# d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=d_lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=10, gamma=0.8)
d_scheduler = lr_scheduler.StepLR(d_optimizer, step_size=10, gamma=0.8)

# tensorboard
writer = SummaryWriter("../logs")

generator.train()
discriminator.train()

loss_fn = nn.BCELoss()
loss_fn = loss_fn.to(device)

true_label = torch.ones((batch_size, 1)).to(device)
false_label = torch.zeros((batch_size, 1)).to(device)

epochs = 100
step = 0
start_time = time.time()
for epoch in range(epochs):
    print("=" * 30 + f" epoch {epoch + 1} " + "=" * 30)

    for mini_batch in dataloader:
        MODIS_image, S1_image, S2_image = mini_batch
        MODIS_image = MODIS_image.to(device)
        S1_image = S1_image.to(device)
        real_S2_image = S2_image.to(device)

        # 生成超分辨率图像
        g_S2_image = generator(S1_image, MODIS_image)  # 一定要按顺序！！！

        # 计算各波段的L2损失
        L2_loss_bands = ((g_S2_image - real_S2_image) ** 2).sum(dim=(0, 2, 3))
        L2_loss_bands /= (real_S2_image.shape[0] * real_S2_image.shape[2] * real_S2_image.shape[3])
        L2_loss = L2_loss_bands.mean()

        # 每3个step更新一次生成器参数
        if step % 3 == 0:
            g_optimizer.zero_grad()
            # g_loss = -torch.mean(discriminator(g_S2_image))
            g_loss = loss_fn(discriminator(g_S2_image), true_label) + L2_loss
            g_loss.backward()
            g_optimizer.step()

        # 更新判别器参数
        d_optimizer.zero_grad()
        # d_fake_loss = torch.mean(discriminator(g_S2_image.detach()))
        # d_real_loss = -torch.mean(discriminator(real_S2_image))
        d_fake_loss = loss_fn(discriminator(g_S2_image.detach()), false_label)
        d_real_loss = loss_fn(discriminator(real_S2_image), true_label)
        d_loss = d_fake_loss + d_real_loss
        d_loss.backward()
        d_optimizer.step()

        # clip param for discriminator
        # for parm in discriminator.parameters():
        #     parm.data.clamp_(-0.01, 0.01)

        if step % 10 == 0:
            end_time = time.time()
            print("训练次数：{}，g_loss：{}，d_loss：{}，L2_loss：{} {}s".format(
                step, g_loss.item(), d_loss.item(), L2_loss_bands.mean().item(), round(end_time - start_time, 2)))
            with torch.no_grad():
                print(discriminator(g_S2_image).mean().item(), discriminator(real_S2_image).mean().item())

            writer.add_scalar("g_loss", g_loss, step)
            writer.add_scalar("d_loss", d_loss, step)
            writer.add_scalar("L2_loss", L2_loss, step)
            for band in range(L2_loss_bands.shape[0]):
                writer.add_scalar("L2_loss_band_" + str(band + 1), L2_loss_bands[band], step)

        step += 1
        g_scheduler.step()
        d_scheduler.step()

writer.close()
torch.save(generator, f"../model/GAN_generator_epoch_{epochs}.pth")
torch.save(discriminator, f"../model/GAN_discriminator_epoch_{epochs}.pth")

if __name__ == '__main__':
    pass
