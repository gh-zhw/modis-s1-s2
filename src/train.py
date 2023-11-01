import time
import torch
import torchvision
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import *
from utils import get_image_path
from load_dataset import SatelliteImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODIS_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(mean=1291.39, std=734.66)])
S1_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean=-12.80, std=5.50)])
S2_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean=1599.95, std=850.08)])
transforms = {"MODIS": MODIS_transform, "S1": S1_transform, "S2": S2_transform}

train_image_paths, val_image_paths, test_image_paths = get_image_path()

train_dataset = SatelliteImageDataset(train_image_paths, transform=transforms)
val_dataset = SatelliteImageDataset(val_image_paths, transform=transforms)
test_dataset = SatelliteImageDataset(test_image_paths, transform=transforms)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=False)

generator = Generator()
discriminator = Discriminator()
generator = generator.to(device)
discriminator = discriminator.to(device)

g_lr = 1e-3
d_lr = 1e-3
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

epochs = 10
step = 0
start_time = time.time()
for epoch in range(epochs):
    print("=" * 30 + f" epoch {epoch + 1} " + "=" * 30)

    # train
    for mini_batch in train_dataloader:
        MODIS_image, S1_image, S2_image = mini_batch
        MODIS_image = MODIS_image.to(device)
        S1_image = S1_image.to(device)
        real_S2_image = S2_image.to(device)

        # 生成超分辨率图像
        generated_S2_image = generator(MODIS_image, S1_image)

        # 计算各波段的L2损失
        L2_loss_bands = ((generated_S2_image - real_S2_image) ** 2).sum(dim=(0, 2, 3))
        L2_loss_bands /= (real_S2_image.shape[0] * real_S2_image.shape[2] * real_S2_image.shape[3])
        L2_loss = L2_loss_bands.mean()

        true_label = torch.ones((real_S2_image.shape[0], 1)).to(device)
        false_label = torch.zeros((real_S2_image.shape[0], 1)).to(device)

        g_optimizer.zero_grad()
        # g_loss = -torch.mean(discriminator(g_S2_image))
        g_loss = loss_fn(discriminator(generated_S2_image), true_label) + L2_loss
        g_loss.backward()
        g_optimizer.step()

        # 更新判别器参数
        if step % 5 == 0:
            d_optimizer.zero_grad()
            # d_fake_loss = torch.mean(discriminator(g_S2_image.detach()))
            # d_real_loss = -torch.mean(discriminator(real_S2_image))
            d_fake_loss = loss_fn(discriminator(generated_S2_image.detach()), false_label)
            d_real_loss = loss_fn(discriminator(real_S2_image), true_label)
            d_loss = d_fake_loss + d_real_loss
            d_loss.backward()
            d_optimizer.step()

        # clip param for discriminator
        # for parm in discriminator.parameters():
        #     parm.data.clamp_(-0.01, 0.01)

        if step % 10 == 0:
            end_time = time.time()
            print("训练次数：{}，g_loss：{}，d_loss：{}，train_L2_loss：{} {}s".format(
                step, g_loss.item(), d_loss.item(), L2_loss_bands.mean().item(), round(end_time - start_time, 2)))

            with torch.no_grad():
                print(discriminator(generated_S2_image).mean().item(), discriminator(real_S2_image).mean().item())

            writer.add_scalar("g_loss", g_loss, step)
            writer.add_scalar("d_loss", d_loss, step)
            writer.add_scalar("train_L2_loss", L2_loss, step)
            # for band in range(L2_loss_bands.shape[0]):
            #     writer.add_scalar("train_L2_loss_band_" + str(band + 1), L2_loss_bands[band], step)

        step += 1

    g_scheduler.step()
    d_scheduler.step()

    # validate
    val_L2_loss_bands = torch.zeros(8).to(device)
    for mini_batch in val_dataloader:
        MODIS_image, S1_image, S2_image = mini_batch
        MODIS_image = MODIS_image.to(device)
        S1_image = S1_image.to(device)
        real_S2_image = S2_image.to(device)

        with torch.no_grad():
            generated_S2_image = generator(MODIS_image, S1_image)

        L2_loss_bands = ((generated_S2_image - real_S2_image) ** 2).sum(dim=(0, 2, 3))
        L2_loss_bands /= (real_S2_image.shape[0] * real_S2_image.shape[2] * real_S2_image.shape[3])
        val_L2_loss_bands += L2_loss_bands

    val_L2_loss_bands /= len(val_dataloader)
    val_L2_loss = val_L2_loss_bands.mean()
    print(f"val_L2_loss：{val_L2_loss}")

    writer.add_scalar("val_L2_loss", val_L2_loss, epoch)
    # for band in range(L2_loss_bands.shape[0]):
    #     writer.add_scalar("val_L2_loss_band_" + str(band + 1), val_L2_loss_bands[band], epoch)

writer.close()
torch.save(generator, f"../model/GAN_generator_epoch_{epochs}.pth")
torch.save(discriminator, f"../model/GAN_discriminator_epoch_{epochs}.pth")

if __name__ == '__main__':
    pass
