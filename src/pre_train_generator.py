import time
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from model import *
from load_dataset import get_dataloader, get_dataset
from utils import generated_S2_to_rgb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16

train_dataloader, val_dataloader, _ = get_dataloader(batch_size, *get_dataset())

generator = Generator()
generator = generator.to(device)

g_lr = 1e-3
g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr, weight_decay=0.0001)
g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=10, gamma=0.9)

# tensorboard
writer = SummaryWriter(r"D:\Code\MODIS_S1_S2\logs\\")

train_loss = {"L2_loss": [], "L2_loss_band_1": [], "L2_loss_band_2": [],
              "L2_loss_band_3": [], "L2_loss_band_4": [], "L2_loss_band_5": [], "L2_loss_band_6": [],
              "L2_loss_band_7": [], "L2_loss_band_8": []}
val_loss = {"L2_loss": [], "L2_loss_band_1": [], "L2_loss_band_2": [], "L2_loss_band_3": [],
            "L2_loss_band_4": [], "L2_loss_band_5": [], "L2_loss_band_6": [], "L2_loss_band_7": [],
            "L2_loss_band_8": []}

epochs = 100
step = 0
start_time = time.time()
for epoch in range(epochs):
    print("=" * 30 + f" epoch {epoch + 1} " + "=" * 30)
    print("Current learning rate:", g_optimizer.param_groups[0]['lr'])

    # train
    generator.train()
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

        g_optimizer.zero_grad()
        L2_loss.backward()
        g_optimizer.step()

        if step % 10 == 0:
            end_time = time.time()
            print("step：{}, train_L2_loss：{} {}s".format(
                step, L2_loss.item(), round(end_time - start_time, 2)))

            train_loss["L2_loss"].append(L2_loss.item())
            for band in range(L2_loss_bands.shape[0]):
                train_loss[f"L2_loss_band_{band + 1}"].append(L2_loss_bands[band].item())

            writer.add_scalar("train_L2_loss", L2_loss, step)
            # for band in range(L2_loss_bands.shape[0]):
            #     writer.add_scalar("train_L2_loss_band_" + str(band + 1), L2_loss_bands[band].item(), step)

        step += 1

    g_scheduler.step()

    # validate
    generator.eval()
    flag = True
    val_L2_loss_bands = torch.zeros(8).to(device)
    for mini_batch in val_dataloader:
        MODIS_image, S1_image, S2_image = mini_batch
        MODIS_image = MODIS_image.to(device)
        S1_image = S1_image.to(device)
        real_S2_image = S2_image.to(device)

        with torch.no_grad():
            generated_S2_image = generator(MODIS_image, S1_image)
            if flag:
                rgb = generated_S2_to_rgb(generated_S2_image[:4])
                writer.add_images("generated_images", rgb, epoch)
                flag = False

        L2_loss_bands = ((generated_S2_image - real_S2_image) ** 2).sum(dim=(0, 2, 3))
        L2_loss_bands /= (real_S2_image.shape[0] * real_S2_image.shape[2] * real_S2_image.shape[3])
        val_L2_loss_bands += L2_loss_bands

    val_L2_loss_bands /= len(val_dataloader)
    val_L2_loss = val_L2_loss_bands.mean()
    print(f"val_L2_loss：{val_L2_loss}")

    val_loss["L2_loss"].append(val_L2_loss.item())
    for band in range(val_L2_loss_bands.shape[0]):
        val_loss[f"L2_loss_band_{band + 1}"].append(val_L2_loss_bands[band].item())

    writer.add_scalar("val_L2_loss", val_L2_loss, epoch)
    for band in range(L2_loss_bands.shape[0]):
        writer.add_scalar("val_L2_loss_band_" + str(band + 1), val_L2_loss_bands[band].item(), epoch)

writer.close()

np.save(r"D:\Code\MODIS_S1_S2\output\loss\pre_train_generator_train_loss.npy", train_loss)
np.save(r"D:\Code\MODIS_S1_S2\output\loss\pre_train_generator_val_loss.npy", val_loss)

torch.save(generator, f"D:\Code\MODIS_S1_S2\model\GAN_generator_epoch_{epochs}.pth")

if __name__ == '__main__': 
    pass
