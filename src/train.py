import time
import numpy as np
from torch.nn.functional import interpolate
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from model import *
from load_dataset import get_dataloader, get_dataset
from utils import generated_S2_to_rgb, gradient_penalty, L1_Loss_for_bands, L2_Loss_for_bands

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16

train_dataloader, val_dataloader, _ = get_dataloader(batch_size, *get_dataset())

generator = Generator()
discriminator = Discriminator()
generator = generator.to(device)
discriminator = discriminator.to(device)

g_lr = 5e-4
d_lr = 5e-4
g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=50, gamma=0.5)
d_scheduler = lr_scheduler.StepLR(d_optimizer, step_size=50, gamma=0.5)

# tensorboard
writer = SummaryWriter(r"D:\Code\MODIS_S1_S2\logs\wgan\\")

train_loss = {"g_loss": [], "d_loss": [], "W_dis": [], "L_loss": [], "L_loss_band_1": [],
              "L_loss_band_2": [], "L_loss_band_3": [], "L_loss_band_4": [], "L_loss_band_5": [],
              "L_loss_band_6": [], "L_loss_band_7": [], "L_loss_band_8": []}
val_loss = {"L_loss": [], "L_loss_band_1": [], "L_loss_band_2": [], "L_loss_band_3": [],
            "L_loss_band_4": [], "L_loss_band_5": [], "L_loss_band_6": [], "L_loss_band_7": [],
            "L_loss_band_8": []}

LAMBDA_GP = 10

epochs = 5
step = 0
total_step = epochs * len(train_dataloader)
start_time = time.time()
for epoch in range(epochs):
    print("=" * 30 + f" epoch {epoch + 1} " + "=" * 30)

    # train
    generator.train()
    discriminator.train()
    for mini_batch in train_dataloader:
        MODIS_image, S1_image, S2_image = mini_batch
        MODIS_image = MODIS_image.to(device)
        S1_image = S1_image.to(device)
        real_S2_image = S2_image.to(device)

        # generated fake image
        generated_S2_image = generator(MODIS_image, S1_image)

        # upsample MODIS(5*5) to MODIS_upsamped(250*250)
        MODIS_image_upsampled = interpolate(MODIS_image, size=250, mode="nearest").requires_grad_(True)

        d_fake_input = torch.cat((generated_S2_image.detach(), S1_image, MODIS_image_upsampled), dim=1)
        d_real_input = torch.cat((real_S2_image, S1_image, MODIS_image_upsampled), dim=1)

        gp = gradient_penalty(discriminator, d_real_input, d_fake_input, device=device)
        d_fake_loss = torch.mean(discriminator(d_fake_input))
        d_real_loss = -torch.mean(discriminator(d_real_input))
        d_loss = d_fake_loss + d_real_loss + LAMBDA_GP * gp
        d_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()
        
        wasserstein_distance = -(d_fake_loss + d_real_loss)

        # update G
        if step % 5 == 0:
            # calculate L2_loss for bands
            L_loss_bands = L2_Loss_for_bands(generated_S2_image, real_S2_image)
            L_loss = L_loss_bands.mean()

            # calculate L1_loss for bands
            # L_loss_bands = L1_Loss_for_bands(generated_S2_image, real_S2_image)
            # L_loss = L_loss_bands.mean()

            g_optimizer.zero_grad()
            g_loss = -torch.mean(discriminator(d_fake_input))
            g_total_loss = g_loss + L_loss
            g_loss.backward()
            g_optimizer.step()

        if step % 20 == 0:
            end_time = time.time()
            print("[step {}/{}] g_loss = {} | d_loss = {} | W_dis = {} | train_L_loss = {}  {}s".format(
                step, total_step, g_loss.item(), d_loss.item(), wasserstein_distance.item(), L_loss.item(),
                round(end_time - start_time, 2)))

            train_loss["g_loss"].append(g_loss.item())
            train_loss["d_loss"].append(d_loss.item())
            train_loss["W_dis"].append(wasserstein_distance.item())
            train_loss["L_loss"].append(L_loss.item())
            for band in range(L_loss_bands.shape[0]):
                train_loss[f"L_loss_band_{band + 1}"].append(L_loss_bands[band].item())

            writer.add_scalar("g_loss", g_loss, step)
            writer.add_scalar("d_loss", d_loss, step)
            writer.add_scalar("W_dis", wasserstein_distance, step)
            writer.add_scalar("train_L_loss", L_loss, step)
            # for band in range(L_loss_bands.shape[0]):
            #     writer.add_scalar("train_L_loss_band_" + str(band + 1), L_loss_bands[band].item(), step)

        step += 1

    g_scheduler.step()
    d_scheduler.step()

    # validate
    generator.eval()
    discriminator.eval()
    flag = True
    val_L_loss_bands = torch.zeros(8).to(device)
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

        # calculate L2_loss for bands
        L_loss_bands = L2_Loss_for_bands(generated_S2_image, real_S2_image)
        L_loss_value = L_loss_bands.mean()

        # calculate L1_loss for bands
        # L_loss_bands = L1_Loss_for_bands(generated_S2_image, real_S2_image)
        # L_loss = L_loss_bands.mean()
        
        val_L_loss_bands += L_loss_bands

    val_L_loss_bands /= len(val_dataloader)
    val_L_loss = val_L_loss_bands.mean()
    print(f"val_L_loss：{val_L_loss}")

    val_loss["L_loss"].append(val_L_loss.item())
    for band in range(val_L_loss_bands.shape[0]):
        val_loss[f"L_loss_band_{band + 1}"].append(val_L_loss_bands[band].item())

    writer.add_scalar("val_L_loss", val_L_loss, epoch)
    # for band in range(L_loss_bands.shape[0]):
    #     writer.add_scalar("val_L_loss_band_" + str(band + 1), val_L_loss_bands[band].item(), epoch)

    if ((epoch+1) % 50 == 0 and epoch > 0) or epoch == epochs-1:
        torch.save(generator, f"D:\Code\MODIS_S1_S2\model\pre_train_generator_epoch_{epoch+1}.pth")
        torch.save(discriminator, f"D:\Code\MODIS_S1_S2\model\GAN_discriminator_epoch_{epoch+1}.pth")


writer.close()

np.save(r"D:\Code\MODIS_S1_S2\output\loss\train_loss.npy", train_loss)
np.save(r"D:\Code\MODIS_S1_S2\output\loss\val_loss.npy", val_loss)


if __name__ == '__main__':
    pass
