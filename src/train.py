import time
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from model import *
from load_dataset import get_dataloader, get_dataset
from utils import generated_S2_to_rgb, gradient_penalty, L1_Loss_for_bands, calc_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8

train_dataloader, val_dataloader, _ = get_dataloader(batch_size, *get_dataset())

try:
    checkpoint = torch.load(r"D:\Code\modis-s1-s2\checkpoint\checkpoint_epoch_0.pth")
except FileNotFoundError:
    checkpoint = None

generator = Generator()
discriminator = Discriminator()
generator = generator.to(device)
discriminator = discriminator.to(device)

g_lr = 3e-4
d_lr = 3e-4
g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=d_lr)

if checkpoint is not None:
    print("Load checkpoint.")
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    g_optimizer.load_state_dict(checkpoint["g_optimizer"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer"])
else:
    print("No such checkpoint.")


# tensorboard
writer = SummaryWriter(r"D:\Code\modis-s1-s2\logs\wgan")

train_loss = {"g_loss": [], "d_loss": [], "W_dis": [], "L_loss": [], "L_loss_band_1": [],
              "L_loss_band_2": [], "L_loss_band_3": [], "L_loss_band_4": [], "L_loss_band_5": [],
              "L_loss_band_6": [], "L_loss_band_7": [], "L_loss_band_8": []}
val_loss = {"L_loss": [], "L_loss_band_1": [], "L_loss_band_2": [], "L_loss_band_3": [],
            "L_loss_band_4": [], "L_loss_band_5": [], "L_loss_band_6": [], "L_loss_band_7": [],
            "L_loss_band_8": []}

LAMBDA_GP = 10
LAMBDA_g_loss = 1e-3

start_epoch = checkpoint["epoch"] if checkpoint is not None else 0
end_epoch = 100
step = start_epoch * len(train_dataloader)
total_step = end_epoch * len(train_dataloader)
start_time = time.time()
for epoch in range(start_epoch, end_epoch):
    print("=" * 30 + f" epoch {epoch + 1} " + "=" * 30)
    print("Current G learning rate:", g_optimizer.param_groups[0]['lr'])
    print("Current D learning rate:", d_optimizer.param_groups[0]['lr'])

    # train
    generator.train()
    discriminator.train()
    for mini_batch in train_dataloader:
        MODIS_image, S1_image, S2_image, ref_image = mini_batch
        MODIS_image = MODIS_image.to(device)
        S1_image = S1_image.to(device)
        real_S2_image = S2_image.to(device)
        ref_image = ref_image.to(device)

        # generated fake image
        generated_S2_image = generator(MODIS_image, S1_image, ref_image)

        gp = gradient_penalty(discriminator, real_S2_image, generated_S2_image, device=device)
        d_fake_loss = torch.mean(discriminator(generated_S2_image))
        d_real_loss = -torch.mean(discriminator(real_S2_image))
        d_loss = d_fake_loss + d_real_loss + LAMBDA_GP * gp
        d_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()
        
        wasserstein_distance = -(d_fake_loss + d_real_loss)

        # update G
        # calculate L1_loss for bands
        L1_loss_bands = L1_Loss_for_bands(generated_S2_image, real_S2_image)
        L1_loss = L1_loss_bands.mean()

        L_loss_bands = L1_loss_bands
        L_loss = L1_loss

        g_optimizer.zero_grad()
        g_loss = -torch.mean(discriminator(generated_S2_image))
        g_total_loss = LAMBDA_g_loss * g_loss + L_loss
        g_total_loss.backward()
        g_optimizer.step()

        if step % 50 == 0:
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

    # validate
    generator.eval()
    discriminator.eval()
    flag = True
    val_data_len = len(val_dataloader)
    val_L_loss_bands = torch.zeros(8).to(device)
    val_L_loss = torch.zeros(1).to(device)
    metric = {"mae": 0, "mse": 0, "sam": 0, "psnr": 0, "ssim": 0}
    for mini_batch in val_dataloader:
        MODIS_image, S1_image, S2_image, ref_image = mini_batch
        MODIS_image = MODIS_image.to(device)
        S1_image = S1_image.to(device)
        real_S2_image = S2_image.to(device)
        ref_image = ref_image.to(device)

        with torch.no_grad():
            generated_S2_image = generator(MODIS_image, S1_image, ref_image)
            if flag:
                fake_S2_rgb = generated_S2_to_rgb(generated_S2_image[:4])
                real_S2_rgb = generated_S2_to_rgb(real_S2_image[:4])
                fake_S2_rbg_grid = torchvision.utils.make_grid(fake_S2_rgb, normalize=True)
                real_S2_rbg_grid = torchvision.utils.make_grid(real_S2_rgb, normalize=True)
                writer.add_image("generated_images", fake_S2_rbg_grid, epoch)
                writer.add_image("real_images", real_S2_rbg_grid, epoch)
                flag = False

        # calculate L1_loss for bands
        L1_loss_bands = L1_Loss_for_bands(generated_S2_image, real_S2_image)
        L1_loss = L1_loss_bands.mean()

        L_loss_bands = L1_loss_bands
        L_loss = L1_loss

        val_L_loss_bands += (L_loss_bands / val_data_len)
        val_L_loss += (L_loss / val_data_len)

        mae, mse, sam, psnr, ssim_value = calc_metric(generated_S2_image, real_S2_image, 1, 1)
        metric["mae"] += (mae / val_data_len)
        metric["mse"] += (mse / val_data_len)
        metric["sam"] += (sam / val_data_len)
        metric["psnr"] += (psnr / val_data_len)
        metric["ssim"] += (ssim_value / val_data_len)

    print(f"val_L_loss：{val_L_loss.item()}")
    for key, value in metric.items():
        print(f"{key}: {value.item()} | ", end="")
    print()

    val_loss["L_loss"].append(val_L_loss.item())
    for band in range(val_L_loss_bands.shape[0]):
        val_loss[f"L_loss_band_{band + 1}"].append(val_L_loss_bands[band].item())

    writer.add_scalar("val_L_loss", val_L_loss, epoch)
    writer.add_scalar("SAM", metric["sam"], epoch)
    writer.add_scalar("PSNR", metric["psnr"], epoch)
    writer.add_scalar("SSIM", metric["ssim"], epoch)
    # for band in range(L_loss_bands.shape[0]):
    #     writer.add_scalar("val_L_loss_band_" + str(band + 1), val_L_loss_bands[band].item(), epoch)

    # if ((epoch+1) % 50 == 0 and epoch > 0) or epoch == end_epoch-1:
    #     torch.save(generator, f"D:\Code\modis-s1-s2\model\wgan\wgan_generator_epoch_{epoch+1}.pth")
    #     torch.save(discriminator, f"D:\Code\modis-s1-s2\model\wgan\wgan_discriminator_epoch_{epoch+1}.pth")

    if ((epoch + 1) % 10 == 0 and epoch > 0) or epoch == end_epoch - 1:
        torch.save(
            {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "d_optimizer": d_optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            f"D:\Code\modis-s1-s2\checkpoint\checkpoint_epoch_{epoch + 1}.pth")
        print("Checkpoint saved.")


writer.close()

# np.save(r"D:\Code\modis-s1-s2\output\loss\wgan\wgan_train_loss.npy", train_loss)
# np.save(r"D:\Code\modis-s1-s2\output\loss\wgan\wgan_val_loss.npy", val_loss)


if __name__ == '__main__':
    pass
