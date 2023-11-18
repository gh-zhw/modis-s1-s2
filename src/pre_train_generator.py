import time
import numpy as np
import torchvision
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from model import *
from load_dataset import get_dataloader, get_dataset
from utils import generated_S2_to_rgb, L1_Loss_for_bands, L2_Loss_for_bands, calc_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 12

train_dataloader, val_dataloader, _ = get_dataloader(batch_size, *get_dataset())

generator = Generator()
generator = generator.to(device)

g_lr = 3e-4
g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)
g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=50, gamma=1)

# tensorboard
writer = SummaryWriter(r"D:\Code\MODIS_S1_S2\logs\\")

train_loss_dict = {"train_loss": [], "L1_loss": [], "L2_loss": [], "L_loss_band_1": [],
                   "L_loss_band_2": [], "L_loss_band_3": [], "L_loss_band_4": [], "L_loss_band_5": [],
                   "L_loss_band_6": [], "L_loss_band_7": [], "L_loss_band_8": []}
val_loss_dict = {"val_loss": [], "L1_loss": [], "L2_loss": [], "L_loss_band_1": [],
                 "L_loss_band_2": [], "L_loss_band_3": [], "L_loss_band_4": [], "L_loss_band_5": [],
                 "L_loss_band_6": [], "L_loss_band_7": [], "L_loss_band_8": []}

LAMBDA_L1 = 0.5
LAMBDA_L2 = 0.5

epochs = 50
step = 0
total_step = epochs * len(train_dataloader)
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

        # generated fake image
        generated_S2_image = generator(MODIS_image, S1_image)

        # calculate L2_loss for bands
        L2_loss_bands = L2_Loss_for_bands(generated_S2_image, real_S2_image)
        L2_loss = L2_loss_bands.mean()

        # calculate L1_loss for bands
        L1_loss_bands = L1_Loss_for_bands(generated_S2_image, real_S2_image)
        L1_loss = L1_loss_bands.mean()

        L_loss_bands = LAMBDA_L1 * L1_loss_bands + LAMBDA_L2 * L2_loss_bands
        L_loss = LAMBDA_L1 * L1_loss + LAMBDA_L2 * L2_loss

        g_loss = L_loss

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if step % 20 == 0:
            end_time = time.time()
            print("[step {}/{}] train_L_loss = {} | L1_loss = {} | L2_loss = {}  {}s".format(
                step, total_step, L_loss.item(), L1_loss.item(), L2_loss.item(), round(end_time - start_time, 2)))

            train_loss_dict["train_loss"].append(L_loss.item())
            train_loss_dict["L1_loss"].append(L1_loss.item())
            train_loss_dict["L2_loss"].append(L2_loss.item())
            for band in range(L_loss_bands.shape[0]):
                train_loss_dict[f"L_loss_band_{band + 1}"].append(L_loss_bands[band].item())

            writer.add_scalar("train_loss", L_loss, step)
            writer.add_scalar("train_L1_loss", L1_loss, step)
            writer.add_scalar("train_L2_loss", L2_loss, step)
            # for band in range(L_loss_bands.shape[0]):
            #     writer.add_scalar("train_L_loss_band_" + str(band + 1), L_loss_bands[band].item(), step)

        step += 1

    g_scheduler.step()

    # validate
    generator.eval()
    flag = True
    val_data_len = len(val_dataloader)
    val_L_loss_bands = torch.zeros(8).to(device)
    val_L_loss = torch.zeros(1).to(device)
    val_L1_loss = torch.zeros(1).to(device)
    val_L2_loss = torch.zeros(1).to(device)
    metric = {"mae": 0, "mse": 0, "psnr": 0, "ssim": 0}
    for mini_batch in val_dataloader:
        MODIS_image, S1_image, S2_image = mini_batch
        MODIS_image = MODIS_image.to(device)
        S1_image = S1_image.to(device)
        real_S2_image = S2_image.to(device)

        with torch.no_grad():
            generated_S2_image = generator(MODIS_image, S1_image)
            if flag:
                fake_S2_rgb = generated_S2_to_rgb(generated_S2_image[:4])
                real_S2_rgb = generated_S2_to_rgb(real_S2_image[:4])
                fake_S2_rbg_grid = torchvision.utils.make_grid(fake_S2_rgb, normalize=True)
                real_S2_rbg_grid = torchvision.utils.make_grid(real_S2_rgb, normalize=True)
                writer.add_image("generated_images", fake_S2_rbg_grid, epoch)
                writer.add_image("real_images", real_S2_rbg_grid, epoch)
                flag = False

        # calculate L2_loss for bands
        L2_loss_bands = L2_Loss_for_bands(generated_S2_image, real_S2_image)
        L2_loss = L2_loss_bands.mean()

        # calculate L1_loss for bands
        L1_loss_bands = L1_Loss_for_bands(generated_S2_image, real_S2_image)
        L1_loss = L1_loss_bands.mean()

        L_loss_bands = LAMBDA_L1 * L1_loss_bands + LAMBDA_L2 * L2_loss_bands
        L_loss = LAMBDA_L1 * L1_loss + LAMBDA_L2 * L2_loss

        val_L_loss_bands += (L_loss_bands / val_data_len)
        val_L_loss += (L_loss / val_data_len)
        val_L1_loss += (L1_loss / val_data_len)
        val_L2_loss += (L2_loss / val_data_len)

        mae, mse, psnr, ssim_value = calc_metric(generated_S2_image, real_S2_image, 1, 1)
        metric["mae"] += (mae / val_data_len)
        metric["mse"] += (mse / val_data_len)
        metric["psnr"] += (psnr / val_data_len)
        metric["ssim"] += (ssim_value / val_data_len)

    print("val_L_loss = {} | L1_loss = {} | L2_loss = {}".format(val_L_loss.item(), val_L1_loss.item(), val_L2_loss.item()))
    for key, value in metric.items():
        print(f"{key}: {value.item()} | ", end="")
    print()

    val_loss_dict["val_loss"].append(val_L_loss.item())
    val_loss_dict["L1_loss"].append(val_L1_loss.item())
    val_loss_dict["L2_loss"].append(val_L2_loss.item())
    for band in range(val_L_loss_bands.shape[0]):
        val_loss_dict[f"L_loss_band_{band + 1}"].append(val_L_loss_bands[band].item())

    writer.add_scalar("val_L_loss", val_L_loss, epoch)
    writer.add_scalar("val_L1_loss", val_L1_loss, epoch)
    writer.add_scalar("val_L2_loss", val_L2_loss, epoch)
    for band in range(val_L_loss_bands.shape[0]):
        writer.add_scalar("val_L_loss_band_" + str(band + 1), val_L_loss_bands[band].item(), epoch)

    if ((epoch+1) % 50 == 0 and epoch > 0) or epoch == epochs - 1:
        torch.save(generator, f"D:\Code\MODIS_S1_S2\model\pre_train_generator_epoch_{epoch+1}.pth")

writer.close()

np.save(r"D:\Code\MODIS_S1_S2\output\loss\pre_train_generator_train_loss.npy", train_loss_dict)
np.save(r"D:\Code\MODIS_S1_S2\output\loss\pre_train_generator_val_loss.npy", val_loss_dict)

if __name__ == '__main__':
    pass
