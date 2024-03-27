import torch
import numpy as np
from load_dataset import get_dataloader, get_dataset
from utils import calc_metric, L1_Loss_for_bands

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
train_dataloader, val_dataloader, test_dataloader = get_dataloader(batch_size, *get_dataset())
generator = torch.load(r"D:\Code\modis-s1-s2\model\generator_epoch_100.pth")
generator = generator.to(device)
generator.eval()

val_data_len = len(val_dataloader)
metric = {"mae": 0, "mse": 0, "sam": 0, "psnr": 0, "ssim": 0}
step = 0
for mini_batch in val_dataloader:
    MODIS_image, S1_image, S2_image, before_image, after_image = mini_batch
    MODIS_image = MODIS_image.to(device)
    S1_image = S1_image.to(device)
    real_S2_image = S2_image.to(device)
    before_image = before_image.to(device)
    after_image = after_image.to(device)

    # permutate MODIS_image
    permuted_index = np.arange(batch_size)
    np.random.shuffle(permuted_index)
    MODIS_image = MODIS_image[permuted_index, :, :, :]

    # permutate S1_image
    permuted_index = np.arange(batch_size)
    np.random.shuffle(permuted_index)
    S1_image = S1_image[permuted_index, :, :, :]

    # permutate ref_S2_image
    # permuted_index = np.arange(batch_size)
    # np.random.shuffle(permuted_index)
    # before_image = before_image[permuted_index, :, :, :]
    # after_image = after_image[permuted_index, :, :, :]

    with torch.no_grad():
        generated_S2_image = generator(MODIS_image, S1_image, before_image, after_image)

    L_loss_bands = L1_Loss_for_bands(generated_S2_image, real_S2_image)
    L_loss = L_loss_bands.mean()
    mae, mse, sam, psnr, ssim_value = calc_metric(generated_S2_image, real_S2_image, 1, 1)
    metric["mae"] += (mae / val_data_len).item()
    metric["mse"] += (mse / val_data_len).item()
    metric["sam"] += (sam / val_data_len).item()
    metric["psnr"] += (psnr / val_data_len).item()
    metric["ssim"] += (ssim_value / val_data_len).item()

    step += 1


for key, value in metric.items():
    print(f"{key}: {value} | ", end="")

