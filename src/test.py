import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from load_dataset import get_dataloader, get_dataset
from utils import generated_S2_to_rgb, calc_metric, L1_Loss_for_bands

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
train_dataloader, val_dataloader, test_dataloader = get_dataloader(batch_size, *get_dataset())
generator = torch.load(r"D:\Code\modis-s1-s2\model\generator_epoch_300.pth")
generator = generator.to(device)
generator.eval()

writer = SummaryWriter(r"D:\Code\modis-s1-s2\logs\test")

test_data_len = len(val_dataloader)
metric = {"mae": 0, "mse": 0, "sam": 0, "psnr": 0, "ssim": 0}
step = 0
for mini_batch in val_dataloader:
    MODIS_image, S1_image, S2_image, before_image, after_image = mini_batch
    MODIS_image = MODIS_image.to(device)
    S1_image = S1_image.to(device)
    real_S2_image = S2_image.to(device)
    before_image = before_image.to(device)
    after_image = after_image.to(device)

    with torch.no_grad():
        generated_S2_image = generator(MODIS_image, S1_image, before_image, after_image)

    L_loss_bands = L1_Loss_for_bands(generated_S2_image, real_S2_image)
    L_loss = L_loss_bands.mean()
    mae, mse, sam, psnr, ssim_value = calc_metric(generated_S2_image, real_S2_image, 1, 1)
    metric["mae"] += (mae / test_data_len).item()
    metric["mse"] += (mse / test_data_len).item()
    metric["sam"] += (sam / test_data_len).item()
    metric["psnr"] += (psnr / test_data_len).item()
    metric["ssim"] += (ssim_value / test_data_len).item()

    fake_S2_rgb = generated_S2_to_rgb(generated_S2_image[:4])
    real_S2_rgb = generated_S2_to_rgb(real_S2_image[:4])
    fake_S2_rbg_grid = torchvision.utils.make_grid(fake_S2_rgb, normalize=True)
    real_S2_rbg_grid = torchvision.utils.make_grid(real_S2_rgb, normalize=True)
    writer.add_image("generated_images", fake_S2_rbg_grid, step)
    writer.add_image("real_images", real_S2_rbg_grid, step)

    step += 1

for key, value in metric.items():
    print(f"{key}: {value} | ", end="")


