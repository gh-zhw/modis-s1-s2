import torch
from load_dataset import get_dataloader, get_dataset
from utils import L2_Loss_for_bands


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
train_dataloader, val_dataloader, test_dataloader = get_dataloader(batch_size, *get_dataset())
dataloader = test_dataloader
generator = torch.load(r"D:\Code\MODIS_S1_S2\model\pre_train\pre_train_generator_epoch_100.pth")
generator = generator.to(device)
generator.eval()

test_L2_loss_bands = torch.zeros(8).to(device)

for mini_batch in dataloader:
    MODIS_image, S1_image, S2_image = mini_batch
    MODIS_image = MODIS_image.to(device)
    S1_image = S1_image.to(device)
    real_S2_image = S2_image.to(device)

    with torch.no_grad():
        generated_S2_image = generator(MODIS_image, S1_image)

    test_L2_loss_bands += L2_Loss_for_bands(generated_S2_image, real_S2_image)


test_L2_loss_bands /= len(dataloader)
test_L2_loss = test_L2_loss_bands.mean()
print(f"test_L2_loss：{test_L2_loss}")
for band in range(test_L2_loss_bands.shape[0]):
    print(f"test_L2_loss_band_{band + 1}:", test_L2_loss_bands[band].item())


if __name__ == '__main__':
    pass