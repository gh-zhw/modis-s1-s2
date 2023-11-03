import numpy as np
import matplotlib.pyplot as plt
import torch
from load_dataset import get_dataloader, get_dataset


def generate_image(generator, MODIS_image, S1_image, S2_image=None):
    generator.eval()
    with torch.no_grad():
        generated_S2 = generator(MODIS_image, S1_image)

    if S2_image is not None:
        S2_image = S2_image
        L2_loss_bands = ((generated_S2 - S2_image) ** 2).sum(dim=(0, 2, 3))
        L2_loss_bands /= (S2_image.shape[0] * generated_S2.shape[2] * S2_image.shape[3])
        L2_loss = L2_loss_bands.mean()
        print(L2_loss.item())

    generated_S2 = np.squeeze(generated_S2.cpu().numpy())

    visible_light_band = generated_S2[:3]
    visible_light_band = visible_light_band * 850.08 + 1599.95

    visible_light_band = visible_light_band.clip(0, 10000)

    # 确保波段数据在0-1范围内
    visible_light_band = (visible_light_band - np.min(visible_light_band)) / (
            np.max(visible_light_band) - np.min(visible_light_band))

    # 创建RGB图像
    rgb = np.dstack((visible_light_band[2], visible_light_band[1], visible_light_band[0]))

    plt.imshow(rgb)
    plt.show()



if __name__ == '__main__':
    generator = torch.load(r"D:\Code\MODIS_S1_S2\model\GAN_generator_epoch_100.pth")
    _, _, test_dataloader = get_dataloader(1, *get_dataset())
    for i, images in enumerate(test_dataloader):
        if i == 0:
            MODIS_image, S1_image, S2_image = images
            generate_image(generator, MODIS_image.cuda(), S1_image.cuda(), S2_image.cuda())
            break

