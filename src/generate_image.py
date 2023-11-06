import matplotlib.pyplot as plt
import numpy as np
import torch
from load_dataset import get_dataloader, get_dataset
from src.utils import generated_S2_to_rgb


def generate_image(generator, MODIS_image, S1_image, save_path=None):
    generator.eval()
    with torch.no_grad():
        generated_S2 = generator(MODIS_image, S1_image)

    rgb = generated_S2_to_rgb(generated_S2)
    rgb = np.transpose(rgb, (1, 2, 0))

    plt.imshow(rgb)
    # plt.show()
    if save_path is not None:
        plt.savefig(save_path)


if __name__ == '__main__':
    generator = torch.load(r"D:\Code\MODIS_S1_S2\model\pre_train_generator_epoch_300.pth")
    _, _, test_dataloader = get_dataloader(1, *get_dataset())
    save_dir = r"D:\Code\MODIS_S1_S2\output\generated_image\\"
    for i, images in enumerate(test_dataloader):
        if i < 10:
            MODIS_image, S1_image, S2_image = images
            save_path = save_dir+str(i)+".png"
            generate_image(generator, MODIS_image.cuda(), S1_image.cuda(), save_path)
            print(save_path)
