import numpy as np
from torch.utils.data import Dataset
from utils import get_image_path


class SatelliteImageDataset(Dataset):
    def __init__(self, group_image_paths, transform=None):
        self.group_image_paths = group_image_paths
        self.transform = transform

    def __len__(self):
        return len(self.group_image_paths)

    def __getitem__(self, idx):
        group_image_path = self.group_image_paths[idx]
        MODIS_image_path, S1_image_path, S2_image_path = group_image_path

        # Pytorch 默认使用 float32
        # Totensor 会改变维度顺序，因此提前调整
        MODIS_image = np.load(MODIS_image_path).astype('float32').transpose((1, 2, 0))
        S1_image = np.load(S1_image_path).astype('float32').transpose((1, 2, 0))
        S2_image = np.load(S2_image_path).astype('float32').transpose((1, 2, 0))

        if self.transform is not None:
            MODIS_image = self.transform["MODIS"](MODIS_image)
            S1_image = self.transform["S1"](S1_image)
            S2_image = self.transform["S2"](S2_image)

        return MODIS_image, S1_image, S2_image


if __name__ == '__main__':
    train_image_paths, val_image_paths, test_image_paths = get_image_path()

    train_images = SatelliteImageDataset(train_image_paths)
    val_images = SatelliteImageDataset(val_image_paths)
    test_images = SatelliteImageDataset(test_image_paths)
    print(len(train_images), len(val_images), len(test_images))
    # sate_data[0]
