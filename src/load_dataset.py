import glob
from torch.utils.data import Dataset
import numpy as np

MODIS_dir = "../dataset/MODIS/MODIS_*.npy"
S1_dir = "../dataset/Sentinel-1/S1_*.npy"
S2_dir = "../dataset/Sentinel-2/S2_*.npy"

MODIS_image_paths = glob.glob(MODIS_dir)
S1_image_paths = glob.glob(S1_dir)
S2_image_paths = glob.glob(S2_dir)
group_image_paths = list(zip(MODIS_image_paths, S1_image_paths, S2_image_paths))
group_num = len(group_image_paths)


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
    sate_data = SatelliteImageDataset(group_image_paths)
    print(len(sate_data))
    print(sate_data[0][0].shape, sate_data[0][1].shape, sate_data[0][2].shape)
    # sate_data[0]
