import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
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


MODIS_mean = [956.77, 2256.87, 684.35, 951.04, 1783.60, 1115.68]
MODIS_std = [464.25, 592.44, 471.64, 440.68, 500.87, 450.20]
S1_mean = [-8.70, -16.90]
S1_std = [3.64, 3.69]
S2_mean = [934.77, 1133.94, 1127.39, 1467.05617328, 2378.66, 2400.76, 1922.42, 1434.61]
S2_std = [486.39, 491.88, 554.30, 526.60, 910.23, 906.85, 644.25, 610.28]


def get_dataset():
    MODIS_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(mean=MODIS_mean, std=MODIS_std)
                                                      ])
    S1_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(mean=S1_mean, std=S1_std)
                                                   ])
    S2_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(mean=S2_mean, std=S2_std)
                                                   ])
    transforms = {"MODIS": MODIS_transform, "S1": S1_transform, "S2": S2_transform}

    train_image_paths, val_image_paths, test_image_paths = get_image_path()
    train_dataset = SatelliteImageDataset(train_image_paths, transform=transforms)
    val_dataset = SatelliteImageDataset(val_image_paths, transform=transforms)
    test_dataset = SatelliteImageDataset(test_image_paths, transform=transforms)

    return train_dataset, val_dataset, test_dataset


def get_dataloader(batch_size, train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = get_dataset()
    print(len(train_dataset[0]))
