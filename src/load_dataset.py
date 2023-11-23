import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from utils import get_image_path


class SatelliteImageDataset(Dataset):
    def __init__(self, group_image_paths, transform=None, data_augment=False):
        self.group_image_paths = group_image_paths
        self.transform = transform
        self.data_augment = data_augment

    def __len__(self):
        return len(self.group_image_paths)

    def __getitem__(self, idx):
        group_image_path = self.group_image_paths[idx]
        MODIS_image_path, S1_image_path, S2_image_path, ref_image_path = group_image_path

        # Pytorch 默认使用 float32
        # Totensor 会改变维度顺序，因此提前调整
        MODIS_image = np.load(MODIS_image_path).astype('float32').transpose((1, 2, 0))
        S1_image = np.load(S1_image_path).astype('float32').transpose((1, 2, 0))
        S2_image = np.load(S2_image_path).astype('float32').transpose((1, 2, 0))
        ref_image = np.load(ref_image_path).astype('float32').transpose((1, 2, 0))

        if self.transform is not None:
            MODIS_image = self.transform["MODIS"](MODIS_image)
            S1_image = self.transform["S1"](S1_image)
            S2_image = self.transform["S2"](S2_image)
            ref_image = self.transform["ref"](ref_image)

        if self.data_augment:
            horizontal_flip_prob = np.random.choice([0, 1])
            vertical_flip_prob = np.random.choice([0, 1])
            flip_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomVerticalFlip(vertical_flip_prob),
                torchvision.transforms.RandomHorizontalFlip(horizontal_flip_prob),
            ])

            MODIS_image = flip_transform(MODIS_image)
            S1_image = flip_transform(S1_image)
            S2_image = flip_transform(S2_image)
            ref_image = flip_transform(ref_image)

        return MODIS_image, S1_image, S2_image, ref_image


def get_dataset():
    MODIS_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(
                                                          mean=[0.5] * 6,
                                                          std=[0.5] * 6),
                                                      ])
    S1_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       mean=[0.5] * 2,
                                                       std=[0.5] * 2),
                                                   ])
    S2_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       mean=[0.5] * 8,
                                                       std=[0.5] * 8),
                                                   ])
    ref_transform = S2_transform
    transforms = {"MODIS": MODIS_transform, "S1": S1_transform, "S2": S2_transform, "ref": ref_transform}

    train_image_paths, val_image_paths, test_image_paths = get_image_path()
    train_dataset = SatelliteImageDataset(train_image_paths, transform=transforms, data_augment=True)
    val_dataset = SatelliteImageDataset(val_image_paths, transform=transforms, data_augment=True)
    test_dataset = SatelliteImageDataset(test_image_paths, transform=transforms, data_augment=True)

    return train_dataset, val_dataset, test_dataset


def get_dataloader(batch_size, train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = get_dataset()
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    print(train_dataset[0][0].shape)