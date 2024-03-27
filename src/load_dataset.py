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
        # MODIS_image_path, S1_image_path, S2_image_path, ref_image_path = group_image_path
        MODIS_image_path, S1_image_path, S2_image_path, before_image_path, after_image_path = group_image_path

        # Pytorch 默认使用 float32
        # Totensor 会改变维度顺序，因此提前调整
        MODIS_image = np.load(MODIS_image_path).astype('float32').transpose((1, 2, 0))
        S1_image = np.load(S1_image_path).astype('float32').transpose((1, 2, 0))
        S2_image = np.load(S2_image_path).astype('float32').transpose((1, 2, 0))
        # ref_image = np.load(ref_image_path).astype('float32').transpose((1, 2, 0))
        before_image = np.load(before_image_path).astype('float32').transpose((1, 2, 0))
        after_image = np.load(after_image_path).astype('float32').transpose((1, 2, 0))

        if self.transform is not None:
            MODIS_image = self.transform["MODIS"](MODIS_image)
            S1_image = self.transform["S1"](S1_image)
            S2_image = self.transform["S2"](S2_image)
            # ref_image = self.transform["ref"](ref_image)
            before_image = self.transform["before"](before_image)
            after_image = self.transform["after"](after_image)

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
            # ref_image = flip_transform(ref_image)
            before_image = flip_transform(before_image)
            after_image = flip_transform(after_image)

        # return MODIS_image, S1_image, S2_image, ref_image
        return MODIS_image, S1_image, S2_image, before_image, after_image


MODIS_mean = [0.0595, 0.1577, 0.0420, 0.0608, 0.1135, 0.0707]
MODIS_std = [0.0239, 0.0405, 0.0238, 0.0228, 0.0273, 0.0221]
S1_mean = [0.7964, 0.6542]
S1_std = [0.0804, 0.0834]
S2_mean = [0.0943, 0.1160, 0.1117, 0.1502,  0.2550, 0.2617, 0.1975, 0.1478]
S2_std = [0.0561, 0.0565, 0.0614, 0.0602, 0.1067, 0.1078, 0.0790, 0.0733]


def get_dataset():
    MODIS_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(
                                                          mean=MODIS_mean,
                                                          std=MODIS_std),
                                                      ])
    S1_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       mean=S1_mean,
                                                       std=S1_std),
                                                   ])
    S2_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       mean=[0.5] * 8,
                                                       std=[0.5] * 8),
                                                   ])
    ref_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        mean=S2_mean,
                                                        std=S2_mean),
                                                    ])
    before_transform = ref_transform
    after_transform = ref_transform
    transforms = {"MODIS": MODIS_transform, "S1": S1_transform, "S2": S2_transform, "ref": ref_transform,
                  "before": before_transform, "after": after_transform}

    train_image_paths, val_image_paths, test_image_paths = get_image_path()
    train_dataset = SatelliteImageDataset(train_image_paths, transform=transforms, data_augment=True)
    val_dataset = SatelliteImageDataset(val_image_paths, transform=transforms, data_augment=False)
    test_dataset = SatelliteImageDataset(test_image_paths, transform=transforms, data_augment=False)

    return train_dataset, val_dataset, test_dataset


def get_dataloader(batch_size, train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = get_dataset()
    print(len(train_dataset), len(val_dataset), len(test_dataset))
