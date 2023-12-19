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


MODIS_mean = [0.0860, 0.2475, 0.0582, 0.0888, 0.1735, 0.1034]
MODIS_std = [0.0397, 0.0671, 0.0395, 0.0378, 0.0455, 0.0365]
S1_mean = [0.7915, 0.6517]
S1_std = [0.0792, 0.0840]
S2_mean = [0.0981, 0.1204, 0.1160, 0.1552, 0.2619, 0.2685, 0.2030, 0.1526]
S2_std = [0.0576, 0.0586, 0.0634, 0.0633, 0.1113, 0.1128, 0.0819, 0.0751]


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
    val_dataset = SatelliteImageDataset(val_image_paths, transform=transforms, data_augment=True)
    test_dataset = SatelliteImageDataset(test_image_paths, transform=transforms, data_augment=False)

    return train_dataset, val_dataset, test_dataset


def get_dataloader(batch_size, train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = get_dataset()
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    print(train_dataset[0][0])
