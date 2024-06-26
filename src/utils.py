import numpy as np
import torch
from pytorch_msssim import ssim
import config


# 从.txt获取训练集、验证集和测试集的数据编号
def get_data_index():
    train_txt_path = config.train_data_index_txt
    val_txt_path = config.val_data_index_txt
    test_txt_path = config.test_data_index_txt

    with open(train_txt_path, "r") as file:
        train_data_index = file.readlines()
    train_data_index = [index.strip("\n") for index in train_data_index]
    with open(val_txt_path, "r") as file:
        val_data_index = file.readlines()
    val_data_index = [index.strip("\n") for index in val_data_index]
    with open(test_txt_path, "r") as file:
        test_data_index = file.readlines()
    test_data_index = [index.strip("\n") for index in test_data_index]

    return train_data_index, val_data_index, test_data_index


def get_image_path():
    train_MODIS_dir = config.train_MODIS_dir
    train_S1_dir = config.train_S1_dir
    train_S2_dir = config.train_S2_dir
    # train_ref_dir = config.train_ref_dir
    train_before_dir = config.train_before_dir
    train_after_dir = config.train_after_dir

    test_MODIS_dir = config.test_MODIS_dir
    test_S1_dir = config.test_S1_dir
    test_S2_dir = config.test_S2_dir
    # test_ref_dir = config.test_ref_dir
    test_before_dir = config.test_before_dir
    test_after_dir = config.test_after_dir

    train_data_index, val_data_index, test_data_index = get_data_index()
    train_image_paths, val_image_paths, test_image_paths = [], [], []
    for index in train_data_index:
        train_image_paths.append([train_MODIS_dir + index + ".npy",
                                  train_S1_dir + index + ".npy",
                                  train_S2_dir + index + ".npy",
                                  #   train_ref_dir + index + ".npy",
                                  train_before_dir + index + ".npy",
                                  train_after_dir + index + ".npy"])
    for index in val_data_index:
        val_image_paths.append([train_MODIS_dir + index + ".npy",
                                train_S1_dir + index + ".npy",
                                train_S2_dir + index + ".npy",
                                # train_ref_dir + index + ".npy",
                                train_before_dir + index + ".npy",
                                train_after_dir + index + ".npy"])
    for index in test_data_index:
        test_image_paths.append([test_MODIS_dir + index + ".npy",
                                 test_S1_dir + index + ".npy",
                                 test_S2_dir + index + ".npy",
                                 #  test_ref_dir + index + ".npy",
                                 test_before_dir + index + ".npy",
                                 test_after_dir + index + ".npy"])

    return train_image_paths, val_image_paths, test_image_paths

def generated_S2_to_rgb(generated_S2_image):
    rgb = generated_S2_image[:, :3, :, :]
    rgb = rgb[:, [2, 1, 0], :, :]
    return rgb


def L2_Loss_for_bands(prediction, target):
    L2_loss_bands = ((prediction - target) ** 2).sum(dim=(0, 2, 3))
    L2_loss_bands /= (target.shape[0] * target.shape[2] * target.shape[3])
    return L2_loss_bands


def L1_Loss_for_bands(prediction, target):
    L1_loss_bands = torch.abs((prediction - target)).sum(dim=(0, 2, 3))
    L1_loss_bands /= (target.shape[0] * target.shape[2] * target.shape[3])
    return L1_loss_bands


def spectral_angle_mapper(image1, image2):
    image1_flat = image1.view(image1.size(0), -1)
    image2_flat = image2.view(image2.size(0), -1)

    image1_unit = image1_flat / torch.norm(image1_flat, p=2, dim=1, keepdim=True)
    image2_unit = image2_flat / torch.norm(image2_flat, p=2, dim=1, keepdim=True)

    cos_theta = torch.sum(image1_unit * image2_unit, dim=1)
    angle = torch.acos(cos_theta).mean()

    return torch.rad2deg(angle)


def calc_metric(prediction, target, max_value=1, data_range=1):
    # MAE
    mae = torch.abs(prediction - target).mean()

    # MSE
    mse = ((prediction - target) ** 2).mean()

    # SAM
    sam = spectral_angle_mapper(prediction, target)

    # PSNR
    denormalied_prediction = (prediction + 1) / 2
    denormalized_target = (target + 1) / 2
    denormalized_rmse = torch.sqrt(((denormalied_prediction - denormalized_target) ** 2).mean())
    psnr = 20 * torch.log10(max_value / denormalized_rmse)

    # SSIM
    ssim_value = ssim(denormalied_prediction, denormalized_target, data_range=data_range)

    return mae, mse, sam, psnr, ssim_value


if __name__ == '__main__':
    a = torch.rand((12, 8, 250, 250))
    b = torch.ones((12, 8, 250, 250))

    mae, mse, sam, psnr, ssim_value = calc_metric(a, b)
    print(mae, mse, sam, psnr, ssim_value)
