import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from pytorch_msssim import ssim


# 从.txt获取训练集、验证集和测试集的数据编号
def get_data_index():
    train_txt_path = r"D:\Code\MODIS_S1_S2\dataset\ImageSets\train.txt"
    val_txt_path = r"D:\Code\MODIS_S1_S2\dataset\ImageSets\val.txt"
    test_txt_path = r"D:\Code\MODIS_S1_S2\dataset\ImageSets\test.txt"

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
    train_MODIS_dir = r"D:\Code\MODIS_S1_S2\dataset\SatelliteImages\train\MODIS\MODIS_"
    train_S1_dir = r"D:\Code\MODIS_S1_S2\dataset\SatelliteImages\train\S1\S1_"
    train_S2_dir = r"D:\Code\MODIS_S1_S2\dataset\SatelliteImages\train\S2\S2_"
    train_ref_dir = r"D:\Code\MODIS_S1_S2\dataset\SatelliteImages\train\ref\ref_"

    test_MODIS_dir = r"D:\Code\MODIS_S1_S2\dataset\SatelliteImages\test\MODIS\MODIS_"
    test_S1_dir = r"D:\Code\MODIS_S1_S2\dataset\SatelliteImages\test\S1\S1_"
    test_S2_dir = r"D:\Code\MODIS_S1_S2\dataset\SatelliteImages\test\S2\S2_"
    test_ref_dir = r"D:\Code\MODIS_S1_S2\dataset\SatelliteImages\test\ref\ref_"

    train_data_index, val_data_index, test_data_index = get_data_index()
    train_image_paths, val_image_paths, test_image_paths = [], [], []
    for index in train_data_index:
        train_image_paths.append([train_MODIS_dir + index + ".npy",
                                  train_S1_dir + index + ".npy",
                                  train_S2_dir + index + ".npy",
                                  train_ref_dir + index + ".npy"])
    for index in val_data_index:
        val_image_paths.append([train_MODIS_dir + index + ".npy",
                                train_S1_dir + index + ".npy",
                                train_S2_dir + index + ".npy",
                                train_ref_dir + index + ".npy"])
    for index in test_data_index:
        test_image_paths.append([test_MODIS_dir + index + ".npy",
                                 test_S1_dir + index + ".npy",
                                 test_S2_dir + index + ".npy",
                                 test_ref_dir + index + ".npy"])

    return train_image_paths, val_image_paths, test_image_paths


def gradient_penalty(critic, real, fake, device='cpu'):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.randn(size=(BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


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
    rmse = torch.sqrt(mse)
    psnr = 20 * torch.log10(max_value / rmse)

    # SSIM
    ssim_value = ssim((prediction + 1) / 2, (target + 1) / 2, data_range=data_range)

    return mae, mse, sam, psnr, ssim_value


def plot_loss(loss_npy, loss_pic_dir, x_label, xtick_gain=1):
    loss_dict = np.load(loss_npy, allow_pickle=True).item()
    plt.figure(dpi=300, figsize=(12, 8))
    for i, loss_key in enumerate(loss_dict.keys()):
        loss_value = loss_dict[loss_key]
        y_min_value = min(loss_value)
        y_max_value = max(loss_value)
        len_range = np.arange(len(loss_value))
        plt.xlabel(x_label)
        plt.ylabel("loss")
        plt.plot(len_range, loss_value, label=loss_key, c="r")
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator((len_range[-1] - len_range[0]) // 10))
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator((y_max_value - y_min_value) / 10))
        plt.grid()
        plt.legend()
        plt.savefig(loss_pic_dir + loss_key + ".png")
        # 清空绘图
        plt.cla()


if __name__ == '__main__':
    # plot_loss(r"D:\Code\MODIS_S1_S2\output\loss\test\Instance Normalization\pre_train_generator_train_loss.npy",
    #           r"D:\Code\MODIS_S1_S2\output\loss\loss_plot\\", x_label="step", xtick_gain=10)
    # plot_loss(r"D:\Code\MODIS_S1_S2\output\loss\test\Instance Normalization\pre_train_generator_val_loss.npy",
    #           r"D:\Code\MODIS_S1_S2\output\loss\loss_plot\\", x_label="epoch")

    a = torch.rand((12, 8, 250, 250))
    b = torch.ones((12, 8, 250, 250))

    mae, mse, sam, psnr, ssim_value = calc_metric(a, b)
    print(mae, mse, sam, psnr, ssim_value)
