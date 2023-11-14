import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from skimage.metrics import structural_similarity


def calc_statistics(image_paths):
    images = []
    for image_path in image_paths:
        images.append(np.load(image_path))

    min_val = np.min(images, axis=(0, 2, 3))
    max_val = np.max(images, axis=(0, 2, 3))
    mean = np.mean(images, axis=(0, 2, 3))
    std = np.std(images, axis=(0, 2, 3))

    return min_val, max_val, mean, std


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

    test_MODIS_dir = r"D:\Code\MODIS_S1_S2\dataset\SatelliteImages\test\MODIS\MODIS_"
    test_S1_dir = r"D:\Code\MODIS_S1_S2\dataset\SatelliteImages\test\S1\S1_"
    test_S2_dir = r"D:\Code\MODIS_S1_S2\dataset\SatelliteImages\test\S2\S2_"

    train_data_index, val_data_index, test_data_index = get_data_index()
    train_image_paths, val_image_paths, test_image_paths = [], [], []
    for index in train_data_index:
        train_image_paths.append([train_MODIS_dir + index + ".npy", train_S1_dir + index + ".npy", train_S2_dir + index + ".npy"])
    for index in val_data_index:
        val_image_paths.append([train_MODIS_dir + index + ".npy", train_S1_dir + index + ".npy", train_S2_dir + index + ".npy"])
    for index in test_data_index:
        test_image_paths.append([test_MODIS_dir + index + ".npy", test_S1_dir + index + ".npy", test_S2_dir + index + ".npy"])

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
    S2_mean = np.array([978.63, 1200.54, 1157.47])[:, np.newaxis, np.newaxis]
    S2_std = np.array([576.95, 588.16, 634.96])[:, np.newaxis, np.newaxis]
    rgb = generated_S2_image[:, :3, :, :].cpu().numpy()
    rgb = np.squeeze(rgb)
    rgb = rgb[:, [2, 1, 0], :, :]
    rgb = rgb * S2_std + S2_mean
    rgb = np.clip(rgb, 0, 10000)
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
    return rgb


def L2_Loss_for_bands(prediction, target):
    L2_loss_bands = ((prediction - target) ** 2).sum(dim=(0, 2, 3))
    L2_loss_bands /= (target.shape[0] * target.shape[2] * target.shape[3])
    return L2_loss_bands


def L1_Loss_for_bands(prediction, target):
    L1_loss_bands = torch.abs((prediction - target)).sum(dim=(0, 2, 3))
    L1_loss_bands /= (target.shape[0] * target.shape[2] * target.shape[3])
    return L1_loss_bands


def calc_metric(prediction, target, max_value, data_range, output="bands"):
    prediction = np.array(prediction)
    target = np.array(target)

    band_num = prediction.shape[1]
    mae_bands = np.zeros(band_num)
    rmse_bands = np.zeros(band_num)
    psnr_bands = np.zeros(band_num)
    ssim_bands = np.zeros(band_num)

    for i in range(band_num):
        target_band = np.squeeze(target[:, i, :, :])
        prediction_band = np.squeeze(prediction[:, i, :, :])

        # calculate MAE
        mae = np.abs(target_band - prediction_band).mean()
        mae_bands[i] = mae

        # calculate RMSE
        rmse = np.sqrt(((target_band - prediction_band) ** 2).mean())
        rmse_bands[i] = rmse

        # calculate PSNR
        psnr = 20 * np.log10(max_value / rmse)
        psnr_bands[i] = psnr

        # calculate SSIM
        ssim = structural_similarity(target_band, prediction_band, data_range=data_range)
        ssim_bands[i] = ssim

    if output == "bands":
        return mae_bands, rmse_bands, psnr_bands, ssim_bands
    elif output == "mean":
        return mae_bands.mean(), rmse_bands.mean(), psnr_bands.mean(), ssim_bands.mean()


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
    plot_loss(r"D:\Code\MODIS_S1_S2\output\loss\test\Instance Normalization\pre_train_generator_train_loss.npy",
              r"D:\Code\MODIS_S1_S2\output\loss\loss_plot\\", x_label="step", xtick_gain=10)
    plot_loss(r"D:\Code\MODIS_S1_S2\output\loss\test\Instance Normalization\pre_train_generator_val_loss.npy",
              r"D:\Code\MODIS_S1_S2\output\loss\loss_plot\\", x_label="epoch")
