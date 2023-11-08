import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def gradient_penalty(discriminator, real, fake, device):
    import torch
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.randn(size=(BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    mixed_scores = discriminator(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penality = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penality


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
    MODIS_dir = r"D:\Code\MODIS_S1_S2\dataset\SatelliteImages\MODIS\\MODIS_"
    S1_dir = r"D:\Code\MODIS_S1_S2\dataset\SatelliteImages\S1\\S1_"
    S2_dir = r"D:\Code\MODIS_S1_S2\dataset\SatelliteImages\S2\\S2_"

    train_data_index, val_data_index, test_data_index = get_data_index()
    train_image_paths, val_image_paths, test_image_paths = [], [], []
    for index in train_data_index:
        train_image_paths.append([MODIS_dir + index + ".npy", S1_dir + index + ".npy", S2_dir + index + ".npy"])
    for index in val_data_index:
        val_image_paths.append([MODIS_dir + index + ".npy", S1_dir + index + ".npy", S2_dir + index + ".npy"])
    for index in test_data_index:
        test_image_paths.append([MODIS_dir + index + ".npy", S1_dir + index + ".npy", S2_dir + index + ".npy"])

    return train_image_paths, val_image_paths, test_image_paths


def calc_statistics(image_paths):
    images = []
    for image_path in image_paths:
        images.append(np.load(image_path))

    min_val = np.min(images, axis=(0, 2, 3))
    max_val = np.max(images, axis=(0, 2, 3))
    mean = np.mean(images, axis=(0, 2, 3))
    std = np.std(images, axis=(0, 2, 3))

    return min_val, max_val, mean, std


def generated_S2_to_rgb(generated_S2_image):
    S2_mean = np.array([934.77, 1133.94, 1127.39])[:, np.newaxis, np.newaxis]
    S2_std = np.array([486.39, 491.88, 554.30])[:, np.newaxis, np.newaxis]
    rgb = generated_S2_image[:, :3, :, :].cpu().numpy()
    rgb = np.squeeze(rgb)
    rgb = rgb[[2, 1, 0], :, :]
    rgb = rgb * S2_std + S2_mean
    rgb = np.clip(rgb, 0, 10000)
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
    return rgb


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
    # plot_loss(r"D:\Code\MODIS_S1_S2\output\loss\test\Instance Normalization\pre_train_generator_train_loss.npy", r"D:\Code\MODIS_S1_S2\output\loss\loss_plot\\", x_label="step", xtick_gain=10)
    plot_loss(r"D:\Code\MODIS_S1_S2\output\loss\test\Instance Normalization\pre_train_generator_val_loss.npy", r"D:\Code\MODIS_S1_S2\output\loss\loss_plot\\", x_label="epoch")
