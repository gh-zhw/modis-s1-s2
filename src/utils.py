import numpy as np


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
    images = np.array(images)

    min_val = np.min(images)
    max_val = np.max(images)
    mean = np.mean(images.reshape(-1))
    std = np.std(images.reshape(-1))

    return min_val, max_val, mean, std


if __name__ == '__main__':
    from load_dataset import MODIS_image_paths, S1_image_paths, S2_image_paths

    min_val, max_val, mean, std = calc_statistics(S2_image_paths)

    print(min_val, max_val, mean, std)
