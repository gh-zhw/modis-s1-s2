import numpy as np

# 数据集总数
N = 72

data_index = np.arange(N)
np.random.shuffle(data_index)

train_data_index, val_data_index, test_data_index = np.split(data_index, (int(N * 0.7), int(N * 0.8)))

print(train_data_index, val_data_index, test_data_index, sep="\n")


def data_index_to_txt(data_index, txt_path):
    with open(txt_path, "w") as file:
        for index in data_index:
            image_path = [str(i) for i in range(index * 21, index * 21 + 21)]
            image_path = "\n".join(image_path)
            file.write(image_path + "\n")


if __name__ == "__main__":
    data_index_to_txt(train_data_index, r"D:\Code\MODIS_S1_S2\dataset\ImageSets\train.txt")
    data_index_to_txt(val_data_index, r"D:\Code\MODIS_S1_S2\dataset\ImageSets\val.txt")
    data_index_to_txt(test_data_index, r"D:\Code\MODIS_S1_S2\dataset\ImageSets\test.txt")
