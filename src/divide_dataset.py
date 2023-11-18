import numpy as np

# 数据集总数
N = 2194

data_index = np.arange(N)
np.random.shuffle(data_index)

train_data_index = data_index[:int(N * 0.9)]
val_data_index = data_index[int(N * 0.9):]
test_data_index = list(range(66))
print(len(train_data_index), len(val_data_index))


def data_index_to_txt(data_index, txt_path):
    with open(txt_path, "w") as file:
        image_path = "\n".join([str(index) for index in data_index])
        file.write(image_path + "\n")


if __name__ == "__main__":
    data_index_to_txt(train_data_index, r"D:\Code\MODIS_S1_S2\dataset\ImageSets\train.txt")
    data_index_to_txt(val_data_index, r"D:\Code\MODIS_S1_S2\dataset\ImageSets\val.txt")
    data_index_to_txt(test_data_index, r"D:\Code\MODIS_S1_S2\dataset\ImageSets\test.txt")