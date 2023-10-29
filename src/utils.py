import numpy as np


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


