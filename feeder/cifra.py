import numpy as np
from loaderx import Dataset, DataLoader
from functools import partial


def transform_cifar(batch, dtype):
    # CIFAR-10图像为3通道(RGB)，形状为(样本数, 高, 宽, 通道)
    # 归一化到[0, 1]并转换数据类型
    images = (batch[0] / 255.0).astype(dtype)
    # 标签转换为int32
    labels = batch[1].astype(np.int32)
    return images, labels


class CIFAR10(Dataset):
    def __init__(self, dataset_path, data, label, group_size=1):
        super().__init__(dataset_path, data, label, group_size)


def loader(dataset_path,data,label,batch_size,num_epochs=1,dtype=np.float32):

    return DataLoader(
        dataset=CIFAR10(dataset_path, data, label),
        batch_size=batch_size,
        num_epochs=num_epochs,
        transform=partial(transform_cifar, dtype=dtype)
    )