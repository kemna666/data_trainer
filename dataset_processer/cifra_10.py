import os
import numpy as np
import pickle

def process_cifar10(root_path):
    """
    处理CIFAR-10数据集，将二进制批次文件转换为npz压缩文件
    CIFAR-10原始文件结构：
    - 训练集：data_batch_1 ~ data_batch_5
    - 测试集：test_batch
    - 标签说明：batches.meta（包含类别名称）
    """
    # 定义训练集和测试集文件路径
    train_files = [
        os.path.join(root_path, f'data_batch_{i}') for i in range(1, 6)  # 5个训练批次
    ]
    test_file = os.path.join(root_path, 'test_batch')  # 1个测试批次

    # 初始化存储列表
    x_train, y_train = [], []
    x_val, y_val = [], []

    # 处理训练集
    for file in train_files:
        with open(file, 'rb') as f:
            # CIFAR-10的批次文件是pickle序列化的字典
            data_dict = pickle.load(f, encoding='bytes')  # 注意用bytes解码
            # 图像数据：shape=(10000, 3072)，其中3072=32×32×3（RGB通道）
            images = data_dict[b'data']  # 键为bytes类型
            # 标签数据：shape=(10000,)
            labels = data_dict[b'labels']
            
            # 转换图像形状：(10000, 3072) → (10000, 32, 32, 3)
            # 原始数据存储格式：每个图像按RGB通道平面存储（先所有R，再G，再B）
            images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 转为(样本数, 高, 宽, 通道)
            x_train.append(images)
            y_train.extend(labels)

    # 处理测试集
    with open(test_file, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 同训练集转换
        x_val = images
        y_val = labels

    # 合并训练集（5个批次合并为50000样本）
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.array(y_train, dtype=np.uint8)
    x_val = np.array(x_val, dtype=np.uint8)
    y_val = np.array(y_val, dtype=np.uint8)

    # 保存为npz压缩文件
    save_path = os.path.join(root_path, 'cifar10.npz')
    np.savez_compressed(save_path, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    print(f"CIFAR-10数据处理完成，保存至：{save_path}")
    print(f"训练集：{x_train.shape}, {y_train.shape}")
    print(f"验证集：{x_val.shape}, {y_val.shape}")