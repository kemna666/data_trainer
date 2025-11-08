import os
import gzip
import struct
import numpy as np
        
def process_mnist(root_path):
    image_pathes = {
        os.path.join(root_path,'train-images-idx3-ubyte.gz'),
        os.path.join(root_path,'t10k-images-idx3-ubyte.gz')
    }
    for path in image_pathes:
         with gzip.open(path,'rb') as file:
            #读取文件头部元数据
            magic,num_images,num_rows,num_cols = struct.unpack(">IIII",file.read(16))
            #读取图像
            image_data = np.frombuffer(file.read(),dtype = np.uint8)
            #塑形数据
            if 'train' in path:
                x_train = image_data.reshape(num_images,num_rows,num_cols).reshape(-1,28,28)
            else:
                x_val = image_data.reshape(num_images,num_rows,num_cols).reshape(-1,28,28)
    label_pathes = {
        os.path.join(root_path,'train-labels-idx1-ubyte.gz'),
        os.path.join(root_path,'t10k-labels-idx1-ubyte.gz')
    }
    for path in label_pathes:
        with gzip.open(path,'rb') as file:
            magic,num_labels = struct.unpack(">II",file.read(8))
            labels = np.frombuffer(file.read(),dtype = np.uint8)
            if 'train' in path:
                y_train = labels
            else:
                y_val = labels
    np.savez_compressed(os.path.join(root_path,'mnist.npz'), x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)