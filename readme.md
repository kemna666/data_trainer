## 简介
这个是一些简单数据集的训练代码
## 使用方法
- step1:下载数据集，放在 data/${dataset_name}文件夹
- step2:执行以下命令进行数据预处理
~~~bash 
python prepare_data.py ${dataset} ${data_path}
~~~
- step3:执行以下命令跑训练流程
~~~bash
python main.py ${config}
~~~

## 数据集下载链接
mnist:
[Training images](https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz),
[Training labels](https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz)
[Testing images](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz),
[Testing labels](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz)

cifar-10:
https://www.cs.toronto.edu/~kriz/cifar.html
