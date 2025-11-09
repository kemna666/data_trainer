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
