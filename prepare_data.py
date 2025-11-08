from dataset_processer import mnist
from dataset_processer.cifra_10 import process_cifra


def prepare(dataset:str,root_path:str):
    datasets = {
        'mnist': mnist.process_mnist(root_path),
        'cifra-10':process_cifra(root_path) 
    }
    return datasets.get(dataset,f'{dataset}输入错误，可选项：{datasets}')


prepare('mnist','data/mnist')