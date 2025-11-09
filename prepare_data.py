from dataset_processer import mnist
from dataset_processer.cifra_10 import process_cifar10
import argparse


def prepare(dataset:str,root_path:str):
    datasets = {
        'mnist': mnist.process_mnist,
        'cifra-10':process_cifar10 
    }
    datasets[dataset](root_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data2Lantent')
    parser.add_argument('dataset',help='数据集名字')
    parser.add_argument('root_path',help='数据集根目录')
    args = parser.parse_args()
    prepare(args.dataset,args.root_path)