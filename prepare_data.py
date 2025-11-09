from dataset_processer import mnist
from dataset_processer.cifra_10 import process_cifar10

def prepare(dataset:str,root_path:str):
    datasets = {
        'mnist': mnist.process_mnist,
        'cifra-10':process_cifar10 
    }
    datasets[dataset](root_path)

prepare('cifra-10','data/cifra_10')