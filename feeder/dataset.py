import numpy as np
from functools import partial
from .mnist import Mnist,transform,loader


dataset_registry = {
    'mnist':Mnist
}

def choose_dataloader(dataset_dict,data,label,num_epochs,dtype=np.float32):
    if dataset_dict['dataset'] =='MNIST':
        print('mnist')
        return loader(dataset = dataset_dict['path'], data=data, label=label,batch_size=dataset_dict['batch_size'],num_epochs=num_epochs,dtype=dtype)