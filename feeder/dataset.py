import numpy as np

from . import mnist,cifra



def choose_dataloader(dataset_dict,data,label,num_epochs=1,dtype=np.float32):
    if dataset_dict['dataset'].lower() =='mnist':
        return mnist.loader(dataset_dict['path'], data=data, label=label,batch_size=dataset_dict['batch_size'],num_epochs=num_epochs,dtype=dtype)
    elif dataset_dict['dataset'].lower() =='cifra':
        return cifra.loader(dataset_dict['path'], data=data, label=label,batch_size=dataset_dict['batch_size'],num_epochs=num_epochs,dtype=dtype)