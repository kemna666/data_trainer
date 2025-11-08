import numpy as np
from loaderx import Dataset,DataLoader

def transform(batch, dtype):
    return (batch[0][..., None] / 255.0).astype(dtype) , batch[1].astype(np.int32)

class Mnist(Dataset):
  def __init__(self, dataset_path, data, label, group_size=1):
    super().__init__(dataset_path, data, label, group_size)

def loader(dataset_path, data, label, batch_size, num_epochs=1, dtype=np.float32):
  return DataLoader(dataset = Mnist(dataset_path, data, label),batch_size=batch_size,num_epochs=num_epochs,transform=partial(transform, dtype=dtype))
