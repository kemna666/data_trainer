from flax import nnx
from functools import partial

class CNN(nnx.Module):
    def __init__(self,in_channels:int,base_channels:int,kernel_size:tuple[int,int],num_classes:int,img_size:tuple[int,int],*,rngs:nnx.Rngs):
        self.conv1 = nnx.Conv(in_features=in_channels,out_features=base_channels,kernel_size=kernel_size,rngs=rngs)
        self.conv2 = nnx.Conv(base_channels,2*base_channels,kernel_size=kernel_size,rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool,window_shape=(2,2),strides=(2,2))
        h,w = img_size
        input_linear = (h//(2*2)) * (w//(2*2)) * (2*base_channels)
        self.linear1 = nnx.Linear(input_linear,256,rngs=rngs)
        self.linear2 = nnx.Linear(256,num_classes,rngs=rngs)
    def __call__(self,x):
        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.avg_pool(x)
        x = self.conv2(x)
        x = nnx.relu(x)
        x = self.avg_pool(x)
        #展平为一维数组
        x = x.reshape(x.shape[0],-1)
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x