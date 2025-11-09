import tomllib
import argparse
from pathlib import Path
from flax import nnx
import optax
import jax.numpy as jnp
from feeder.mnist import Mnist
from stage.train import process_train
from model.model_registry import choose_model 
from feeder.dataset import choose_dataloader
from utils.loss_function import loss_fn


if __name__ =='__main__':    
    parser = argparse.ArgumentParser(description='Data2Lantent')
    parser.add_argument('config',help='配置文件')
    args = parser.parse_args()
    tomlfile = Path(args.config)
    with open(tomlfile,'rb') as file:
        configs = tomllib.load(file)
    rngs=nnx.Rngs(0)
    metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss'),
    )
    num_epochs = configs['train']['epoches']
    model = choose_model(configs['model'],rngs=rngs)
    loss_function = loss_fn(configs['train']['loss_function'])
    train_loader = choose_dataloader(configs['dataset'],data='x_train',label='y_train',num_epochs=num_epochs,dtype=jnp.float32)
    val_loader = choose_dataloader(configs['dataset'],data='x_val',label='y_val',dtype=jnp.float32)
    tx=optax.chain(
    optax.clip_by_global_norm(1.0),  # 添加梯度裁剪
    optax.adamw(
        learning_rate=configs['train']['learning_rate'],
        b1=0.9,
        weight_decay=1e-4
    )
)
    print(configs['train']['learning_rate'])
    optimizer = nnx.Optimizer(model,tx, wrt=nnx.Param)
    process_train(train_loader=train_loader,val_loader=val_loader,loss_function=loss_function,model=model,optimizer=optimizer,metrics=metrics)