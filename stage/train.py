import os
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import tqdm



@nnx.jit(static_argnames=["loss_fn"]) 
def train_step(model,loss_fn,optimizer:nnx.Optimizer,metrics:nnx.MultiMetric,batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch,training=True)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
    optimizer.update(model, grads)  # In-place updates.
@nnx.jit(static_argnames=["loss_fn"]) 
def eval_step(model,metrics:nnx.MultiMetric,batch,loss_fn):
  loss, logits = loss_fn(model, batch,training=False)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.


def process_train(train_loader,val_loader,loss_function,model,optimizer,metrics,num_epochs):
    best_acc = 0
    step = 0
    
    # 获取一批验证数据用于定期验证
    val_batches = list(val_loader)
    
    # 将训练数据转换为列表，以便我们可以按epochs处理
    train_batches = list(train_loader)
    steps_per_epoch = len(train_batches)
    total_steps = steps_per_epoch * num_epochs
    
    print(f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}")
    
    # 按epochs进行训练
    for epoch in range(num_epochs):
        # 训练一个epoch
        pbar = tqdm(train_batches, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            train_step(model, loss_function, optimizer, metrics, batch)
            step += 1
            pbar.set_postfix({'loss': f"{metrics.compute()['loss']:.4f}"})
        # 在每个epoch结束后进行验证
        train_metrics = metrics.compute()
        print(f"Epoch:{epoch+1}_Train Acc@1: {train_metrics['accuracy']:.4f} loss: {train_metrics['loss']:.4f}")
        metrics.reset()  # 重置指标
        
        val_pbar = tqdm(val_batches, desc=f"Validation {epoch+1}/{num_epochs}", leave=False)
        for val_batch in val_pbar:
            eval_step(model, metrics, val_batch, loss_function)
        
        val_metrics = metrics.compute()
        print(f"Epoch:{epoch+1}_Val Acc@1: {val_metrics['accuracy']:.4f} loss: {val_metrics['loss']:.4f}")
        
        if val_metrics['accuracy'] > best_acc:
            save_checkpoint(model, step)
            best_acc = val_metrics['accuracy']
        
        metrics.reset()  

def save_checkpoint(model,step):
    # save checkpoint
      with ocp.CheckpointManager(os.path.join(os.getcwd(), 'checkpoints/'),options = ocp.CheckpointManagerOptions(max_to_keep=1),) as mngr:
        _, state = nnx.split(model)
        mngr.save(step, args=ocp.args.StandardSave(state))