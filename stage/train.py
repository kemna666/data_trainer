import optax
import os
import orbax.checkpoint as ocp
from flax import nnx

@nnx.jit
def train_step(model,loss_function,optimizer:nnx.Optimizer,metrics:nnx.MultiMetric,batch):
    grad_fn = nnx.value_and_grad(loss_function,has_aux=True)
    (loss,logits),grads = grad_fn(model,batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label']) 
    optimizer.update(model,grads)

@nnx.jit
def eval_step(model,metrics:nnx.MultiMetric,batch,loss_fn):
    loss,logits = loss_fn(model,batch)
    metrics.update(loss,logits,labels=batch['label'])

def process_train(train_loader,val_loader,model,optimizer,metrics):
    for step, batch in enumerate(train_loader):
        train_step(model, optimizer, metrics, batch)
        if step > 0 and step % 1000 == 0:
            train_metrics = metrics.compute()
            print("Step:{}_Train Acc@1: {} loss: {} ".format(step,train_metrics['accuracy'],train_metrics['loss']))
            metrics.reset()  # Reset the metrics for the train set.
            # Compute the metrics on the test set after each training epoch.
            for val_batch in val_loader:
                eval_step(model, metrics, val_batch)
            val_metrics = metrics.compute()
            print("Step:{}_Val Acc@1: {} loss: {} ".format(step,val_metrics['accuracy'],val_metrics['loss']))
            if val_metrics['accuracy'] > best_acc:
                  save_checkpoint(model,step)
                  best_acc = val_metrics['accuracy']
            metrics.reset()  # Reset the metrics for the val set.

def save_checkpoint(model,step):
    # save checkpoint
      with ocp.CheckpointManager(os.path.join(os.getcwd(), 'checkpoints/'),options = ocp.CheckpointManagerOptions(max_to_keep=1),) as mngr:
        _, state = nnx.split(model)
        mngr.save(step, args=ocp.args.StandardSave(state))