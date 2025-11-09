import os
import orbax.checkpoint as ocp
from flax import nnx




@nnx.jit(static_argnames=["loss_fn"]) 
def train_step(model,loss_fn,optimizer:nnx.Optimizer,metrics:nnx.MultiMetric,batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
    optimizer.update(model, grads)  # In-place updates.
@nnx.jit(static_argnames=["loss_fn"]) 
def eval_step(model,metrics:nnx.MultiMetric,batch,loss_fn):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.


def process_train(train_loader,val_loader,loss_function,model,optimizer,metrics):
    best_acc = 0
    for step, batch in enumerate(train_loader):
        train_step(model,loss_function, optimizer, metrics, batch)
        if step > 0 and step % 1000 == 0:
            train_metrics = metrics.compute()
            print(f"Step:{step}_Train Acc@1: {train_metrics['accuracy']:.4f} loss: {train_metrics['loss']:.4f}")
            metrics.reset()  # Reset the metrics for the train set.
            # Compute the metrics on the test set after each training epoch.
            for val_batch in val_loader:
                eval_step(model, metrics, val_batch,loss_function)
            val_metrics = metrics.compute()
            print(f"Step:{step}_Val Acc@1: {val_metrics['accuracy']:.4f} loss: {val_metrics['loss']:.4f}")
            if val_metrics['accuracy'] > best_acc:
                  save_checkpoint(model,step)
                  best_acc = val_metrics['accuracy']
            metrics.reset()  # Reset the metrics for the val set.

def save_checkpoint(model,step):
    # save checkpoint
      with ocp.CheckpointManager(os.path.join(os.getcwd(), 'checkpoints/'),options = ocp.CheckpointManagerOptions(max_to_keep=1),) as mngr:
        _, state = nnx.split(model)
        mngr.save(step, args=ocp.args.StandardSave(state))