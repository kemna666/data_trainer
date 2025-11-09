import optax

def loss_fn(func):
    if func =='softmax_cross_entropy_with_integer_labels':
       return softmax_cross_entropy_with_integer_labels 



def softmax_cross_entropy_with_integer_labels(model, batch,training):
  logits = model(batch['data'],training)
  loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['label']).mean()
  return loss, logits