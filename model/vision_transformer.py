from flax import nnx
import jax.nn as nn
import jax.numpy as jnp
from .transformer import TransformerEncorder

class VIT(nnx.Module):
    def __init__(self,num_classes:int,in_channels:int,img_size:int,patch_size:int,num_layers:int,num_heads:int,mlp_dim:int,hidden_size:int,dropout_rate:float,*,rngs:nnx.Rngs):        
        self.patches = (img_size//patch_size)**2
        self.initializer = nn.initializers.truncated_normal(stddev=0.02)
        self.patch_embeddings = nnx.Conv(in_channels,hidden_size,kernel_size=(patch_size,patch_size),strides=(patch_size,patch_size),padding="VALID",use_bias=True,rngs=rngs)
        self.position_embeddings = nnx.Param(self.initializer(rngs.params(),(1,self.patches+1,hidden_size),jnp.float32))
        self.dropout = nnx.Dropout(dropout_rate,rngs=rngs)
        self.class_token = nnx.Param(jnp.zeros((1, 1, hidden_size)))
        self.encoder_layer = nnx.Sequential(*[
            TransformerEncorder(hidden_size=hidden_size,mlp_dim=mlp_dim,num_heads=num_heads,dropout_rate=dropout_rate,rngs=rngs)
            for i in range(num_layers)
        ])
        self.final_norm = nnx.LayerNorm(hidden_size,rngs=rngs)
        self.classifier = nnx.Linear(hidden_size,num_classes,rngs=rngs)

    def __call__(self,x,training=True):
        patches = self.patch_embeddings(x)

        batch_size = patches.shape[0]

        patches = patches.reshape(batch_size,-1,patches.shape[-1])
        cls_token = jnp.tile(self.class_token,[batch_size,1,1])
        x = jnp.concat([cls_token,patches],axis=1)

        embeddings = x + self.position_embeddings
        if training == True: 
            embeddings = self.dropout(embeddings, deterministic=not training)
        else:
            embeddings = self.dropout(embeddings)

        x = self.encoder_layer(embeddings)
        x = self.final_norm(x)

        x = x[:,0]

        x = self.classifier(x)
        return x