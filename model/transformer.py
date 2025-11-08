from flax import nnx


class TransformerEncorder(nnx.Module):
    def __init__(self,hidden_size,mlp_dim,num_heads,dropout_rate,*,rngs:nnx.Rngs):
        self.norm1 = nnx.LayerNorm(hidden_size,rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads = num_heads,
            in_features = hidden_size,
            dropout_rate = dropout_rate,
            broadcast_dropout = False,
            decode = False,
            deterministic = False,
            rngs = rngs  
        )
        self.norm2 = nnx.LayerNorm(hidden_size,rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size,mlp_dim,rngs=rngs),
            nnx.gelu,
            nnx.Dropout(dropout_rate,rngs=rngs),
            nnx.Linear(mlp_dim,hidden_size,rngs=rngs),
            nnx.Dropout(dropout_rate,rngs=rngs)
        )
    def __call__(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x