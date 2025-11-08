from .cnn import CNN
from .vision_transformer import VIT

model_registry = {
    'CNN':CNN,
    'VIT':VIT
}


def choose_model(model_dict,rngs):
    model = model_registry.get(model_dict['model'],f'{model_dict['model']}不存在')
    if model == CNN:
        return CNN(in_channels=model_dict['in_channels'],
                   base_channels=model_dict['base_channels'],
                   kernel_size=tuple(model_dict['kernel_size']),
                   num_classes=model_dict['num_classes'],
                   img_size=tuple(model_dict['img_size']),
                   rngs=rngs
                   )
    elif model==VIT:
        return VIT(num_classes=model_dict['num_classes'],
                   in_channels=model_dict['in_channels'],
                   img_size=model_dict['img_size'],
                   patch_size=model_dict['patch_size'],
                   num_layers=model_dict['num_layers'],
                   num_heads=model_dict['num_heads'],
                   mlp_dim=model_dict['mlp_dim'],
                   hidden_size=model_dict['hidden_size'],
                   dropout_rate=model_dict['dropout_rate'],
                   rngs=rngs)