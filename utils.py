import torch
import torchvision
from torchvision import transforms

import os
import timm
import random
import numpy as np
from PIL import Image
    
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def eval_attack(model, x_adv, targets):
    
    outputs = model(x_adv)
    _, predicted_adv = outputs.max(1)
    adv_correct = predicted_adv.eq(targets).sum().item()

    return adv_correct

def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def wrap_model(model):
    """
    Add normalization layer with mean and std in training configuration
    """
    if hasattr(model, 'default_cfg'):
        """timm.models"""
        normalize = transforms.Normalize(mean=model.default_cfg['mean'], std=model.default_cfg['std'])
    else:
        """torchvision.models"""
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return torch.nn.Sequential(normalize, model)

def calc_distance(perturb):
    '''
    perturb: torch.Tensor (Batch, 3, 224, 224)
    
    return: torch.Tensor (Batch)
    '''
    perturb = perturb.reshape(perturb.size(0), -1)
    l2 = torch.norm(perturb, p=2, dim=1)
    linf = torch.norm(perturb, p=float('inf'), dim=1)
    return l2, linf

def load_single_model(model_name):
    if model_name in torchvision.models.__dict__.keys():
        print('=> Loading model {} from torchvision.models'.format(model_name))
        model = torchvision.models.__dict__[model_name](weights="DEFAULT")
    elif model_name in timm.list_models():
        print('=> Loading model {} from timm.models'.format(model_name))
        model = timm.create_model(model_name, pretrained=True)
    else:
        raise ValueError('Model {} not supported'.format(model_name))
    
    return model