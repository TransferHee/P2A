import torch

import os
import argparse
from tqdm.auto import tqdm

from p2a import P2A
from dataset import ImageNetDataset
from utils import seed_everything, calc_distance, save_images

os.makedirs('./result', exist_ok=True)

def args_parser():
    parser = argparse.ArgumentParser(description='Generate adversarial examples')
    parser.add_argument('--input_dir', type=str, default='./ImageNet1000', help='the path for your image dataset')
    
    parser.add_argument('--model', type=str, default='resnet101', help='Source model name')
    parser.add_argument('--attack', type=str, default='P2A', help='Attack method')
    
    parser.add_argument('--eps', type=float, default=16/255, help='Maximum perturbation')
    parser.add_argument('--alpha', type=float, default=1.6/255, help='Alpha')
    parser.add_argument('--decay', type=int, default=1, help='Decay')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--num_ens', type=int, default=30, help='Num of ensemble')
    
    parser.add_argument('--layer', type=str, default='layer1', help='Target feature layer')
    parser.add_argument('--tau', type=float, default=1e+1, help='Tau')
    parser.add_argument('--eta', type=float, default=0.3, help='Attenuation factor')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
    
    args = parser.parse_args()
    return args

def main():
    args = args_parser()
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    seed_everything(args.seed)
    
    print('==> Prepare the data')
    dataset = ImageNetDataset(data_dir=args.input_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    if args.attack == 'P2A':
        attacker = P2A(model_name=args.model,
                       epsilon=args.eps,
                       alpha=args.alpha,
                       epoch=args.epochs,
                       decay=args.decay,
                       num_ens=args.num_ens,
                       tau=args.tau,
                       eta=args.eta,
                       attack=args.attack,
                       feature_layer=args.layer)
    else:
        print(args.attack, 'is not supported')
        raise NotImplementedError
    
    os.makedirs(f'./result/{args.attack}', exist_ok=True)
    os.makedirs(f'./result/{args.attack}/{args.model}', exist_ok=True)
    
    l2_total = torch.Tensor(1000)
    linf_total = torch.Tensor(1000)
    
    print('==> Start generating adversarial examples')
    for batch_idx, (inputs, targets, filenames) in tqdm(enumerate(dataloader), total=1000 // args.batch_size):
    
        perturbation = attacker(inputs, targets)
        x_adv = inputs + perturbation.cpu()
        save_images(f'./result/{args.attack}/{args.model}', x_adv, filenames)
        
        start_idx = batch_idx * args.batch_size
        end_idx = start_idx + inputs.size(0)
        
        l2, linf = calc_distance(perturbation.detach().cpu())
        l2_total[start_idx:end_idx] = l2
        linf_total[start_idx:end_idx] = linf
    
    print(f'Result of {args.attack} attack')
    print(f'L2 norm: Mean {l2_total.mean().item():.5f}, Max {l2_total.max().item():.5f}, Min {l2_total.min().item():.5f}')
    print(f'Linf norm: Mean {linf_total.mean().item():.5f}, Max {linf_total.max().item():.5f}, Min {linf_total.min().item():.5f}')

if __name__ == '__main__':
    main()