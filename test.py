import torch

import os
import argparse
from tqdm.auto import tqdm

from dataset import ImageNetDataset_test
from utils import seed_everything, wrap_model, load_single_model, eval_attack

os.makedirs('./result', exist_ok=True)

def args_parser():
    parser = argparse.ArgumentParser(description='Test the generated adversarial examples')
    parser.add_argument('--input_dir', type=str, default='./ImageNet1000', help='the path for your image dataset')
    
    parser.add_argument('--model', type=str, default='resnet101', help='Source model name')
    parser.add_argument('--attack', type=str, default='P2A', help='Attack method')
    
    parser.add_argument('--layer', type=str, default='layer1', help='Target feature layer')
    parser.add_argument('--tau', type=float, default=1e+1, help='Temperature')
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('==> Prepare the data')
    dataset = ImageNetDataset_test(data_dir=args.input_dir, adv_dir=f'./result/{args.attack}/{args.model}')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    total = len(dataset)
    
    model_ls = ['inception_v3', 'inception_v4', 'inception_resnet_v2',
                'resnet101', 'resnet152',
                'vgg16', 'resnext50_32x4d',
                'vit_base_patch16_224', 'pit_b_224', 'swin_tiny_patch4_window7_224', 'deit_base_patch16_224']
    
    desc = f'Case of {args.attack} crafted on {args.model} | Layer: {args.layer} | | Eta: {args.eta} | Tau: {args.tau} |'
    print(desc)
    with open(f'{args.attack}_{args.model} results_eval.txt', 'a') as f:
        f.write('='*110 + '\n')
        f.write(desc + '\n')
        f.write('='*110 + '\n')
    print('==> Start evaluating')
    acc_total = []
    for model_name in model_ls:
        model = load_single_model(model_name)
        model = wrap_model(model.eval().to(device))
        
        for param in model.parameters():
            param.requires_grad = False
            
        total_correct_adv = 0
        for inputs, x_adv, targets in tqdm(dataloader, total=len(dataset) // args.batch_size):
            inputs = inputs.to(device)
            x_adv = x_adv.to(device)
            targets = targets.to(device)
            
            adv_correct = eval_attack(model, x_adv, targets)
            total_correct_adv += adv_correct

        print(f'Target model {model_name} ASR: {100 * (1 - total_correct_adv / total):.1f}%\n')
        result = f'To {model_name:20} | {100 * (1 - total_correct_adv / total):.1f}%'
        acc_total.append(100 * (1 - total_correct_adv / total))
        with open(f'{args.attack}_{args.model} results_eval.txt', 'a') as f:
            f.write(result + '\n')
        
        del model
        torch.cuda.empty_cache()
    
    print(f'Average ASR: {sum(acc_total) / len(acc_total):.1f}%')
    with open(f'{args.attack}_{args.model} results_eval.txt', 'a') as f:
        f.write(f'Average ASR: {sum(acc_total) / len(acc_total):.1f}%' + '\n')
    
if __name__ == '__main__':
    main()