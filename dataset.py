import torch
from torch.utils.data import Dataset

import os
import numpy as np
import pandas as pd
from PIL import Image

class ImageNetDataset(Dataset):
    def __init__(self, data_dir='./ImageNet1000'):
        self.data_dir = os.path.join(data_dir, 'images')
        self.f2l = self.load_labels(os.path.join(data_dir, 'labels.csv'))

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]

        assert isinstance(filename, str)

        filepath = os.path.join(self.data_dir, filename)
        image = Image.open(filepath)
        image = image.resize((224, 224)).convert('RGB')
        image = np.array(image).astype(np.float32)/255
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.f2l[filename]-1

        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label']
                   for i in range(len(dev))}
        return f2l

class ImageNetDataset_test(Dataset):
    def __init__(self, data_dir='./ImageNet1000', adv_dir='./results/P2A/resnet101'):
        self.data_dir = os.path.join(data_dir, 'images')
        self.adv_dir = adv_dir
        self.f2l = self.load_labels(os.path.join(data_dir, 'labels.csv'))

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]

        assert isinstance(filename, str)

        filepath = os.path.join(self.data_dir, filename)
        image = Image.open(filepath)
        image = image.resize((224, 224)).convert('RGB')
        image = np.array(image).astype(np.float32)/255
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # load adv image
        filepath = os.path.join(self.adv_dir, filename)
        adv_image = Image.open(filepath)
        adv_image = adv_image.resize((224, 224)).convert('RGB')
        adv_image = np.array(adv_image).astype(np.float32)/255
        adv_image = torch.from_numpy(adv_image).permute(2, 0, 1)
        
        label = self.f2l[filename]-1

        return image, adv_image, label

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label']
                   for i in range(len(dev))}
        return f2l
