"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import os
import pandas as pd
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

plt.rcParams["savefig.bbox"] = 'tight'

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:

        self.transform = transform
        self.paths = []
        for path in glob.iglob(os.path.join(path_to_folder, "**", "*.jpg")):
            self.paths.append(path)
            
    def __len__(self):
        print(f'images_count: {len(self.paths)}')
        return len(self.paths) # TODO: fill out
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        img = Image.open(self.paths[index])
        # print(torch.from_numpy(np.array(self.transform(img))).shape)
        return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='.\data\lfw', type=str)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-num_workers', default=1, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = np.asarray(img).transpose(1,2,0)
            axs[0, i].imshow(img)
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        fig.savefig('figures\image_grid.png')

    # for batch_idx, batch in enumerate(dataloader):
    #     print(batch)

    if args.visualize_batch:
        # TODO: visualize a batch of images
        grid = make_grid(next(iter(dataloader)))
        show(grid)
        
        
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print('Timing: {np.mean(res)}+-{np.std(res)}')
