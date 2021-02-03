import sys
sys.path.append('..')
from model_building import HDRPointwiseNN
from dataset import HDRDataset
import os
from PIL import Image
import math
import torch
from torchvision import transforms

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_one_image(image_path,model_path):

    hdrnet = HDRPointwiseNN()
    with open(model_path,'rb') as f:
        checkpoint = torch.load(f, map_location=lambda storage, loc: storage)
    hdrnet.load_state_dict(checkpoint['state_dict'])
    hdrnet.eval()
    reduced_transforms = transforms.Compose([
            transforms.Resize((256,256), Image.BICUBIC),
            transforms.ToTensor(),
        ])
    
    full_transforms = transforms.ToTensor()
    
    input_image = Image.open(image_path).convert('RGB')
    input_image_reduced = reduced_transforms(input_image).unsqueeze(0)
    input_image_full = full_transforms(input_image).unsqueeze(0)
    pred = hdrnet.forward(input_image_reduced,input_image_full,)
    pred = pred.detach()
    
    f, axs = plt.subplots(pred.shape[0],2,figsize=(30,30))
    for ii in range(pred.shape[0]):

        plt.subplot(pred.shape[0],3,3*ii+1)
        plt.imshow(input_image_full[ii].permute(1, 2, 0))

        plt.subplot(pred.shape[0],3,3*ii+2)
        plt.imshow(pred[ii].detach().permute(1, 2, 0))

def plot_batch_of_images(dataloader,model_path):

    hdrnet = HDRPointwiseNN()
    with open(model_path,'rb') as f:
        checkpoint = torch.load(f, map_location=lambda storage, loc: storage)
    hdrnet.load_state_dict(checkpoint['state_dict'])
    hdrnet.eval()
    reduced_transforms = transforms.Compose([
            transforms.Resize((256,256), Image.BICUBIC),
            transforms.ToTensor(),
        ])
    
    full_transforms = transforms.ToTensor()
    
    input_images = next(iter(dataloader))
    input_image_reduced = reduced_transforms(input_image)
    input_image_full = full_transforms(input_image)
    pred = hdrnet.forward(input_image_reduced,input_image_full,)
    pred = pred.detach()
    
    f, axs = plt.subplots(pred.shape[0],2,figsize=(30,30))
    for ii in range(pred.shape[0]):

        plt.subplot(pred.shape[0],3,3*ii+1)
        plt.imshow(input_image_full[ii].permute(1, 2, 0))

        plt.subplot(pred.shape[0],3,3*ii+2)
        plt.imshow(pred[ii].detach().permute(1, 2, 0))
