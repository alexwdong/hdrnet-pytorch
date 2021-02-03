import sys
from model_building import HDRPointwiseNN
from dataset import HDRDataset
import os
from PIL import Image
import math
import torch
from torchvision import transforms
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='take input path full of images, makes the outputpath with images at reduced size')

parser.add_argument('-m','--mode',help='mode. 1 = run on ONE image, 2= run on FOLDER of images',required=True)
parser.add_argument('-i', '--image_path', help='path to image file or folder',)
parser.add_argument('-c', '--checkpoint_path', help='path to ckpt file',)
parser.add_argument('-o','--out_size',help='size to resize the output files to',)
args = vars(parser.parse_args())

mode = int(args['mode'])
image_path = args['image_path']
checkpoint_path = args['checkpoint_path']
out_size=args['out_size']
if out_size is not None:
    out_size = int(out_size)

if __name__ == '__main__':
    
    #Figure out which mode
    if mode ==1:
        output_path = 'out_image.jpeg'
        orig_output_path = 'orig_out_image.jpeg'
    else: 
        raise NotImplementedError('mode 2 not implemented yet')
    
    
    #Initialize model with checkpoint
    hdrnet = HDRPointwiseNN()
    with open(checkpoint_path,'rb') as f:
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
    print('pred shape:', pred.shape)
    if out_size is not None:    
        pred = F.interpolate(pred,size=out_size)
        input_image_resized = F.interpolate(input_image_full,size=out_size)
    else:
        pass
    
    
    if pred.shape[0]==1:
        print('blah', pred[0,:,:,:].shape)
        im = transforms.ToPILImage()(pred[0,:,:,:])
        im.save(output_path, 'JPEG')
        im_orig = transforms.ToPILImage()(pred[0,:,:,:])
        im_orig.save(orig_output_path,'JPEG')
        
        
    else:
        raise NotImplementedError('you shouldnt have gottent his error')
        
        
    
    