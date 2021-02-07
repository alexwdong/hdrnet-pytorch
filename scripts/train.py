from model_building import HDRPointwiseNN
from pretrained_hdrnet import HDRnetPretrained
from dataset import HDRDataset
import os
from PIL import Image
import math
import argparse
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader,random_split,ConcatDataset
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

parser = argparse.ArgumentParser(description='take input path full of images, makes the outputpath with images at reduced size')

parser.add_argument('-l', '--logging_path', help='logging_path',default='./')
parser.add_argument('-g', '--gpus', help='number of gpus',default = 1)
parser.add_argument('-d', '--dataset_flag', help='1 = hdrnet, 2 = JustinHo',default=1)
parser.add_argument('-m', '--mode',help='1=HDRnetPointwiseNN, 2=pretrained_Resnet')
parser.add_argument('-c', '--checkpoint_path', help='checkpoint file,if this is present, the model will load the checkpoint before training',required=False)
parser.add_argument('-b', '--batch_size',help='batch_size, default=16', default=16)
args = vars(parser.parse_args())

logging_path = args['logging_path']
dataset_flag = int(args['dataset_flag'])
mode = int(args['mode'])
num_gpus = int(args['gpus'])
checkpoint_path = args['checkpoint_path']
batch_size=int(args['batch_size'])

###Define skip_list here
skip_list=[]

if __name__ == '__main__':

    if dataset_flag== 1:
        orig_photos_path = '/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_original'
        editor_A_path = '/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_A'
        #editor_B_path = '/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_B'
        #editor_C_path = '/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_C'
        #editor_D_path = '/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_D'
        #editor_E_path = '/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_E'

        dataset_A = HDRDataset(orig_photos_path,editor_A_path,skip_list=[])
        #dataset_B = HDRDataset(orig_photos_path,editor_B_path)
        #dataset_C = HDRDataset(orig_photos_path,editor_C_path)
        #dataset_D = HDRDataset(orig_photos_path,editor_D_path)
        #dataset_E = HDRDataset(orig_photos_path,editor_E_path)

        
        #dataset = ConcatDataset([dataset_A,dataset_B,dataset_C,dataset_D,dataset_E])
        dataset=dataset_A
    elif dataset_flag == 2:
        orig_photos_path = '/scratch/awd275/StyleTransfer/data/justinho/small_original/'
        edited_photos_path = '/scratch/awd275/StyleTransfer/data/justinho/small_edited'

        dataset = HDRDataset(orig_photos_path,edited_photos_path)

    else: 
        raise ValueError('mode should be 1 or 2')


    dataset_length = len(dataset)
    train_length = math.ceil(.9 * dataset_length)
    val_length = dataset_length - train_length
    train_set, val_set = random_split(dataset, [train_length, val_length], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size = batch_size, num_workers=24)
    val_loader =  DataLoader(val_set, batch_size = batch_size, num_workers=24)

    print(dataset_length)
    print(train_length)
    print(val_length)
    if mode==1:
        hdrnet = HDRPointwiseNN()
    else:
        hdrnet=HDRnetPretrained()
    if checkpoint_path: 
        hdrnet.load_from_checkpoint(checkpoint_path=checkpoint_path)

    trainer = pl.Trainer(gpus=num_gpus,
                         max_epochs=30,
                         val_check_interval=.5,
                         terminate_on_nan=True,
                         default_root_dir=logging_path)
    trainer.fit(hdrnet, train_dataloader = train_loader, val_dataloaders = val_loader)







