from model_building import HDRPointwiseNN
from dataset import HDRDataset
import os
from PIL import Image
import math

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader,random_split,ConcatDataset
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

logging_path = '/home/awd275/hdrnet-pytorch/fivek_logs/'

orig_photos_path = '/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_original'
editor_A_path = '/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_A'
editor_B_path = '/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_B'
editor_C_path = '/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_C'
editor_D_path = '/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_D'
editor_E_path = '/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_E'

dataset_A = HDRDataset(orig_photos_path,editor_A_path)
dataset_B = HDRDataset(orig_photos_path,editor_B_path)
dataset_C = HDRDataset(orig_photos_path,editor_C_path)
dataset_D = HDRDataset(orig_photos_path,editor_D_path)
dataset_E = HDRDataset(orig_photos_path,editor_E_path)


dataset = ConcatDataset([dataset_A,dataset_B,dataset_C,dataset_D,dataset_E])


dataset_length = len(dataset)
train_length = math.ceil(.8 *dataset_length)
val_length = dataset_length-train_length
train_set, val_set = random_split(dataset, [train_length, val_length], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size = 4, num_workers=24)
val_loader =  DataLoader(val_set, batch_size = 4, num_workers=24)

print(dataset_length)
print(train_length)
print(val_length)

hdrnet = HDRPointwiseNN()

trainer = pl.Trainer(gpus=1,
                     max_epochs=50,
                     val_check_interval=.5,
                     terminate_on_nan=True,
                     default_root_dir=logging_path)
trainer.fit(hdrnet, train_dataloader = train_loader, val_dataloaders = val_loader)







