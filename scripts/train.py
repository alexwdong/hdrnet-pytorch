from model_building import HDRPointwiseNN
from dataset import HDRDataset
import os
from PIL import Image
import math

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader,random_split
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


orig_photos_path = 'E:\Projects\StyleTransfer\justinho_dataset_processed\small_original_reduced_size'
target_photos_path = 'E:\Projects\StyleTransfer\justinho_dataset_processed\small_edited_reduced_size'



dataset = HDRDataset(orig_photos_path,target_photos_path)
dataset_length = len(dataset)
train_length = math.ceil(.8 *dataset_length)
val_length = dataset_length-train_length
train_set, val_set = random_split(dataset, [train_length, val_length], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size = 4)
val_loader =  DataLoader(val_set, batch_size = 4)



print(dataset_length)
print(train_length)
print(val_length)

hdrnet = HDRPointwiseNN()

trainer = pl.Trainer(gpus=1,
                     max_epochs=500,
                     val_check_interval=.5,
                     terminate_on_nan=True)
trainer.fit(hdrnet, train_dataloader = train_loader, val_dataloaders = val_loader)







