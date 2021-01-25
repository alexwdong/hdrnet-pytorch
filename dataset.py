import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import torch
from torchvision import transforms
from torch.utils.data import Dataset



skiplist = ['a1233-DSC_0064.jpg',
            'a1234-_DGW6333.jpg'
           ]
class HDRDataset(Dataset):
    def __init__(self, input_path,target_path, full_size=2048,reduced_size=256):
        self.input_path = input_path
        self.target_path = target_path
        # Make list of in_files and out_files
        self.file_names = [ f for f in listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
        self.input_files=[]
        self.target_files=[]
        for f in self.file_names:
            if f not in skiplist:
                self.input_files.append(os.path.join(self.input_path,f))
                self.target_files.append(os.path.join(self.target_path,f))
        
        
        self.full_size = full_size
        self.reduced_size = reduced_size
        
        self.reduced_transforms = transforms.Compose([
            transforms.Resize((self.reduced_size,self.reduced_size), Image.BICUBIC),
            transforms.ToTensor()
        ])
        
        self.full_transforms = transforms.Compose([
            transforms.Resize((self.full_size,self.full_size), Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_image = Image.open(self.input_files[idx]).convert('RGB')
        output_image = Image.open(self.target_files[idx]).convert('RGB')
        
        input_image_reduced = self.reduced_transforms(input_image)
        input_image_full= self.full_transforms(input_image)
        output_image_full = self.full_transforms(output_image)

        return input_image_reduced, input_image_full, output_image_full
