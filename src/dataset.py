import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import torch
from torchvision import transforms
from torch.utils.data import Dataset



skip_list = [ #fivek skip list
    'a1233-DSC_0064.jpg',
    'a1234-_DGW6333.jpg',
    #jho skip list
    'JH2_5161.jpg',
    'JH2_5182.jpg'
           ]

def make_skip_list(input_path,target_path):
    file_names = [ f for f in listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    for f in file_names:
        try:
            #Try opening the files, if they work, then we append to the input_files and target_files lists
            input_image_path = os.path.join(self.input_path, f)
            output_image_path = os.path.join(self.target_path, f)
        except Exception: #If they don't work, append to skip_list, and print skip_list
            skip_list.append(f)
    print("Skip List Below:")
    print(skip_list)
    return skip_list
    
class HDRDataset(Dataset):
    def __init__(self, input_path,target_path, full_size=2048,reduced_size=256,create_skip_list=False,skip_list=[]):
        self.input_path = input_path
        self.target_path = target_path
        # Make list of in_files and out_files
        self.file_names = [ f for f in listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
        self.input_files=[]
        self.target_files=[]
        
        if create_skip_list is False:
            pass
        else:
            skip_list = make_skip_list(input_path,target_path)
            
        for f in self.file_names:
            if f in skip_list:
                pass
            else:
                self.input_files.append(os.path.join(self.input_path,f))
                self.target_files.append(os.path.join(self.target_path,f))

        print("skip_list is of length:", str(len(skip_list)))
        print(skip_list)
        
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
