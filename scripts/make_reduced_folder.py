import os
from PIL import Image
from os import listdir,mkdir
from os.path import isfile, join
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import argparse

parser = argparse.ArgumentParser(description='take input path full of images, makes the outputpath with images at reduced size')
parser.add_argument('-i','--input_path', help='Input path containing images', required=True)
parser.add_argument('-o','--output_path', help='output path that will contain the reduced size images', required=True)
parser.add_argument('-s','--reduced_size', help='size to reduce to (will make a square (size, size) image)',default=2048)
args = vars(parser.parse_args())
input_path = args['input_path']
output_path = args['output_path']
reduced_pixel_size = args['reduced_size']
if __name__ == '__main__':

    # Get all the input file names, and make output file names for them (will be same name, but in output directory)
    file_names = [ f for f in listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    input_files = [os.path.join(input_path,f) for f in file_names]
    output_files = [os.path.join(output_path,f) for f in file_names]
    
    # Reduce size to (reduced_pix_size, reduced_pix_size)    
    reduce_transform = transforms.Compose([
        transforms.Resize((reduced_pixel_size,reduced_pixel_size), Image.BILINEAR),
    ])
    #Make output dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Reduce each image
    for ii in range(len(input_files)):
        #Get names
        input_file = input_files[ii]
        output_file = output_files[ii]
        # Open, reduce, save
        input_image_full = Image.open(input_file)
        try:
            input_image_reduced = reduce_transform(input_image_full)
            input_image_reduced.save(output_file, "JPEG")
            print('saved to output file: ' + output_file) 
        except:
            print('file: ' + input_file + ' could not be resized (one dimension was too small most likely)')
    