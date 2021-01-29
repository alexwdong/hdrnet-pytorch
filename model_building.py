import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import pytorch_lightning as pl
#---------------------------------------------------------------------------------------------
# Start: Define Generic Layers
#---------------------------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, inc , outc, kernel_size, padding=0, stride=1, use_bias=True, activation=nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

class FcBlock(nn.Module):
    def __init__(self, inc , outc, activation=nn.ReLU, batch_norm=False):
        super(FcBlock, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.fc(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

    
    
#---------------------------------------------------------------------------------------------
# Start: Low Resolution Path
#---------------------------------------------------------------------------------------------

class LowLevelFeatures(nn.Module):
    '''
    Current expects input to be of size 256x256
    '''
    
    def __init__(self,):
        super(LowLevelFeatures, self).__init__()
        self.conv1 = ConvBlock(inc=3, outc=8, kernel_size=3, padding=1, stride=2, batch_norm=True)
        self.conv2 = ConvBlock(inc=8, outc=16, kernel_size=3, padding=1, stride=2, batch_norm=True)
        self.conv3 = ConvBlock(inc=16, outc=32, kernel_size=3, padding=1, stride=2, batch_norm=True)
        self.conv4 = ConvBlock(inc=32, outc=64, kernel_size=3, padding=1, stride=2, batch_norm=True)
        
        
        
    def forward(self,x):
        if x.shape[-1] != 256 or x.shape[-2]!= 256:
            raise ValueError('input image here needs to be 3x256x256 (channel x height x width )')
        x = self.conv1(x) # 256pix -> 128pix
        x = self.conv2(x) # 128pix -> 64pix
        x = self.conv3(x) # 64pix -> 32pix
        x = self.conv4(x) # 32pix -> 16pix
       
        return x

class LocalFeatures(nn.Module):
    def __init__(self,):
        super(LocalFeatures, self).__init__()
        self.conv1 = ConvBlock(inc=64, outc=64, kernel_size=3, padding=1, stride=1, batch_norm=True)
        self.conv2 = ConvBlock(inc=64, outc=64, kernel_size=3, padding=1, stride=1, batch_norm=True)

    def forward(self,x):
        x = self.conv1(x) 
        x = self.conv2(x) 
        return x
class GlobalFeatures(nn.Module):
    def __init__(self,):
        super(GlobalFeatures, self).__init__()
        self.conv1 = ConvBlock(inc=64, outc=64, kernel_size=3, padding=1, stride=2, batch_norm=True)
        self.conv2 = ConvBlock(inc=64, outc=64, kernel_size=3, padding=1, stride=2, batch_norm=True)
        self.view = View((-1,64*4*4))
        self.fc1 = FcBlock(inc=64*4*4,outc=256)
        self.fc2 = FcBlock(inc=256,outc=128)
        self.fc3 = FcBlock(inc=128,outc=64)
        
    def forward(self,x):
        x = self.conv1(x) # 16 -> 8
        x = self.conv2(x) #8 -> 4 
        x = self.view(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
class FusionLayer(nn.Module):
    def __init__(self,):
        super(FusionLayer, self).__init__()
    
    def forward(self,local_input,global_input):
        # Local Input is size bsize x channel x height x width
        # Global Input is size bsize x n_feat
        # we need to first shape global input to be bsize x n_feat x 1 x 1
        # Then copy it height*width times to get bsize x n_feat x height x width
        # Then concatentate it to local input to get bsize x (channel+n_feat) x height x width
        
        local_height = local_input.shape[2]
        local_width = local_input.shape[3]
        
        
        global_input = torch.unsqueeze(global_input,2)
        global_input = torch.unsqueeze(global_input,3)
        
        global_input = global_input.expand(-1,-1,local_height,local_width)
        fused_output = torch.cat((local_input,global_input),1) 
        return fused_output
    
class PointwiseChannelMixingLayer(nn.Module):
    def __init__(self,):
        super(PointwiseChannelMixingLayer, self).__init__()
        self.conv = ConvBlock(inc=96, outc=96, kernel_size=1, padding=0, stride=1,)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
#---------------------------------------------------------------------------------------------
# Start: Define High Resolution Path
#---------------------------------------------------------------------------------------------

class GuidanceLayer(nn.Module):
    def __init__(self,):
        super(GuidanceLayer, self).__init__()
        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=True)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Sigmoid) #nn.Tanh

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SlicingLayer(nn.Module):
    def __init__(self,):
        super(SlicingLayer, self).__init__()
        
    def forward(self, bilateral_grid, guidance_map):
        '''
        Input Structure:
        (blg stands for bilateral grid)
        bilateral_grid should be 5 dimensional: (bsize,12,blg_depth, blg_height, blg_width)
        
        bilateral_grid[n,:,d,h,w] should index the 12 coefficients for the 3x4 affine color transformation.
        
        guidance map should be 4 dimensional: (bsize,guided_val,gmap_y,gmap_x)
        gmap_y and gmap_x ranges should correspond to the pixel indicies (e.g, 0 to 1920-1, 0 to 1280-1) over the full res image
        
        '''
        # Nx12x8x16x16
        device = bilateral_grid.get_device()
        bsize, _, gmap_height, gmap_width = guidance_map.shape
        
        # We need to create the "grid" input to grid_sample. (PLEASE check documentation of grid_sample, it's confusing)
        # 
        gmap_y, gmap_x = torch.meshgrid([torch.arange(0, gmap_height), torch.arange(0, gmap_width)]) # gmap_x and gmap_y range from 0 to size of full resolution image
        #Send to device
        if device >= 0:
            gmap_y = gmap_y.to(device)
            gmap_x = gmap_x.to(device)
       
        gmap_y = gmap_y.float() #Convert to float
        gmap_x = gmap_x.float()
        
        gmap_y = (gmap_y / (gmap_height-1) ) * 2 - 1 # Normalize to be in range [-1,1], this is needed for the input to grid_sample.
        gmap_x = (gmap_x / (gmap_width-1) ) * 2 - 1 
        #^^^ At this point,  gmap_x and gmap_y should be of shape (gmap_height, gmap_width)
        gmap_y = gmap_y.repeat(bsize, 1, 1) # Should be of shape (b_size, gmap_height,gmap_width)
        gmap_x = gmap_x.repeat(bsize, 1, 1) # Should be of shape (b_size, gmap_height,gmap_width)
        gmap_y = gmap_y.unsqueeze(3) # Should be of shape (b_size, gmap_height,gmap_width,1)
        gmap_x = gmap_x.unsqueeze(3) # Should be of shape (b_size, gmap_height,gmap_width,1)
        gmap_z = guidance_map.permute(0,2,3,1).contiguous() # Go from (bsize,guided_val,gmap_height,gmap_width) to (bsize,gmap_height,gmap_width,guided_val)
        guidemap_guide = torch.cat([gmap_x, gmap_y, gmap_z ], dim=3).unsqueeze(1) # Make sure to concatenate in x,y,guided_val order.
        #print(bilateral_grid.shape,guidemap_guide.shape)
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, 'bilinear', align_corners=True)
        #print('coeff size:',coeff.shape)
        return coeff.squeeze(2)

    
class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, sliced_coeff, full_res_input):

        '''
        full_res_input
        
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''
        
        R_out = torch.sum(full_res_input * sliced_coeff[:, 0:3, :, :], dim=1, keepdim=True) + sliced_coeff[:, 3:4, :, :]
        G_out = torch.sum(full_res_input * sliced_coeff[:, 4:7, :, :], dim=1, keepdim=True) + sliced_coeff[:, 7:8, :, :]
        B_out = torch.sum(full_res_input * sliced_coeff[:, 8:11, :, :], dim=1, keepdim=True) + sliced_coeff[:, 11:12, :, :]

        return torch.cat([R_out, G_out, B_out], dim=1)
    
        
#---------------------------------------------------------------------------------------------
# Start: Define HDRNet
#---------------------------------------------------------------------------------------------


class HDRPointwiseNN(pl.LightningModule):

    def __init__(self):
        super(HDRPointwiseNN, self).__init__()
        self.low_level = LowLevelFeatures()
        self.local_features = LocalFeatures()
        self.global_features = GlobalFeatures()
        self.fusion_layer = FusionLayer()
        self.pwc_mixing = PointwiseChannelMixingLayer()
        
        
        self.reshape = View((-1,12,8,32,32))
        self.guidance_layer = GuidanceLayer()
        self.slicing_layer = SlicingLayer()
        self.apply_coeffs = ApplyCoeffs()

    def forward(self, lowres, fullres):
        low_level_output = self.low_level(lowres)
        local_output = self.local_features(low_level_output)
        global_output = self.global_features(low_level_output)
        fusion_output = self.fusion_layer(local_output,global_output)
        pwc_mix_output = self.pwc_mixing(fusion_output)
        bilateral_grid_output = self.reshape(pwc_mix_output)
        
        guidance_output = self.guidance_layer(fullres)
        
        slice_coeffs = self.slicing_layer(bilateral_grid_output, guidance_output)
        out = self.apply_coeffs(slice_coeffs, fullres)
        return out
    
    def training_step(self, batch, batch_idx):
        
        input_reduced, input_full, target_full = batch
        ### Start HDRnet Model
        low_level_output = self.low_level(input_reduced)
        local_output = self.local_features(low_level_output)
        global_output = self.global_features(low_level_output)
        fusion_output = self.fusion_layer(local_output,global_output)
        pwc_mix_output = self.pwc_mixing(fusion_output)
        bilateral_grid_output = self.reshape(pwc_mix_output)
        
        guidance_output = self.guidance_layer(input_full)
        
        slice_coeffs = self.slicing_layer(bilateral_grid_output, guidance_output)
        pred = self.apply_coeffs(slice_coeffs, input_full)
        ### End HDRnet Model
        loss = F.mse_loss(pred, target_full)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
  
    def validation_step(self, batch, batch_idx):
        input_reduced, input_full, target_full = batch
        ### Start HDRnet Model
        low_level_output = self.low_level(input_reduced)
        local_output = self.local_features(low_level_output)
        global_output = self.global_features(low_level_output)
        fusion_output = self.fusion_layer(local_output,global_output)
        pwc_mix_output = self.pwc_mixing(fusion_output)
        bilateral_grid_output = self.reshape(pwc_mix_output)

        guidance_output = self.guidance_layer(input_full)

        slice_coeffs = self.slicing_layer(bilateral_grid_output, guidance_output)
        pred = self.apply_coeffs(slice_coeffs, input_full)
        ### End HDRnet Model
        loss = F.mse_loss(pred, target_full)
        self.log('val_loss', loss)
        return loss
