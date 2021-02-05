import torch
import torchvision
import inspect
from torch import nn
from model_building import LocalFeatures,FusionLayer,PointwiseChannelMixingLayer,GuidanceLayer,SlicingLayer,ApplyCoeffs
class ResnetPartial(nn.Module):
    def __init__(self, ):
        super(ResnetPartial, self).__init__()
        self.partial_resnet = make_partial_resnet()
    def forward(self, x):
        out = self.partial_resnet(x)
        return out
        
def make_partial_resnet():
    full_resnet = torchvision.models.resnet50(pretrained=True, progress=True,)
    resnet_modules = []
    #Input is 229x229, output is b_size x 1024channels x 15 x 15
    for ii,layer in enumerate(full_resnet.children()):
        print(ii)
        resnet_modules.append(layer)
        if ii == 7:# 
            break
    partial_resnet = nn.Sequential(*resnet_modules)
    return partial_resnet
    
    
class GlobalFeatures(nn.Module):
    def __init__(self,):
        super(GlobalFeatures, self).__init__()
        self.conv1 = ConvBlock(inc=1024, outc=512, kernel_size=3, padding=1, stride=2, batch_norm=True)
        self.conv2 = ConvBlock(inc=512, outc=256, kernel_size=3, padding=1, stride=2, batch_norm=True)
        self.view = View((-1,256*5*5))
        self.fc1 = FcBlock(inc=256*5*5,outc=1024)
        self.fc2 = FcBlock(inc=1024,outc=256)
        self.fc3 = FcBlock(inc=256,outc=64)
        
    def forward(self,x):
        x = self.conv1(x) # 15 -> 8 (stride 2 with padding 1)
        x = self.conv2(x) #8 -> 5 (stride 2 with padding 1)
        x = self.view(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x  
    
class HDRnetPretrained(pl.LightningModule):

    def __init__(self,freeze_pretrained_weights=True):
        super(HDRPointwiseNN, self).__init__()
        self.low_level = make_partial_resnet()
        if freeze_pretrained_weights==True:
            self.low_level.requires_grad_(False)
        
        self.local_features = LocalFeatures()
        self.global_features = GlobalFeatures()
        self.fusion_layer = FusionLayer()
        self.pwc_mixing = PointwiseChannelMixingLayer()
        
        
        self.reshape = View((-1,12,8,16,16))
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
