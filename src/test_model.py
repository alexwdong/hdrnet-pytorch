import pretrained_hdrnet
import torch

low_x = torch.zeros(5,3,256,256)
high_x = torch.zeros(5,3,2048,2048)

model = pretrained_hdrnet.HDRnetPretrained()
model(low_x,high_x)


