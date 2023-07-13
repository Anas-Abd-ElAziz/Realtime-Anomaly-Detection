import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import skvideo.io

def fix_shape(ten_cropped):
    from torchvision.transforms.functional import Tensor
    ten_cropped = Tensor(ten_cropped)
    ten_cropped = ten_cropped.permute(1,0,2,3,4)
    ten_cropped = np.array(ten_cropped)
    return ten_cropped

videodata = skvideo.io.vread("project.mp4")  
print(videodata.shape)
Lambda= lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.TenCrop((224,224)),
])
ten_crop = [Lambda((transform(image))) for image in videodata]
ten_crop = np.array([np.array(i) for i in ten_crop])

ten_crop = fix_shape(ten_cropped=ten_crop)
print(ten_crop.shape)

np.save("ten_crop.npy", ten_crop)