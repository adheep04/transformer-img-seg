'''
This is a fully convolutional neural network implementation using pytorch with a few minor architectural 
modifications from the original paper


'''
from torch import nn
import torch
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
import torchvision.transforms as T
from torch.nn import functional as F 
import math
from pathlib import Path
from PIL import Image


class SwinSeg(nn.Module):
    def __init__(self, n_class):        
        super().__init__()

        # pretrained base net extractor swin-v2-t 
        base_model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        self.base = base_model.features
        self.permute = base_model.permute
        self.norm = base_model.norm
        
        # returns prediction scores for intermediate layers and final
        self.fm96_proj = nn.Conv2d(96, 256, 1)
        self.fm192_proj = nn.Conv2d(192, 256, 1)
        self.fm384_proj = nn.Conv2d(384, 256, 1)
        self.fm768_proj = nn.Conv2d(768, 256, 1)
        self.smooth4 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.smooth8 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.smooth16 = nn.Conv2d(256, 256, 3, 1, padding=1)
        
        # 256 -> class size
        self.score = nn.Conv2d(256, n_class, 1)
        
        # initialize weights with kaiming
        nn.init.kaiming_uniform_(self.fm96_proj.weight,)
        nn.init.kaiming_uniform_(self.fm192_proj.weight,)
        nn.init.kaiming_uniform_(self.fm384_proj.weight,)
        nn.init.kaiming_uniform_(self.fm768_proj.weight,)
        nn.init.kaiming_uniform_(self.smooth4.weight,)
        nn.init.kaiming_uniform_(self.smooth8.weight,)
        nn.init.kaiming_uniform_(self.smooth16.weight,)
        nn.init.kaiming_uniform_(self.score.weight,)
        
        
        
    def forward(self, x):
        '''
        args:
        - x: tensor(batch_size, channel_size, height, width)
        
        output:
        - tensor(batch_size, class_size, height, width)
        
        '''
        # get img spatial dimensions
        img_res = x.shape[-2:]
        
        ''' forward pass through swin backbone'''
        
        x = self.base[0](x)
        fm96 = x = self.base[1](x)        
        x = self.base[2](x)         
        fm192 = x = self.base[3](x)
        x = self.base[4](x)         
        fm384 = x = self.base[5](x)
        x = self.base[6](x)
        x = self.base[7](x)         
        
        ''' norm and rearrange tensor dims'''
        
        fm96 = self.permute(fm96)
        fm192 = self.permute(fm192)
        fm384 = self.permute(fm384)
        fm768 = self.permute(self.norm(x))
        
        ''' feature pyramid network '''
        
        # project to 256 dimensions
        fm96 = self.fm96_proj(fm96)
        fm192 = self.fm192_proj(fm192)
        fm384 = self.fm384_proj(fm384)
        fm768 = self.fm768_proj(fm768)
                
        fuse16 = self.smooth16(fm384 + F.interpolate(fm768, size=fm384.shape[-2:], mode='bilinear', align_corners=False))
        fuse8 = self.smooth8(fm192 + F.interpolate(fuse16, size=fm192.shape[-2:], mode='bilinear', align_corners=False))
        fuse4 = self.smooth4(fm96 + F.interpolate(fuse8, size=fm96.shape[-2:], mode='bilinear', align_corners=False))
        
        score = self.score(fuse4)
        
        return F.interpolate(score, size=img_res, mode='bilinear', align_corners=False)



transform = T.Compose([
    # transforms.Resize((224, 224)),  
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])  
])

def resize(img, max_pixels=1000*1000):
    # max is 1024*1024 pixels
    
    # get original image dimensions
    w, h = img.size
    # calculate scale ratio between original image necessary to hit max pixel size
    ratio = math.sqrt(max_pixels / (w * h))
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    # resize height and width by proportional values to maintain aspect ratio
    return T.Resize((new_h, new_w))(img)


img_path = Path(r'data\donker_bergen_sneeuw_5000x5000-wallpaper-5120x2880.jpg')
img_path.exists()
img = Image.open(img_path)
img_r = resize(img)
img_tensor = transform(img_r).unsqueeze(0)

swinseg = SwinSeg(30)
swingseg = swinseg.to(torch.device('cuda'))
img_tensor = img_tensor.to(torch.device('cuda'))
print(img_tensor.shape)
pred = swinseg(img_tensor)
print(pred.shape)