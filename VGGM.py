import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from PIL import Image

class VGGM(nn.Module):
    def __init__(self,n_channels):
        super(VGGM,self).__init__()

        self.ReLU = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self.Upsample = nn.Upsample(scale_factor=2)

        ## -------------Encoder--------------

        self.vgg = models.vgg16(pretrained=bool(1)).features
        
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.conv_layer1 = self.vgg[:4]     # conv1&2   64
        self.conv_layer2 = self.vgg[4:9]   # conv3&4    128
        self.conv_layer3 = self.vgg[9:16]   # conv5&6&7 256
        self.conv_layer4 = self.vgg[16:23]  # conv8&9&10    512
        self.conv_layer5 = self.vgg[23:30]  # conv11&12&13  512

        ## ------------Decoder----------------

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.outconv = nn.Conv2d(64, 1, kernel_size=1, padding=0)


    def forward(self,x):

        hx = x      # size=256
        ## -------------Encoder-------------
        #show_gray(torch.mean(hx, axis=1),0)
        out1 = self.conv_layer1(hx)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)

        ## -------------Decoder-------------
        hx = self.ReLU(self.conv5_1(hx))
        hx = self.ReLU(self.conv5_2(hx))
        hx = self.ReLU(self.conv5_3(hx))  # 32,512,12,16
        
        hx = self.Upsample(hx)  # 16 ->32
        hx = self.ReLU(self.conv4_1(hx))
        hx = self.ReLU(self.conv4_2(hx))
        hx = self.ReLU(self.conv4_3(hx))

        hx = self.Upsample(hx)  # 32 ->64
        hx = self.ReLU(self.conv3_1(hx))
        hx = self.ReLU(self.conv3_2(hx))
        hx = self.ReLU(self.conv3_3(hx))
        

        hx = self.Upsample(hx)  # 64 ->128
        hx = self.ReLU(self.conv2_1(hx))
        hx = self.ReLU(self.conv2_2(hx))
        hx = self.Upsample(hx)  # 128 ->256

        hx = self.ReLU(self.conv1_1(hx))
        hx = self.ReLU(self.conv1_2(hx))
        hx = self.outconv(hx)
        hx = F.sigmoid(hx)
        hx = hx.squeeze(1)

        return hx