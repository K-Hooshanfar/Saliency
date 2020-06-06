import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from PIL import Image

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.hhh_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.conv = nn.Conv2d(in_channels = in_dim//8, out_channels = in_dim, kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):

        m_batchsize,C,width ,height = x.size()   
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height)
        proj_hhh = self.hhh_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        energy =  torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)
        out = torch.bmm(attention,proj_hhh)
        out = out.view(m_batchsize,C//8,width,height)
        out = self.conv(out)
        out = self.gamma*out + x 
        return out, attention

class VGGSAM(nn.Module):
    def __init__(self,n_channels):
        super(VGGSAM,self).__init__()

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

        self.attn1 = Self_Attn(512, 'relu')
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

        hx = x
        ## -------------Encoder-------------

        out1 = self.conv_layer1(hx)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)

        ## -------------Decoder-------------
        hx,p1 = self.attn1(out5)
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