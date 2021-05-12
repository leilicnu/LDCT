# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:09:59 2021

@author: lml
"""

import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.init as init
import torch.optim as optim


class Wgan_Vgg(nn.Module):
    def __init__(self, g_net, d_net, vgg, args):
        super(Wgan_Vgg, self).__init__()
        self.G = g_net
        self.D = d_net
        self.vgg = vgg
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.alpha, betas=(args.beta1, args.beta2))
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.alpha, betas=(args.beta1, args.beta2))
        self.loss_D_list = []
        self.loss_G_list = []
          
    def forward(self, x):
        out = self.G(x)
        return out

class Generator_net(nn.Module):
    def __init__(self):
        super(Generator_net, self).__init__()
        layers_generator = []
        
        layers_generator.append(nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        layers_generator.append(nn.ReLU(inplace=True))
        for _ in range(6):
            layers_generator.append(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            layers_generator.append(nn.ReLU(inplace=True))
        layers_generator.append(nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        layers_generator.append(nn.ReLU(inplace=True))
        self.gen_cnn = nn.Sequential(*layers_generator)
        
    def forward(self, x):
        out = self.gen_cnn(x)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
   
class Discrimintor_net(nn.Module):
    def __init__(self, args):
        super(Discrimintor_net, self).__init__()
        self.N = args.patch_size
        for i in range(3):
            self.N = (self.N-3)//1 + 1
            self.N = (self.N-3)//2 + 1
        layers_discrimintor = []
        fcn_layels = []
        
        layers_discrimintor.append(nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1)))
        layers_discrimintor.append(nn.LeakyReLU(inplace=True, negative_slope = 0.1))
        
        layers_discrimintor.append(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2)))
        layers_discrimintor.append(nn.LeakyReLU(inplace=True, negative_slope = 0.1))
        
        layers_discrimintor.append(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)))
        layers_discrimintor.append(nn.LeakyReLU(inplace=True, negative_slope = 0.1))
        
        layers_discrimintor.append(nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2)))
        layers_discrimintor.append(nn.LeakyReLU(inplace=True, negative_slope = 0.1))
        
        layers_discrimintor.append(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)))
        layers_discrimintor.append(nn.LeakyReLU(inplace=True, negative_slope = 0.1))
        
        layers_discrimintor.append(nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2)))
        layers_discrimintor.append(nn.LeakyReLU(inplace=True, negative_slope = 0.1))
        
        #对于64*64的输入,卷积后变为5*5 (N+2*pad-ker)//s+1
        fcn_layels.append(nn.Linear(1*256*self.N*self.N, 1024))
        fcn_layels.append(nn.LeakyReLU(inplace=True, negative_slope = 0.1))
        
        fcn_layels.append(nn.Linear(1024, 1))
        #fcn_layels.append(nn.LeakyReLU(inplace=True, negative_slope = 0.1))
        
        self.dis_cnn = nn.Sequential(*layers_discrimintor)
        self.dis_fcn = nn.Sequential(*fcn_layels)

    def forward(self, x):
        x = self.dis_cnn(x)
        x = x.view(-1)
        x = self.dis_fcn(x)
        #x = torch.sigmoid(x)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
   
class Vgg_net(nn.Module):
    def __init__(self):
        super(Vgg_net, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        layers_features = []
        for layel in vgg_features:
            layers_features.append(layel)
        self.features = nn.Sequential(*layers_features)
        
    def extract_feature(self, img):
        img = torch.cat([img]*3, dim=1)
        for i, layer in enumerate(self.features):
            img = layer(img)
            if isinstance(layer, nn.Conv2d):
                feature = img.clone()
        return feature
    
    def perceptual_loss(self, x, z):
        vgg_x = self.extract_feature(x)
        vgg_z = self.extract_feature(z)
        _, d, h, w = vgg_x.shape
        loss_vgg = torch.sqrt(torch.sum(torch.square(vgg_z-vgg_x)))
        loss_vgg /= (w*h*d)
        return loss_vgg.item()

    def forward(self, x):
        out = self.features(x)
        return out

         
         
         
         
         
         





