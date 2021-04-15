# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:09:59 2021

@author: lml
"""
import argparse
import os, datetime
import numpy as np
from matplotlib import pyplot as plt
import random

import torch
import torch.nn as nn
import torchvision.models as models

from torch import autograd 
import torch.nn.init as init
import torch.optim as optim

import creat_data


    
parser = argparse.ArgumentParser(description='PyTorch WGAN_Vgg')
parser.add_argument('--model', default='WGAN_Vgg', type=str, help='choose a type of model')
parser.add_argument('--m', default=20, type=int, help='batch size')
parser.add_argument('--patch_size', default=80, type=int, help='patch size')
parser.add_argument('--patch_stride', default=35, type=int, help='patch stride')
parser.add_argument('--data', default="F:\BaiduNetdiskDownload\LDCT_ImageData\SV_50mAs"
                    , type=str, help='path of train data and validation data')
parser.add_argument('--N_epoch', default=100, type=int, help='number of total epoches')
parser.add_argument('--N_D', default=4, type=int, 
                    help='number of iteration for discrimintar train')
parser.add_argument('--lambda_', default=10, type=float, help='gradient penalty')
parser.add_argument('--alpha', default=1e-5, type=float, help='learning reta')
parser.add_argument('--beta1', default=0.5, type=float, help='adam optimizer parameter')
parser.add_argument('--beta2', default=0.9, type=float, help='adam optimizer parameter')
parser.add_argument('--lambda1', default=0.1, type=float, help='perceptual loss weight ')
parser.add_argument('--print_freq', default=20, type=int, help='print frequence ')

args = parser.parse_args()
cuda = torch.cuda.is_available()
vgg = models.vgg19(pretrained=True).features

save_dir = os.path.join('E:\li\Desktop\首都师范大学\课程\learn\LDCT\models', args.model )
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

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
    def __init__(self):
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
        fcn_layels.append(nn.LeakyReLU(inplace=True, negative_slope = 0.1))
        
        self.dis_cnn = nn.Sequential(*layers_discrimintor)
        self.dis_fcn = nn.Sequential(*fcn_layels)

    def forward(self, x):
        x = self.dis_cnn(x)
        x = x.view(-1)
        out = self.dis_fcn(x)
        return out
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
   

def vgg_feature(img, cnn = vgg):
    img = torch.cat([img]*3, dim=1)
    for i, layer in enumerate(cnn):
        img = layer(img)
        if isinstance(layer, nn.Conv2d):
            feature = img.clone()
    return feature


def perceptual_loss(x, z):
    vgg_x = vgg_feature(x)
    vgg_z = vgg_feature(z)
    _, d, h, w = vgg_x.shape
    loss_vgg = torch.sqrt(torch.sum(torch.square(vgg_z-vgg_x)))
    loss_vgg /= (w*h*d)
    return loss_vgg.item()

def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


class Wgan_Vgg(nn.Module):
    def __init__(self, g_net, d_net):
        super(Wgan_Vgg, self).__init__()
        self.G = g_net
        self.D = d_net
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.alpha, betas=(args.beta1, args.beta2))
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.alpha, betas=(args.beta1, args.beta2))
        self.loss_D_list = []
        self.loss_G_list = []
                                  
    def train(self, patch_in, patch_ta):
        num = len(patch_in)
        l = list(np.linspace(0,num-1,num, dtype = 'int'))#用于随机采样
        for epoch in range(args.N_epoch):
            #D training
            for t in range(args.N_D):
                random.shuffle(l)
                z = patch_in[l[0:args.m]]
                x = patch_ta[l[0:args.m]]
                G_z = self.G(z)
                epsilon = torch.rand(args.m)
                loss_D = 0
                for i in range(args.m):
                    #一张一张的判断
                    xi = x[i].unsqueeze(dim=1)
                    G_zi = G_z[i].unsqueeze(dim=1)
                    x_hati = epsilon[i]*xi + (1-epsilon[i])*G_zi
                    x_hati = x_hati.detach().requires_grad_(True)
                    D_G_zi = self.D(G_zi)
                    D_xi = self.D(xi)
                    D_x_hati = self.D(x_hati)
            
                    grad_x_hat = autograd.grad(D_x_hati, x_hati)[0]
                    gradient_penalty = torch.square(torch.norm(grad_x_hat, p=2)-1)
                    loss_Di = D_G_zi - D_xi + args.lambda_*gradient_penalty
                    loss_D += loss_Di
                loss_D /= args.m
                self.loss_D_list.append(loss_D.item())
                self.optim_D.zero_grad()
                loss_D.backward()
                self.optim_D.step()
            if epoch % args.print_freq == 0:
                string = 'epcoh = %d, loss_D = %4.4f, L(Di) = %4.4f, D(G(zi)) = %4.4f, D(xi) = %4.4f, grad_pen = %4.4f'
                log(string % (epoch+1, loss_D, loss_Di, D_G_zi,  D_xi, gradient_penalty))
            #G training
            random.shuffle(l)
            z = patch_in[l[0:args.m]]
            x = patch_ta[l[0:args.m]]
            G_z = self.G(z)
            loss_G = 0
            for i in range(args.m):
                 xi = x[i].unsqueeze(dim=1)
                 G_zi = G_z[i].unsqueeze(dim=1)
                 zi = z[i].unsqueeze(dim=1)
                 D_G_zi = self.D(G_zi)
                 loss_vgg = perceptual_loss(xi, zi)
            loss_Gi = args.lambda1*loss_vgg - D_G_zi
            loss_G += loss_Gi
            loss_G /= args.m
            if epoch % args.print_freq == 0:
                string = 'epcoh = %d, loss_G = %4.4f, L(Gi) = %4.4f, D(G(zi)) = %4.4f, loss_vgg = %4.4f\n'
                log(string % (epoch+1, loss_G, loss_Gi, D_G_zi,  loss_vgg))
            self.loss_G_list.append(loss_G.item())
            self.optim_G.zero_grad()
            loss_G.backward()
            self.optim_G.step()
          
    def forward(self, x):
        out = self.G(x)
        return out





if __name__ == '__main__': 


    

    train_in, train_ta, val_in, val_ta = creat_data.div_train_val(args.data, ratio = .2)
    
    train_in, train_ta = train_in[0:2], train_ta[0:2]
    patch_in, patch_ta = creat_data.datagenerator(train_in, train_ta, batch_size = args.m, 
                  patch_size = args.patch_size, patch_stride = args.patch_stride, ratio = .8)
    def normalize(patch, max_ = 4000, min_=0):
        #-1 ~ 1
        return 2 * ((patch - min_) / (max_  -  min_)) -1
        

    g_net = Generator_net()
    d_net = Discrimintor_net()  
    '''d_net._initialize_weights()
    g_net._initialize_weights()'''

    wgan_vgg = Wgan_Vgg(g_net, d_net) 
    if cuda:
        wgan_vgg = wgan_vgg.cuda()
        patch_in = patch_in.cuda()
        patch_ta = patch_ta.cuda()
        vgg = vgg.cuda()

        patch_ta = patch_ta.cuda()
            
    patch_in = normalize(patch_in)
    patch_ta = normalize(patch_ta)

    wgan_vgg.train(patch_in, patch_ta)
        
    plt.plot(wgan_vgg.loss_D_list)
    plt.plot(wgan_vgg.loss_G_list)
    
    creat_data.show(patch_in[0,0].cpu(), cbar = 1)
    creat_data.show(patch_ta[0,0].cpu(), cbar = 1)
    g = wgan_vgg.forward(patch_in)
    creat_data.show(g[0,0].cpu().detach().numpy() , cbar = 1)
    
    creat_data.show(train_in[0], cbar = 1)
    creat_data.show(train_ta[0], cbar = 1)
    output = wgan_vgg.forward(torch.from_numpy(train_in).unsqueeze(dim=1).cuda())
    creat_data.show(output[0,0].cpu().detach().numpy(), cbar = 1)
    creat_data.show(output[0,0].cpu().detach().numpy()-train_ta[0], cbar = 1)
    
    

    

