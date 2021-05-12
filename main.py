# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 17:30:56 2021

@author: Team1
"""
import os, datetime
import argparse
from torch import autograd 

import torch.nn as nn
import torch
from matplotlib import pyplot as plt
import numpy as np
import random

import modules 
import creat_data

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser(description='PyTorch WGAN_Vgg')
parser.add_argument('--model', default='WGAN_Vgg', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--patch_size', default=80, type=int, help='patch size')
parser.add_argument('--patch_stride', default=60, type=int, help='patch stride')
parser.add_argument('--data', default="C:\\Users\\Team1\\lml\\LDCT\\Data\\Mayo_chest"
                    , type=str, help='path of train data and validation data')
parser.add_argument('--N_epoch', default=10000, type=int, help='number of total epoches')
parser.add_argument('--N_D', default=4, type=int, 
                    help='number of iteration for discrimintar train')
parser.add_argument('--lambda_', default=10, type=float, help='gradient penalty')
parser.add_argument('--alpha', default=1e-5, type=float, help='learning reta')
parser.add_argument('--beta1', default=0.5, type=float, help='adam optimizer parameter')
parser.add_argument('--beta2', default=0.9, type=float, help='adam optimizer parameter')
parser.add_argument('--lambda1', default=.10, type=float, help='perceptual loss weight ')
parser.add_argument('--print_freq', default=100, type=int, help='print frequence ')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_dir = os.path.join('C:\\Users\\Team1\\lml\\LDCT', args.model )
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

cri = nn.MSELoss()
def train(wgan_vgg, patch_in, patch_ta, args):
    num = len(patch_in)
    l = list(np.linspace(0,num-1,num, dtype = 'int'))#用于随机采样
    for epoch in range(args.N_epoch):
        #D training
        for t in range(args.N_D):
            random.shuffle(l)
            z = patch_in[l[0:args.batch_size]]
            x = patch_ta[l[0:args.batch_size]]
            G_z = wgan_vgg.G(z)
            epsilon = torch.rand(args.batch_size)
            loss_D = 0
            for i in range(args.batch_size):
                #一张一张的判断
                xi = x[i].unsqueeze(dim=1)
                G_zi = G_z[i].unsqueeze(dim=1)
                x_hati = epsilon[i]*xi + (1-epsilon[i])*G_zi
                x_hati = x_hati.detach().requires_grad_(True)
                D_G_zi = wgan_vgg.D(G_zi)
                D_xi = wgan_vgg.D(xi)
                D_x_hati = wgan_vgg.D(x_hati)
                grad_x_hat = autograd.grad(D_x_hati, x_hati)
                grad_norm = torch.norm(grad_x_hat[0], p=2)
                gradient_penalty = torch.square(grad_norm-1)
                loss_Di = D_G_zi - D_xi + args.lambda_*gradient_penalty
                loss_D += loss_Di
            loss_D /= args.batch_size
            wgan_vgg.loss_D_list.append(loss_D.item())
            wgan_vgg.optim_D.zero_grad()
            loss_D.backward()
            wgan_vgg.optim_D.step()
                
        if epoch % args.print_freq == 0:
            string = 'epcoh = %d, loss_D = %4.10f, L(Di) = %4.10f, D(G(zi)) = %4.10f, D(xi) = %4.10f, grad_pen = %4.10f'
            log(string % (epoch+1, loss_D, loss_Di, D_G_zi,  D_xi, gradient_penalty))
        #G training
        random.shuffle(l)
        z = patch_in[l[0:args.batch_size]]
        x = patch_ta[l[0:args.batch_size]]
        G_z = wgan_vgg.G(z)
        loss_G = 0
        for i in range(args.batch_size):
            xi = x[i].unsqueeze(dim=1)
            G_zi = G_z[i].unsqueeze(dim=1)
            #zi = z[i].unsqueeze(dim=1)
            D_G_zi = wgan_vgg.D(G_zi)
            loss_vgg = wgan_vgg.vgg.perceptual_loss(xi, G_zi)
        loss_Gi = args.lambda1*loss_vgg - 10*D_G_zi #+ cri(G_zi, xi)
        loss_G += loss_Gi
        loss_G /= args.batch_size
        if epoch % args.print_freq == 0:
            string = 'epcoh = %d, loss_G = %4.10f, L(Gi) = %4.10f, D(G(zi)) = %4.10f, loss_vgg = %4.10f\n'
            log(string % (epoch+1, loss_G, loss_Gi, D_G_zi,  loss_vgg))
                
            with torch.no_grad():
                output = wgan_vgg.forward(torch.from_numpy(train_in[0]).unsqueeze(dim=0).unsqueeze(dim=0).to(device))  
                creat_data.show(output[0,0].cpu().detach().numpy() , title = 'output %d'%epoch, cbar = True)
                plt.pause(0.01)
        wgan_vgg.loss_G_list.append(loss_G.item())
        wgan_vgg.optim_G.zero_grad()
        loss_G.backward()
        wgan_vgg.optim_G.step()
    
def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)
     
        
if __name__ == '__main__':
    #create data
    train_in = creat_data.get_inputs(args.data + "\\trainset", train = True, verbose = False)
    train_ta = creat_data.get_targets(args.data + "\\trainset", train = True, verbose = False)
    ldct, ndct = creat_data.get_traindata(train_in, train_ta, patch_size = args.patch_size,
                                          patch_stride = args.patch_stride, verbose = False)

    #create net
    g_net = modules.Generator_net()
    d_net =  modules.Discrimintor_net(args)
    vgg_net = modules.Vgg_net()
    wgan_vgg = modules.Wgan_Vgg(g_net, d_net, vgg_net, args)
    #cuda
    wgan_vgg = wgan_vgg.to(device)
    ldct, ndct = ldct.to(device), ndct.to(device)
        
    #training
    creat_data.show(train_in[0], title = 'LDCT', cbar = True)
    creat_data.show(train_ta[0], title = 'NDCT', cbar = True)
    train(wgan_vgg, ldct, ndct, args)
    
    plt.plot(wgan_vgg.loss_D_list)
    plt.plot(wgan_vgg.loss_G_list)
    
    torch.save(wgan_vgg.state_dict(), save_dir +"wgan_vgg_params.pkl")    
    wgan_vgg.load_state_dict(torch.load(save_dir +"wgan_vgg_params.pkl"))

    creat_data.show(ldct[7].cpu(), title = 'LDCT', cbar = 1)
    creat_data.show(ndct[7].cpu(), title = 'NDCT', cbar = 1)
    output = wgan_vgg.forward(ldct[0:10])    
    creat_data.show(output[7,0].cpu().detach().numpy() , title = 'output', cbar = 1)




