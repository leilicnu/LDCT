import argparse
import os
import time
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from inout_util import *
from modules import *

parser = argparse.ArgumentParser(description='PyTorch WGAN_Vgg')
parser.add_argument('--model', default='WGAN_Vgg', type=str, help='choose a type of model')
parser.add_argument('--data_path', default='C:\\Users\\Team1\\lml\\LDCT\\Data\\Mayo_chest\\trainset',
                    type=str, help='path of data')
parser.add_argument('--save_path', default='C:\\Users\\Team1\\lml\\LDCT\\WGAN_Vgg\\abdomen', type=str, help='path of saved data')
parser.add_argument('--save_iters', default=100, type=int, help='save frequency')
parser.add_argument('--show_output', default=True, type=bool, help='show picture')
parser.add_argument('--result_fig', default=True, type=bool, help='save test picture')

parser.add_argument('--data_range', default=255, type=float, help='0 ~ 255')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--patch_size', default=80, type=int, help='patch size')
parser.add_argument('--patch_stride', default=30, type=int, help='patch stride')
parser.add_argument('--air_ratio', default=.1, type=float, help='the ratio of air')
parser.add_argument('--print_progress', default=False, type=bool, help='loading image')

parser.add_argument('--train', default=True, type=bool, help='train or test')
parser.add_argument('--N_epoch', default=10, type=int, help='number of total epochs')
parser.add_argument('--N_D', default=4, type=int, help='number of iteration for discriminator train')
parser.add_argument('--print_freq', default=100, type=int, help='print frequency ')

parser.add_argument('--lambda_', default=10, type=float, help='gradient penalty')
parser.add_argument('--alpha', default=1e-5, type=float, help='learning reta')
parser.add_argument('--beta1', default=0.5, type=float, help='adam optimizer parameter')
parser.add_argument('--beta2', default=0.9, type=float, help='adam optimizer parameter')
parser.add_argument('--lambda1', default=.10, type=float, help='perceptual loss weight ')

parser.add_argument('--device', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), type=str)
args = parser.parse_args()


def train(wgan_vgg, data_loader, args):
    wgan_vgg = wgan_vgg.to(args.device)
    train_losses = {'G':[],'D':[],'W':[],'grad':[]}
    total_iters = 0
    start_time = time.time()
    for epoch in range(args.N_epoch):
        for iter_, (z, x) in enumerate(data_loader):
            total_iters += 1
            z = z.to(args.device)
            x = x.to(args.device)
            # D training
            wgan_vgg.optim_D.zero_grad()
            wgan_vgg.D.zero_grad()
            for _ in range(args.N_D):
                G_z = wgan_vgg.G(z)
                epsilon = torch.cuda.FloatTensor(np.random.random((x.size(0), 1, 1, 1)))
                x_hat = (epsilon * x + (1 - epsilon) * G_z).detach().requires_grad_(True)
                D_x = wgan_vgg.D(x)
                D_G_z = wgan_vgg.D(G_z)
                D_x_hat = wgan_vgg.D(x_hat)

                fake_ = torch.cuda.FloatTensor(x.shape[0], 1).fill_(1.0).requires_grad_(False)
                gradients = torch.autograd.grad(outputs=D_x_hat, inputs=x_hat, grad_outputs=fake_,
                    create_graph=True, retain_graph=True, only_inputs=True)[0]

                gradients = gradients.view(gradients.size(0), -1)
                g_p = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                loss_D = torch.mean(D_G_z - D_x + args.lambda_ * g_p)
                loss_D.backward()
                wgan_vgg.optim_D.step()
            # G training
            wgan_vgg.optim_G.zero_grad()
            wgan_vgg.G.zero_grad()
            G_z = wgan_vgg.G(z)
            D_G_z = wgan_vgg.D(G_z)
            loss_vgg = wgan_vgg.vgg.perceptual_loss(x, G_z)
            loss_G = torch.mean(args.lambda1 * loss_vgg - D_G_z)
            loss_G.backward()
            wgan_vgg.optim_G.step()
            # save loss
            train_losses['D'].append(loss_G.item())
            train_losses['G'].append(loss_D.item())
            train_losses['W'].append(loss_D.item() - g_p.item())
            train_losses['grad'].append(g_p.item())
            if total_iters % args.print_freq == 0:
                print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}], TIME [{:.1f}s],\n G_LOSS: {:.8f}, D_LOSS: {:.8f}\n".format(
                    total_iters, epoch+1, args.N_epoch, iter_ + 1, len(data_loader), time.time() - start_time,
                    loss_G.item(), loss_D.item()))
                if args.show_output:
                    output = wgan_vgg.forward(z[0].unsqueeze(dim=0))
                    show(output[0,0].cpu().detach().numpy(), title = 'output %d'%(total_iters), cbar = True) 
            # save model
            if total_iters % args.save_iters == 0:
                torch.save(wgan_vgg.state_dict(), args.save_path + '\\param' +"\\wgan_vgg_params%d.pkl"%total_iters)
    return train_losses

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill=' '):
    # referred from https://gist.github.com/snakers4/91fa21b9dda9d055a02ecd23f24fbc3d
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '=' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()

def save_fig(z, x, G_z, fig_name, ori_psnr_avg, ori_ssim_avg, ori_rmse_avg, pred_psnr_avg, pred_ssim_avg, pred_rmse_avg):
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(z, cmap=plt.cm.gray)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(ori_psnr_avg,
                                                                           ori_ssim_avg,
                                                                           ori_rmse_avg), fontsize=20)
        ax[1].imshow(G_z, cmap=plt.cm.gray)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_psnr_avg,
                                                                           pred_ssim_avg,
                                                                           pred_rmse_avg), fontsize=20)
        ax[2].imshow(x, cmap=plt.cm.gray)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(args.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()
  
def compare(z, x, data_range):
    psnr = peak_signal_noise_ratio(x, z, data_range=data_range)
    ssim = structural_similarity(x, z, data_range=data_range)
    rmse = np.sqrt(((x-z)**2).mean())
    return [psnr, ssim, rmse]

def test(wgan_vgg, data_loader, args):
    wgan_vgg = wgan_vgg.to(args.device)
    with torch.no_grad():
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        for i, (z, x) in enumerate(data_loader):
            z = z.to(args.device)
            G_z = wgan_vgg.forward(z).cpu().detach().numpy()[0,0]
            z = z.cpu().detach().numpy()[0,0]
            x = x.cpu().detach().numpy()[0,0]
            ori_psnr, ori_ssim, ori_rmse = compare(z, x, data_range=args.data_range) 
            pred_psnr, pred_ssim, pred_rmse = compare(G_z, x, data_range=args.data_range) 
            ori_psnr_avg += ori_psnr
            ori_ssim_avg += ori_ssim
            ori_rmse_avg += ori_rmse
            pred_psnr_avg += pred_psnr
            pred_ssim_avg += pred_ssim
            pred_rmse_avg += pred_rmse
            printProgressBar(i, len(data_loader),prefix="Compute measurements ..",suffix='Complete', length=25)
            # save result figure
            if args.result_fig:
                save_fig(z, x, G_z, i, ori_psnr, ori_ssim, ori_rmse, pred_psnr, pred_ssim, pred_rmse)
        print('\n')
        print('Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
            ori_psnr_avg/len(data_loader), ori_ssim_avg/len(data_loader), ori_rmse_avg/len(data_loader)))
        print('After learning\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
            pred_psnr_avg/len(data_loader), pred_ssim_avg/len(data_loader), pred_rmse_avg/len(data_loader)))
        
def save_loss(train_losses):
    f = plt.figure(figsize=(30, 10))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses['G'])
    plt.title('loss_G')
    plt.subplot(1, 2, 2)
    plt.plot(train_losses['D'])
    plt.title('loss_D')
    f.savefig(os.path.join(args.save_path, 'loss', '{}.png'.format('G & D')))
    plt.close()
    f = plt.figure(figsize=(30, 10))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses['W'])
    plt.title('loss_Wasserstein')
    plt.subplot(1, 2, 2)
    plt.plot(train_losses['grad'])
    plt.title('gradient punishment')
    f.savefig(os.path.join(args.save_path, 'loss', '{}.png'.format('W & grad')))
    plt.close()



if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # create data
    train_dataset = MyDataset(args)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    #print(len(train_dataset))
    
    # create net
    wgan_vgg = Wgan_Vgg(Generator_net(), Discriminator_net(args), Vgg_net(), args)
    # training
    train_losses = train(wgan_vgg, train_data_loader, args)
    # save loss
    save_loss(train_losses)
    
    #test
    args.train = False
    args.data_path = 'C:\\Users\\Team1\\lml\\LDCT\\Data\\Mayo_abdomen\\testset'
    test_dataset = MyDataset(args)
    test_data_loader = DataLoader(dataset=test_dataset, shuffle=True)
    
    test(wgan_vgg, test_data_loader, args)

    #print(torch.cuda.memory_summary())
    #torch.cuda.empty_cache()
    
    #save and load
    #torch.save(wgan_vgg.state_dict(), args.save_path +'\\param'+"\\wgan_vgg_params14700.pkl")    
    #wgan_vgg.load_state_dict(torch.load(args.save_path +'\\param'+"\\wgan_vgg_params14700.pkl"))
    """for i, (z, x) in enumerate(test_data_loader):
        z = z.to(args.device)
        o = wgan_vgg.forward(z[0].unsqueeze(dim=0))
        x = x[0,0].cpu().detach().numpy()
        z = z[0,0].cpu().detach().numpy()
        o = o[0,0].cpu().detach().numpy()
        img = np.hstack((z,o,x))
        show(img, 'LDCT                OUT                 NDCT ')
        break
        
        """
    

    
