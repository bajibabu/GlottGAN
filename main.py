from __future__ import print_function
import argparse
import os
import random
#import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from data import load_data
from utils import plot_feats, read_binary_file
from models import define_netD, define_netG


def train(netD, netG, opt):
    # data preparation
    pls_input = torch.FloatTensor(opt.batchSize, 1, opt.pulseLen)
    ac_input = torch.FloatTensor(opt.batchSize, opt.acSize)
    noise = torch.FloatTensor(opt.batchSize, opt.nz)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz).normal_(0, 1)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        criterion.cuda()
        pls_input, label = pls_input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        ac_input = ac_input.cuda()

    pls_input = Variable(pls_input)
    ac_input = Variable(ac_input)
    label = Variable(label)
    noise = Variable(noise)
    fixed_noise = Variable(fixed_noise)
    
    # cost criterion
    criterion = nn.BCELoss() # normal gan 
    #criterion = nn.MSELoss() # lsgan
    
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    if opt.dataset in ['nick', 'jenny']:
        # folder dataset
        dataset, ac_dataset = load_data(opt.dataroot, num_files=5)
    else:
        raise
    dataset = torch.from_numpy(dataset)
    ac_dataset = torch.from_numpy(ac_dataset)
    train_dataset = torch.utils.data.TensorDataset(dataset,ac_dataset)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, 
                                            shuffle=True, num_workers=int(opt.workers))

    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            #################################
            # (1) Updata D network: maximize log(D(x)) + log(1 - D(G(z)))
            #################################
            # train with real 
            netD.zero_grad()
            real_pls_cpu, ac_feats_cpu = data
            
            if real_pls_cpu.size(0) != opt.batchSize:
                continue
                
            batch_size = real_pls_cpu.size(0)
            pls_input.data.copy_(real_pls_cpu)
            ac_input.data.copy_(ac_feats_cpu)
            label.data.fill_(real_label)
            
            output = netD(pls_input, ac_input)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake 
            noise.data.resize_(batch_size, nz)
            noise.data.normal_(0, 1)
            fake = netG(noise, ac_input)
            label.data.fill_(fake_label)
            output = netD(fake.detach(), ac_input)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ############################
            netG.zero_grad()
            label.data.fill_(real_label) # fake labels are real for generator cost
            output = netD(fake, ac_input)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                %(epoch, opt.niter, i, len(dataloader), 
                errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                fake = netG(fixed_noise, ac_input)
                fake = fake.data.cpu().numpy()
                fake = fake.reshape(batch_size, -1)
                real_data = real_pls_cpu.numpy()
                plot_feats(real_data, fake, epoch, i, opt.outf)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' %(opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' %(opt.outf, epoch))

def test(netG, opt):
    assert opt.netG != ''
    test_dir = opt.testdata_dir
    for f in os.listdir(test_dir):
        fname, ext = os.path.splitext(f)
        if ext == '.cmp':
            print(fname)
            cmp_file = os.path.join(test_dir, f)
            ac_data = read_binary_file(cmp_file, dim=47)
            ac_data = torch.FloatTensor(ac_data)
            noise = torch.FloatTensor(ac_data.size(0), nz)
            if opt.cuda:
                ac_data, noise = ac_data.cuda(), noise.cuda()
            ac_data = Variable(ac_data)
            noise = Variable(noise)
            noise.data.normal_(0, 1)
            generated_pulses = netG(noise, ac_data)
            generated_pulses = generated_pulses.data.cpu().numpy()
            generated_pulses = generated_pulses.reshape(ac_data.size(0), -1)
            out_file = os.path.join(test_dir, fname + '.pls')
            with open(out_file, 'wb') as fid:
                generated_pulses.tofile(fid)    


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='nick | jenny ')
    parser.add_argument('--mode', required=True, type=str, help='train | test')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--pulseLen', type=int, default=400, help='the length of each pulse')
    parser.add_argument('--acSize', type=int, default=47,  help='the acoustic input size')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--testdata_dir', type=str, help='path to test data')
    opt = parser.parse_args()
    print(opt)

    # prepare the output directories
    try:
        os.makedirs(opt.outf)
        os.makedirs(os.path.join(opt.outf, 'figures'))
    except OSError:
        pass
    # if manual seed is not provide then pick one randomly
    if opt.manualSeed is None:
        opt.manualSeed  = random.randint(1, 10000)
    print('Random Seed: ', opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    nz = int(opt.nz)
    # define the generator 
    netG = define_netG(opt.nz, opt.acSize)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    # define the discriminator
    netD = define_netD(opt.acSize)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    if opt.mode == 'train':
        train(netD, netG, opt)
    elif opt.mode == 'test':
        test(netG, opt)
    else:
        print('Mode must be either train or test only')