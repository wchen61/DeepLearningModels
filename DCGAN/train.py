from model import Generator, Discriminator
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "data/celeba"
workers = 2
batch_size = 128

#the height/width, channel of the input image to network
image_size = 64
nc = 3

#size of latent z vector
nz = 100

#feature size for networks
ngf = 64
ndf = 64

num_epochs = 5
lr = 0.00002
beta1 = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG = Generator(nz, ngf, nc).to(device)
netG.apply(weight_init)
print(netG)

netD = Discriminator(nc, ndf).to(device)
netD.apply(weight_init)
print(netD)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

dataset = datasets.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers)

img_list = []
G_losses = []
D_losses = []
imag_list = []

output_dir = 'results/'

print("Starting Training Loop...")

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        #1. update D network: maximize log(D(x)) + log(1-D(G(z)))
        #1.1 train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label,  
                            dtype=real_cpu.dtype, device=device)
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
    
        #1.2 train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        #2. update G network: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d %d][%d/%d]\tLoss_D:%.4f\tLoss_G:%.4f\tD(x):%.4f\tD(G(z)):%.4f/%.4f'
                    %(epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(),
                        D_x, D_G_z1, D_G_z2))
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (i % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            vutils.save_image(real_cpu, '%s/real_samples_epoch%d_iter%d.png'%(output_dir, epoch, i), normalize=True)
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                vutils.save_image(fake, '%s/fake_samples_epoch%d_iter%d.png'%(output_dir, epoch, i), normalize=True)
            imag_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth'%(output_dir, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth'%(output_dir, epoch))
