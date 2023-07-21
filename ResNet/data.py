import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

data_train = CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=2)
data_test = CIFAR10('./data', train=False, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
data_test_loader = DataLoader(data_test, batch_size=256, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(0.4914, 0.4822, 0.4465) 
    std = np.array(0.2023, 0.1994, 0.2010)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

if __name__ == '__main__':
    figure = plt.figure()
    num_of_images = 60
    for imgs, targets in data_train_loader:
        break
    
    for index in range(num_of_images):
        plt.subplot(6, 10, index+1)
        plt.axis('off')
        img = imgs[index, ...]
        plt.imshow(img.numpy().squeeze().transpose((1, 2, 0)))
    plt.show()