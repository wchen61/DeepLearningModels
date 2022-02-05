import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser

from Classification.resnet import Resnet50
from Classification.alexnet import AlexNet
from Classification.vgg import VGG16
import Common.data as data
from Common.trainer import train


parser = ArgumentParser()
parser.add_argument('--model', default='Resnet50', help='Network')
parser.add_argument('--lr', type=int, default=0.01, help='Learning Rate')
parser.add_argument('--epoches', type=int, default=20, help='Epoches')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
args = parser.parse_args()

def get_model(model_str):
    if model_str == 'Resnet50':
        model = Resnet50()
    elif model_str == 'AlexNet':
        model = AlexNet()
    elif model_str == 'VGG16':
        model = VGG16()
    return model

model = get_model(args.model)

num_epochs, lr, batch_size = 10, 0.1, 256
train_iter, test_iter = data.load_data_fashion_mnist(batch_size)
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(256, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')
train(net, train_iter, test_iter, num_epochs, lr, 'cpu')

