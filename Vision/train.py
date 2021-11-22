import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser

from resnet import Resnet50
from alexnet import AlexNet
from vgg import VGG16
from data import data_train_loader, data_test_loader

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
    
lr = args.lr
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

model.train()
epoches = args.epoches
for epoch in range(epoches):
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(data_train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    print(epoch, 'Loss:%.3f | Acc: %.3f%%(%d/%d)' %(
        train_loss/(batch_idx+1), 100. * correct/total, correct, total))


save_info = {}
save_info['iter_num'] = epoches
save_info['optimizer'] = optimizer.state_dict()
save_info['model'] = model.state_dict()

torch.save(save_info, 'model_' + args.model + '.pth')

