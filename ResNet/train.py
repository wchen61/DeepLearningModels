import torch
import torch.nn as nn
import torch.optim as optim

from model import resnet50
from data import data_train_loader, data_test_loader

model = resnet50()
model.train()
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

epoches = 20
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

torch.save(save_info, 'model.pth')

