import torch
import torch.nn as nn
from model import LeNet
from data import data_test_loader

model_path = "./model.pth"
save_info = torch.load(model_path)
model = LeNet()
criterion = nn.CrossEntropyLoss()
model.load_state_dict(save_info['model'])
model.eval()

test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(data_test_loader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print(batch_idx, len(data_test_loader), 'Loss:%.3f | Acc:%.3f%%(%d/%d)' %
                (test_loss/(batch_idx+1), 100.*correct/total, correct, total))