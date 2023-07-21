import copy
import time
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from model import TransformerModel
from data import vocab, train_data, val_data, device
import math

ntokens = len(vocab) #size of vocabulary
emsize = 200 #embedding dimension
d_hid = 200 #dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 #number of nn.TransfomerEncoderLayer in nn.TransformEncoder
nhead = 2 #number of heads in nn.MultiheadAttention
dropout = 0.2 #dropout probability

model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)

criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

bptt = 35
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    '''
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int
    Returns:
        tuple(data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    '''
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def generate_square_subsequent_mask(sz: int) -> Tensor:
    '''Generates an upper-triangular matrix of -inf, with zeros on diag.'''
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def train(model: nn.Module) -> None:
    model.train()
    total_loss = 0.
    log_interval = 1
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        batch_size = data.size(0)
        if batch_size != bptt:
            src_mask = src_mask[:batch_size, :batch_size]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} |'
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
    
def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            batch_size = data.size(0)
            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += batch_size * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)

best_val_loss = float('inf')
epochs = 3
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model, val_data)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        torch.save(best_model.state_dict(), 'best_model.pt')
    scheduler.step()
