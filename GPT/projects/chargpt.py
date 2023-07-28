import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from model import GPT
from trainer import Trainer
from utils import set_seed, setup_logging, CfgNode as CN

def get_config():
    C = CN()
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'
    
    C.data = CharDataset.get_default_config()
    
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'
    
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4
    return C

class CharDataset(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C
    
    def __init__(self, config, data):
        self.config = config
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i, ch in enumerate(chars) }
        self.itos = { i:ch for i, ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data
        print(self.stoi)
        print(self.itos)
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        return self.config.block_size
    
    def __len__(self):
        return len(self.data) - self.config.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.config.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
    
if __name__ == '__main__':
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    setup_logging(config)
    set_seed(config.system.seed)
    
    text = open('data/input.txt', 'r').read()
    train_dataset = CharDataset(config.data, text)
    
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    trainer = Trainer(config.trainer, model, train_dataset)
    print(config)
    print(model)
    
    def batch_end_callback(trainer):
        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        
        if trainer.iter_num % 500 == 0:
            model.eval()
            with torch.no_grad():
                context = "God, God!"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            model.train()
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()