import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import RNN, train_dl, test_dl, valid_dl 
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 100_000
num_embd = 256
rnn_hidden = 128
fcl_hidden = 64
lr = 3e-3

model = RNN(vocab_size, num_embd, rnn_hidden, fcl_hidden)
model = model.to(device)

model.load_state_dict(torch.load('app/model/model_taghche-0.1.0.pth'))

optim  = AdamW(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=5, verbose=True)
loss_fn = nn.BCELoss()

total_acc, total_loss = 0, 0

def evaluate(dataloader, minimize = True):
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        i = 0
        for j, (text_batch, label_batch, lengths) in enumerate(dataloader):
            out = model(text_batch, lengths)
            
            pred = out.squeeze()
            target = label_batch.float()
            
            loss = loss_fn(pred, target)
            batch_acc = ((pred > 0.5) == (target==1)).sum()/len(pred)
            total_acc += batch_acc.item()
            total_loss += loss.item()
            i+=1
            if(j>1000 and minimize):
                break
            
    return total_acc/i, total_loss/i

num_epoch = 100

for i in range(num_epoch):
    model.train()
    for text_batch, label_batch, lengths in tqdm(train_dl, total=len(train_dl)):
        out = model(text_batch, lengths)
        
        pred = out.squeeze()
        target = label_batch.float()
        
        optim.zero_grad()
        loss = loss_fn(pred, target)
        
        loss.backward()
        optim.step()
        with torch.no_grad():
            batch_acc = ((pred > 0.5) == (target==1)).sum()/len(pred)
            total_acc += batch_acc.item()
            total_loss += loss.item()
        
    scheduler.step(loss_valid)
    print(f'Loss: {loss.item()}')
    acc_valid, loss_valid = evaluate(valid_dl)
    print(f'Epoch {i} accuracy: {acc_valid:.4f}')
    

model.eval()

acc_valid, loss_valid = evaluate(test_dl, False)
print(f'Test Accuracy: {acc_valid:.4f}')