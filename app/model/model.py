from hazm import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from app.model.dataset import SentimentDataset


dataset = SentimentDataset()

train_valid_dataset, test_dataset = random_split(list(dataset), [len(dataset)-10_000, 10_000])
train_dataset, valid_dataset = random_split(list(train_valid_dataset), [len(train_valid_dataset)-10_000, 10_000])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")

def text_pipeline(text):
    encoded = tokenizer.encode(text)
    return encoded

def collate_fn(batch):
    text_list, lengths = [], []

    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    for text in texts:
        transformed = text_pipeline(text)
        text_list.append(torch.tensor(transformed))
        lengths.append(len(transformed))
    
    label_list = torch.tensor(labels)    
    lengths = torch.tensor(lengths)
    padded_text_list = pad_sequence(text_list, batch_first=True)

    return padded_text_list.to(device), label_list.to(device), lengths.to(device)

batch_size = 256

train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_dl  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

class RNN(nn.Module):
    def __init__(self, vocab_size, num_embd, rnn_h, fcl_h):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, num_embd)
        self.rnn  = nn.LSTM(num_embd, rnn_h, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(rnn_h * 2, fcl_h)
        self.batch_norm1 = nn.BatchNorm1d(fcl_h)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(fcl_h, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, lengths):
        x = self.embd(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        _, (h, c) = self.rnn(x)
        x = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1) # -2 is the last backward state and -1 is the last forward state 
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.gelu(x)
        x = self.fc2(x) 
        x = self.sigmoid(x)
        return x