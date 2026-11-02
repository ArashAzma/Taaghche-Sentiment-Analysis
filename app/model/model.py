from functools import lru_cache

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")

def text_pipeline(text):
    encoded = get_tokenizer().encode(text)
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
        x = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1) # final forward and backward states
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.gelu(x)
        x = self.fc2(x) 
        x = self.sigmoid(x)
        return x
