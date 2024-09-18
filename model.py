#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import pandas as pd
from hazm import *

from sklearn.utils import resample

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import AutoTokenizer
from tqdm import tqdm


# In[1]:


df = pd.read_csv('taghche.csv', encoding='utf-8')

df.head()


# In[2]:


df.isnull().sum()


# In[3]:


df.dropna(inplace=True)


# In[4]:


df.isnull().sum()


# In[23]:


stopword_files = ['verbal.txt', 'nonverbal.txt', 'chars.txt']
stopwords = []

for file in stopword_files:
    with open('stopwords\\' + file, encoding='utf-8') as f:
        stopwords += f.read().split('\n')

stopwords = set(stopwords)


# In[25]:


len(stopwords)


# In[31]:


normalizer = Normalizer()

def normal(text):
    text=str(text)
    text = normalizer.character_refinement(text)
    text = normalizer.punctuation_spacing(text)
    text = normalizer.affix_spacing(text)
    text = normalizer.normalize(text)
    return text

lemmatizer = Lemmatizer()

for word in df['comment'][3].split():
    print(word, lemmatizer.lemmatize(word), normalizer.normalize(word))
    

lemmatizer.lemmatize(df['comment'][3])


# In[44]:


normalizer = Normalizer(correct_spacing=True, remove_diacritics=True, remove_specials_chars=True, unicodes_replacement=True)
lemmatizer = Lemmatizer()
stemmer = Stemmer()

def remove_stopwords(text):
    text=str(text)
    filtered_tokens = [token for token in text.split() if token not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_emoji(text): 
    # Define a regex pattern to match various emojis and special characters
    emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags
                    u"\U00002702-\U000027B0"  # dingbats
                    u"\U000024C2-\U0001F251"  # enclosed characters
                    u"\U0001f926-\U0001f937"  # supplemental symbols and pictographs
                    u'\U00010000-\U0010ffff'  # supplementary private use area-A
                    u"\u200d"                 # zero-width joiner
                    u"\u200c"                 # zero-width non-joiner
                    u"\u2640-\u2642"          # gender symbols
                    u"\u2600-\u2B55"          # miscellaneous symbols
                    u"\u23cf"                 # eject symbol
                    u"\u23e9"                 # fast forward symbol
                    u"\u231a"                 # watch
                    u"\u3030"                 # wavy dash
                    u"\ufe0f"                 # variation selector-16
        "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r' ', text)

def remove_halfspace(text): 
    emoji_pattern = re.compile("["                
                u"\u200c"              
    "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r' ', text) 

def remove_link(text): 
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str(text))

def remove_picUrl(text):
    return re.sub(r'pic.twitter.com/[\w]*',"", str(text))

def remove_rt(text):
    z = lambda text: re.compile('\#').sub('', re.compile('RT @').sub('@', str(text), count=1).strip())
    return z(text)

def remove_hashtag(text):
    return re.sub(r"#[^\s]+", '', str(text))

def remove_mention(text):
    return re.sub(r"@[^\s]+", '', str(text))

def remove_email(text): 
    return re.sub(r'\S+@\S+', '', str(text))

def remove_numbers(text): 
    return re.sub(r'^\d+\s|\s\d+\s|\s\d+$', ' ', str(text))

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', str(text))

def remove_quote(text): 
    return  str(text).replace("'","")

def remove_chars(text): 
    return  re.sub(r'[$+&+;+]|[><!+،:’,\(\).+]|[-+]|[…]|[\[\]»«//]|[\\]|[#+]|[_+]|[—+]|[*+]|[؟+]|[?+]|[""]', ' ', str(text))

def remove_englishword(text): 
    return re.sub(r'[A-Za-z]+', '', str(text))

def remove_extraspaces(text):
    return re.sub(r' +', ' ', text)

def remove_extranewlines(text):
    return re.sub(r'\n\n+', '\n\n', text)


# In[50]:


def lemmatizer_text(text):
    words = []
    for word in text.split():
        words.append(lemmatizer.lemmatize(word))
    return ' '.join(words)

def stemmer_text(text):
    words = []
    for word in text.split():
        words.append(stemmer.stem(word))
    return ' '.join(words)

def normalizer_text(text):
    text = normalizer.normalize(text)
    text = stemmer_text(text)
    text = lemmatizer_text(text)
    return text


# In[51]:


def preprocess(text):
    text = remove_link(text)
    text = remove_picUrl(text)
    text = remove_englishword(text)
    text = normalizer_text(text)
    text = remove_stopwords(text)
    text = remove_emoji(text)
    text = remove_rt(text)
    text = remove_mention(text)
    text = remove_emoji(text)
    text = remove_hashtag(text)   
    text = remove_email(text) 
    text = remove_html(text) 
    text = remove_chars(text)
    text = remove_numbers(text)
    text = remove_quote(text)
    text = remove_extraspaces(text)
    text = remove_extranewlines(text)
    text = remove_halfspace(text) 
    text = remove_stopwords(text)
    return text


# In[52]:


df_cleaned = list(map(preprocess, df["comment"]))


# In[53]:


df = df.assign(comment_cleaned = df_cleaned)


# In[54]:


df.head()


# In[57]:


df.drop_duplicates(subset='comment_cleaned', inplace=True)


# In[59]:


df.duplicated().sum()


# In[78]:


max_count = df['rate'].value_counts().max()/4

df = pd.concat([
    resample(df[df['rate'] == rate], replace=True, n_samples=int(max_count), random_state=42)
    for rate in df['rate'].unique()
])
df['rate'].value_counts()


# In[80]:


df['rate_filtered'] = df['rate'].apply(lambda x: 1 if x>=3 else 0)


# In[84]:


df.drop(columns=['date', 'comment', 'bookname', 'bookID', 'like'], inplace=True)


# In[85]:


df.head()


# In[87]:


class SentimentDataset(Dataset):
    
    def __init__(self, df, filtered=True):
        self.df = df
        self.filtered = filtered
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature = row['comment_cleaned']
        target = row['rate_filtered'] if self.filtered else row['rate']
        return feature, target


# In[90]:


dataset = SentimentDataset(df)

train_valid_dataset, test_dataset = random_split(list(dataset), [len(dataset)-10_000, 10_000])
train_dataset, valid_dataset = random_split(list(train_valid_dataset), [len(train_valid_dataset)-10_000, 10_000])

print(len(train_dataset))
print(len(test_dataset))
print(len(valid_dataset))


# In[91]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")


# In[93]:


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


# In[147]:


batch_size = 256

train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_dl  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# In[167]:


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


# In[168]:


vocab_size = tokenizer.vocab_size
num_embd = 256
rnn_hidden = 128
fcl_hidden = 64
lr = 3e-3


# In[169]:


model = RNN(vocab_size, num_embd, rnn_hidden, fcl_hidden)
model = model.to(device)


# In[175]:


optim  = AdamW(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=5, verbose=True)
loss_fn = nn.BCELoss()

total_acc, total_loss = 0, 0


# In[176]:


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


# In[177]:


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
    


# 1 Test Accuracy: 0.8562
# 
# 2 Test Accuracy: 0.8627
# 
# 3 Test Accuracy: 0.8382
# 
# 4 Test Accuracy: 0.8668
# 
# **Test Accuracy: 0.8674**

# In[178]:


model.eval()

acc_valid, loss_valid = evaluate(test_dl, False)
print(f'Test Accuracy: {acc_valid:.4f}')


# In[179]:


# torch.save(model.state_dict(), 'model_taghche.pth')

