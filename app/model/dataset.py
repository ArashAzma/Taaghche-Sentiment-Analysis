import pandas as pd
from sklearn.utils import resample
from torch.utils.data import Dataset

from app.model.preprocess import preprocess

df = pd.read_csv('app/model/taghche.csv', encoding='utf-8')

df.dropna(inplace=True)

df_cleaned = list(map(preprocess, df["comment"]))

df = df.assign(comment_cleaned = df_cleaned)

df.drop_duplicates(subset='comment_cleaned', inplace=True)

max_count = df['rate'].value_counts().max()/4

df = pd.concat([
    resample(df[df['rate'] == rate], replace=True, n_samples=int(max_count), random_state=42)
    for rate in df['rate'].unique()
])

df['rate_filtered'] = df['rate'].apply(lambda x: 1 if x>=3 else 0)

df.drop(columns=['date', 'comment', 'bookname', 'bookID', 'like'], inplace=True)

class SentimentDataset(Dataset):
    
    def __init__(self, filtered=True):
        self.df = df
        self.filtered = filtered
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature = row['comment_cleaned']
        target = row['rate_filtered'] if self.filtered else row['rate']
        return feature, target