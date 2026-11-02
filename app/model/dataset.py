from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import Dataset


DATA_PATH = Path(__file__).resolve().parent / 'taghche.csv'


def load_reviews(csv_path=DATA_PATH):
    """Clean each review once, without balancing the evaluation population."""
    from app.model.preprocess import preprocess

    reviews = pd.read_csv(csv_path, encoding='utf-8')
    reviews = reviews.dropna(subset=['comment', 'rate']).copy()
    reviews['comment_cleaned'] = reviews['comment'].map(preprocess)
    reviews = reviews.drop_duplicates(subset='comment_cleaned')
    reviews['rate_filtered'] = (reviews['rate'] >= 3).astype(int)
    return reviews.reset_index(drop=True)


def split_reviews(reviews, valid_size=10_000, test_size=10_000, random_state=42):
    """Split unique reviews first, then balance only the training partition."""
    reviews = reviews.drop_duplicates(subset='comment_cleaned')
    train_valid, test = train_test_split(
        reviews, test_size=test_size, stratify=reviews['rate_filtered'],
        random_state=random_state,
    )
    train, valid = train_test_split(
        train_valid, test_size=valid_size, stratify=train_valid['rate_filtered'],
        random_state=random_state,
    )

    # Resampled copies can only belong to training, never validation or test.
    target_count = train['rate_filtered'].value_counts().max()
    train = pd.concat([
        resample(
            group, replace=len(group) < target_count,
            n_samples=int(target_count), random_state=random_state,
        )
        for _, group in train.groupby('rate_filtered')
    ])
    return tuple(part.reset_index(drop=True) for part in (train, valid, test))


class SentimentDataset(Dataset):
    def __init__(self, filtered=True, reviews=None):
        self.df = load_reviews() if reviews is None else reviews.reset_index(drop=True)
        self.filtered = filtered

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = row['rate_filtered'] if self.filtered else row['rate']
        return row['comment_cleaned'], target
