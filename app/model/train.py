import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.model.dataset import SentimentDataset, load_reviews, split_reviews
from app.model.model import RNN, collate_fn, device


DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'model_taghche-0.2.0.pth'


def train_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    for text, labels, lengths in tqdm(dataloader, total=len(dataloader)):
        optimizer.zero_grad()
        predictions = model(text, lengths).reshape(-1)
        loss = loss_fn(predictions, labels.float())
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_correct, total_loss, total_samples = 0, 0.0, 0
    with torch.no_grad():
        for text, labels, lengths in dataloader:
            predictions = model(text, lengths).reshape(-1)
            targets = labels.float()
            batch_size = targets.numel()
            total_loss += loss_fn(predictions, targets).item() * batch_size
            total_correct += ((predictions > 0.5) == (targets == 1)).sum().item()
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError('Cannot evaluate an empty dataset.')
    return total_correct / total_samples, total_loss / total_samples


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train with disjoint review splits.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size must be at least 2')

    torch.manual_seed(args.seed)
    train, valid, test = split_reviews(load_reviews(), random_state=args.seed)
    train_dl = DataLoader(
        SentimentDataset(reviews=train), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn,
        # BatchNorm needs at least two samples during training.
        drop_last=True,
    )
    if len(train_dl) == 0:
        parser.error('batch-size exceeds the number of training examples')
    valid_dl, test_dl = [
        DataLoader(
            SentimentDataset(reviews=part), batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn,
        )
        for part in (valid, test)
    ]

    # Start from scratch: the legacy checkpoint may have seen held-out reviews.
    model = RNN(100_000, 256, 128, 64).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    loss_fn = nn.BCELoss()
    best_valid_loss = float('inf')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_epoch(model, train_dl, optimizer, loss_fn)
        valid_accuracy, valid_loss = evaluate(model, valid_dl, loss_fn)
        scheduler.step(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.output)
        print(f'Epoch {epoch + 1}: validation loss {valid_loss:.4f}, '
              f'accuracy {valid_accuracy:.4f}')

    # Select by validation loss; use the test set only for the final evaluation.
    model.load_state_dict(torch.load(args.output, map_location=device, weights_only=True))
    test_accuracy, test_loss = evaluate(model, test_dl, loss_fn)
    print(f'Test loss: {test_loss:.4f}; test accuracy: {test_accuracy:.4f}')
    print(f'Checkpoint: {args.output}')


if __name__ == '__main__':
    main()
