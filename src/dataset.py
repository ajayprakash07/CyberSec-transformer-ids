import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FlowSequenceDataset(Dataset):
    """
    Converts flat flow data into sequences using a sliding window.

    Input:
        X: numpy array of shape (n_flows, n_features)
        y: numpy array of shape (n_flows,)
        seq_len: how many flows per sequence (default 10)
        stride: how many steps to move window (default 1)

    Output per sample:
        X: tensor of shape (seq_len, n_features)
        y: tensor scalar — 1 if ANY flow in window is attack, else 0
    """

    def __init__(self, X, y, seq_len=10, stride=1):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride

        # calculate how many sequences we can make
        # example: 100 flows, seq_len=10, stride=1
        # → sequences start at index 0,1,2,...,90 → 91 sequences
        self.indices = list(range(0, len(X) - seq_len + 1, stride))

    def __len__(self):
        # tells PyTorch how many samples are in this dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # fetch one sequence
        start = self.indices[idx]
        end   = start + self.seq_len

        X_seq = self.X[start:end]          # shape: (seq_len, n_features)

        # flag sequence as attack if 30%+ of flows are attacks
        # more realistic than max() and more sensitive than majority vote
        attack_ratio = self.y[start:end].float().sum() / self.seq_len
        y_seq = (attack_ratio >= 0.3).long()

        return X_seq, y_seq


def get_dataloaders(batch_size=64, seq_len=10, stride=1):
    """
    Loads preprocessed data and returns train, val, test DataLoaders.
    """

    print("Loading preprocessed data...")

    # load the .npy files saved by preprocess.py
    X_train = np.load('outputs/X_train.npy')
    X_val   = np.load('outputs/X_val.npy')
    X_test  = np.load('outputs/X_test.npy')
    y_train = np.load('outputs/y_train.npy')
    y_val   = np.load('outputs/y_val.npy')
    y_test  = np.load('outputs/y_test.npy')

    print(f"X_train shape: {X_train.shape}")
    print(f"Creating sequences with seq_len={seq_len}, stride={stride}...")

    # create dataset objects
    train_dataset = FlowSequenceDataset(X_train, y_train, seq_len, stride)
    val_dataset   = FlowSequenceDataset(X_val,   y_val,   seq_len, stride)
    test_dataset  = FlowSequenceDataset(X_test,  y_test,  seq_len, stride)

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences:   {len(val_dataset)}")
    print(f"Test sequences:  {len(test_dataset)}")

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # shuffle training data every epoch
        num_workers=0       # keep 0 for Windows (avoids multiprocessing issues)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # never shuffle val/test — order doesn't matter here
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader