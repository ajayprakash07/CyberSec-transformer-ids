import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FlowSequenceDataset(Dataset):
    def __init__(self, X, y, seq_len=10, stride=1):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride
        self.indices = list(range(0, len(X) - seq_len + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end   = start + self.seq_len

        X_seq = self.X[start:end]       

        # flag sequence as attack if 30%+ of flows are attacks
        attack_ratio = self.y[start:end].float().sum() / self.seq_len
        y_seq = (attack_ratio >= 0.3).long()

        return X_seq, y_seq


def get_dataloaders(batch_size=64, seq_len=10, stride=1):
    
    print("Loading preprocessed data...")
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
        shuffle=True,       
        num_workers=0     
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,    
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader