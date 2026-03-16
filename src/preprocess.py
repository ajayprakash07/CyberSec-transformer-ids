import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle
import json
import random


# ─────────────────────────────────────────────
# REPRODUCIBILITY — set all random seeds
# ─────────────────────────────────────────────

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seeds set to {seed}")


# ─────────────────────────────────────────────
# STEP 1 — Load
# ─────────────────────────────────────────────

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    print(f"Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# STEP 2 — Clean
# ─────────────────────────────────────────────

def clean_data(df):
    print("\nCleaning data...")

    # check duplicates — CICIDS2017 is known to have them
    dupes = df.duplicated().sum()
    print(f"Duplicate rows found: {dupes}")
    if dupes > 0:
        df.drop_duplicates(inplace=True)
        print(f"Duplicates removed. Remaining rows: {len(df)}")

    # replace infinity with NaN then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f"Removed {before - after} rows with NaN/Inf")
    print(f"Remaining rows: {after}")
    return df


# ─────────────────────────────────────────────
# STEP 3 — Features & Labels
# ─────────────────────────────────────────────

def prepare_features(df):
    print("\nPreparing features...")

    feature_cols = [
        'Flow Duration',
        'Total Fwd Packets',
        'Total Backward Packets',
        'Flow Bytes/s',
        'Flow Packets/s',
        'Destination Port',
        'Flow IAT Mean',
        'Flow IAT Std',
        'Fwd IAT Mean',
        'Bwd IAT Mean',
        'SYN Flag Count',
        'FIN Flag Count',
        'RST Flag Count',
        'PSH Flag Count',
        'ACK Flag Count',
        'Packet Length Mean',
        'Packet Length Std',
        'Average Packet Size',
        'Init_Win_bytes_forward',
        'Down/Up Ratio',
    ]

    for col in feature_cols:
        if col not in df.columns:
            print(f"WARNING: '{col}' not found")
            raise ValueError(f"Missing column: {col}")

    X = df[feature_cols].values
    y = df['Label'].apply(
        lambda label: 0 if label.strip() == 'BENIGN' else 1
    ).values

    # ── Dataset Diagnostics ───────────────────
    total   = len(y)
    benign  = (y == 0).sum()
    attack  = (y == 1).sum()

    print(f"\nDataset Diagnostics:")
    print(f"  Total samples : {total:,}")
    print(f"  Benign  (0)   : {benign:,} ({100*benign/total:.1f}%)")
    print(f"  Attack  (1)   : {attack:,} ({100*attack/total:.1f}%)")
    print(f"  Features      : {len(feature_cols)}")

    return X, y, feature_cols


# ─────────────────────────────────────────────
# STEP 4 — Time-based Split with Gap
# ─────────────────────────────────────────────

def split_data(X, y, seq_len=10):
    print("\nSplitting data with boundary gap to prevent leakage...")

    n         = len(X)
    gap       = seq_len * 2      # 20 row gap at each boundary

    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train = X[:train_end]
    X_val   = X[train_end + gap : val_end]
    X_test  = X[val_end   + gap :]

    y_train = y[:train_end]
    y_val   = y[train_end + gap : val_end]
    y_test  = y[val_end   + gap :]

    print(f"  Gap size : {gap} rows at each boundary")
    print(f"  Train    : {len(X_train):,} samples")
    print(f"  Val      : {len(X_val):,} samples")
    print(f"  Test     : {len(X_test):,} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# STEP 5 — Scale
# ─────────────────────────────────────────────

def scale_features(X_train, X_val, X_test):
    print("\nScaling features with StandardScaler...")

    scaler          = StandardScaler()
    X_train_scaled  = scaler.fit_transform(X_train)
    X_val_scaled    = scaler.transform(X_val)
    X_test_scaled   = scaler.transform(X_test)

    print("Scaling done.")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# STEP 6 — Save
# ─────────────────────────────────────────────

def save_processed(X_train, X_val, X_test,
                   y_train, y_val, y_test,
                   scaler, feature_cols,
                   out_dir='outputs/'):

    os.makedirs(out_dir, exist_ok=True)

    np.save(f'{out_dir}X_train.npy', X_train)
    np.save(f'{out_dir}X_val.npy',   X_val)
    np.save(f'{out_dir}X_test.npy',  X_test)
    np.save(f'{out_dir}y_train.npy', y_train)
    np.save(f'{out_dir}y_val.npy',   y_val)
    np.save(f'{out_dir}y_test.npy',  y_test)

    with open(f'{out_dir}scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open(f'{out_dir}feature_cols.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\nAll data saved to '{out_dir}'")
    print(f"Feature list saved. Total features: {len(feature_cols)}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    set_seeds(42)

    DATA_PATH = 'data/kaggle_data.csv'

    df                              = load_data(DATA_PATH)
    df                              = clean_data(df)
    X, y, feature_cols             = prepare_features(df)
    X_train, X_val, X_test, \
    y_train, y_val, y_test         = split_data(X, y, seq_len=10)
    X_train, X_val, X_test, scaler = scale_features(
                                         X_train, X_val, X_test)
    save_processed(X_train, X_val, X_test,
                   y_train, y_val, y_test,
                   scaler, feature_cols)