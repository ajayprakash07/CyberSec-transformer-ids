import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import pickle
# ─────────────────────────────────────────────
# STEP 1 — Load the CSV
# ─────────────────────────────────────────────
def load_data(filepath): 
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # strip whitespace from column names
    # CICIDS2017 has annoying spaces like ' Label', ' Flow Duration'
    df.columns = df.columns.str.strip()

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df
# ─────────────────────────────────────────────
# STEP 2 — Clean the data
# ─────────────────────────────────────────────
def clean_data(df):
    print("\nCleaning data...")

    # replace infinity values with NaN first
    # (Flow Bytes/s = bytes/duration, if duration=0 → infinity)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(df)
    df.dropna(inplace=True)
    after = len(df)

    print(f"Removed {before - after} rows with NaN/Inf values")
    print(f"Remaining rows: {after}")
    return df
# ─────────────────────────────────────────────
# STEP 3 — Select features & encode labels
# ─────────────────────────────────────────────
def prepare_features(df):
    print("\nPreparing features...")

    # these are the features your assignment specifies
    feature_cols = [
    # original 6
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Destination Port',

    # inter arrival times — best bot detector
    'Flow IAT Mean',
    'Flow IAT Std',
    'Fwd IAT Mean',
    'Bwd IAT Mean',

    # tcp flags — reveals attack type signatures
    'SYN Flag Count',
    'FIN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',

    # packet length stats — DDoS signature
    'Packet Length Mean',
    'Packet Length Std',
    'Average Packet Size',

    # connection behavior
    'Init_Win_bytes_forward',
    'Down/Up Ratio',
]

    # check all columns exist
    for col in feature_cols:
        if col not in df.columns:
            print(f"WARNING: '{col}' not found in dataset")
            print("Available columns:", list(df.columns))
            raise ValueError(f"Missing column: {col}")

    X = df[feature_cols].values  # numpy array of shape (n_rows, 6)

    # encode labels
    # BENIGN → 0, everything else → 1 (binary classification)
    y = df['Label'].apply(
        lambda label: 0 if label.strip() == 'BENIGN' else 1
    ).values

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        name = 'BENIGN' if u == 0 else 'ATTACK'
        print(f"  {name} ({u}): {c} samples ({100*c/len(y):.1f}%)")

    return X, y
# ─────────────────────────────────────────────
# STEP 4 — Split into train / val / test
# ─────────────────────────────────────────────
def split_data(X, y):
    print("\nSplitting data (70% train, 15% val, 15% test)...")

    # first split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y        # maintain same class ratio in each split
    )

    # second split: split the 30% into 15% val + 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test
# ─────────────────────────────────────────────
# STEP 5 — Scale features
# ─────────────────────────────────────────────
def scale_features(X_train, X_val, X_test):
    print("\nScaling features with StandardScaler...")

    scaler = StandardScaler()

    # fit ONLY on training data — learn mean & std from train
    X_train_scaled = scaler.fit_transform(X_train)

    # apply same scaler to val and test — no fitting here
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    print("Scaling done.")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
# ─────────────────────────────────────────────
# STEP 6 — Save everything
# ─────────────────────────────────────────────
def save_processed(X_train, X_val, X_test,
                   y_train, y_val, y_test, scaler,
                   out_dir='outputs/'):

    os.makedirs(out_dir, exist_ok=True)

    np.save(f'{out_dir}X_train.npy', X_train)
    np.save(f'{out_dir}X_val.npy',   X_val)
    np.save(f'{out_dir}X_test.npy',  X_test)
    np.save(f'{out_dir}y_train.npy', y_train)
    np.save(f'{out_dir}y_val.npy',   y_val)
    np.save(f'{out_dir}y_test.npy',  y_test)

    # save scaler so we can reuse it later (e.g. real-time inference)
    with open(f'{out_dir}scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\nAll processed data saved to '{out_dir}'")


# ─────────────────────────────────────────────
# MAIN — run everything in order
# ─────────────────────────────────────────────

if __name__ == '__main__':

    # ← change this to your actual filename if different
    
    DATA_PATH = 'data\kaggle_data.csv'

    df                                = load_data(DATA_PATH)
    df                                = clean_data(df)
    X, y                              = prepare_features(df)
    X_train, X_val, X_test, \
    y_train, y_val, y_test            = split_data(X, y)
    X_train, X_val, X_test, scaler   = scale_features(
                                            X_train, X_val, X_test)
    save_processed(X_train, X_val, X_test,
                   y_train, y_val, y_test, scaler)
    
     # save feature list for reference
    import json
    feature_cols = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Destination Port',
        'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean',
        'SYN Flag Count', 'FIN Flag Count', 'RST Flag Count',
        'PSH Flag Count', 'ACK Flag Count', 'Packet Length Mean',
        'Packet Length Std', 'Average Packet Size',
        'Init_Win_bytes_forward', 'Down/Up Ratio'
    ]
    with open('outputs/feature_cols.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Feature list saved. Total features: {len(feature_cols)}")