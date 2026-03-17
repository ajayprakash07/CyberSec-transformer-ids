import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random
from sklearn.metrics import (precision_score, recall_score,
                              f1_score, confusion_matrix,
                              roc_auc_score, ConfusionMatrixDisplay,
                              RocCurveDisplay)
from model import get_model
from dataset import get_dataloaders


# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"All seeds set to {seed}")


CONFIG = {
    'epochs'        : 30,
    'batch_size'    : 64,
    'learning_rate' : 0.001,
    'seq_len'       : 10,
    'stride'        : 1,
    'dropout'       : 0.1,
    'patience'      : 5,
    'save_path'     : 'outputs/best_model.pt',
    'seed'          : 42
}
# ─────────────────────────────────────────────
# WEIGHTED LOSS — fixes class imbalance
# ─────────────────────────────────────────────

def get_loss_function(y_train_path, device):
    y_train = np.load(y_train_path)

    total   = len(y_train)
    benign  = (y_train == 0).sum()
    attack  = (y_train == 1).sum()

    weight_benign = total / (2 * benign)
    weight_attack = total / (2 * attack)

    weights = torch.tensor(
        [weight_benign, weight_attack],
        dtype=torch.float32
    ).to(device)

    print(f"Class weights → Benign: {weight_benign:.3f}, "
          f"Attack: {weight_attack:.3f}")

    return nn.CrossEntropyLoss(weight=weights)


# ─────────────────────────────────────────────
# ONE TRAINING EPOCH
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    
    # enable dropout — training mode
    model.train()

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    for X_batch, y_batch in loader:
        # move batch to GPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        predictions = model(X_batch)         # shape: (64, 2)

        loss = criterion(predictions, y_batch)

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # update parameters — take one step down the gradient
        optimizer.step()

        # ── Track Progress ────────────────────
        total_loss += loss.item()

        # predicted class = index of highest score
        predicted = predictions.argmax(dim=1)
        correct       += (predicted == y_batch).sum().item()
        total_samples += y_batch.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total_samples
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# ONE VALIDATION EPOCH
# ─────────────────────────────────────────────

def validate(model, loader, criterion, device):
    # disable dropout — evaluation mode
    model.eval()

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    # torch.no_grad() — don't track gradients
    # saves memory and speeds up validation
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss        = criterion(predictions, y_batch)

            total_loss    += loss.item()
            predicted      = predictions.argmax(dim=1)
            correct       += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total_samples
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# PLOT TRAINING CURVES
# ─────────────────────────────────────────────

def plot_curves(train_losses, val_losses, val_accuracies):
    """
    Plots training loss, validation loss, and validation accuracy.
    Saves to outputs/ folder.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Loss Plot ─────────────────────────────
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss',     markersize=4)
    ax1.plot(epochs, val_losses,   'r-o', label='Val Loss',       markersize=4)
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # ── Accuracy Plot ─────────────────────────
    ax2.plot(epochs, val_accuracies, 'g-o', label='Val Accuracy', markersize=4)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('outputs/training_curves.png', dpi=150)
    plt.close()
    print("Training curves saved to outputs/training_curves.png")

# ─────────────────────────────────────────────
# FULL EVALUATION METRICS
# ─────────────────────────────────────────────

def evaluate_metrics(model, loader, device, split_name='Test'):
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            # probabilities via softmax
            probs      = torch.softmax(outputs, dim=1)[:, 1]
            predicted  = outputs.argmax(dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc       = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc   = roc_auc_score(all_labels, all_probs)
    cm        = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*50}")
    print(f"  {split_name} Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    # ── Save Confusion Matrix Plot ─────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['BENIGN', 'ATTACK']
    )
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{split_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'outputs/confusion_matrix.png', dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to outputs/confusion_matrix.png")

    # ── Save ROC Curve ─────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        all_labels, all_probs,
        name='FlowTransformer',
        ax=ax
    )
    ax.set_title(f'{split_name} ROC Curve')
    plt.tight_layout()
    plt.savefig(f'outputs/roc_curve.png', dpi=150)
    plt.close()
    print(f"ROC curve saved to outputs/roc_curve.png")

    return {
        'accuracy'  : float(acc),
        'precision' : float(precision),
        'recall'    : float(recall),
        'f1'        : float(f1),
        'roc_auc'   : float(roc_auc)
    }
# ─────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────

def train():
    print("=" * 50)
    print("   CyberSec Transformer — Training")
    print("=" * 50)

    set_seeds(CONFIG['seed']) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = get_model(device)

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=CONFIG['batch_size'],
        seq_len=CONFIG['seq_len'],
        stride=CONFIG['stride']
    )

    # loss function with class weights
    criterion = get_loss_function('outputs/y_train.npy', device)

    # Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate']
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        # minimize val loss
        factor=0.5,        # multiply lr by 0.5
        patience=2,        # wait 2 epochs before reducing
    )

    # ── Training Loop ─────────────────────────
    train_losses    = []
    val_losses      = []
    val_accuracies  = []

    best_val_loss   = float('inf')  # infinity — any loss will be lower
    patience_counter = 0

    for epoch in range(1, CONFIG['epochs'] + 1):

        # train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # adjust learning rate based on val loss
        scheduler.step(val_loss)

        # record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # print progress
        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # ── Save Best Model ────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['save_path'])
            print(f"  → Best model saved (val loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  → No improvement. Patience: "
                  f"{patience_counter}/{CONFIG['patience']}")

        # ── Early Stopping ─────────────────────
        if patience_counter >= CONFIG['patience']:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # ── Final Steps ───────────────────────────
    # ── Final Steps ───────────────────────────
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    plot_curves(train_losses, val_losses, val_accuracies)

    # ── Load Best Model & Evaluate on Test ────
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(CONFIG['save_path'],
                          weights_only=True))

    _, _, test_loader = get_dataloaders(
        batch_size=CONFIG['batch_size'],
        seq_len=CONFIG['seq_len'],
        stride=CONFIG['stride']
    )

    test_metrics = evaluate_metrics(model, test_loader, device, 'Test')

    # ── Save All Metrics to JSON ───────────────
    results = {
        'train_losses'   : train_losses,
        'val_losses'     : val_losses,
        'val_accuracies' : val_accuracies,
        'test_metrics'   : test_metrics,
        'config'         : CONFIG
    }

    with open('outputs/training_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nAll metrics saved to outputs/training_metrics.json")

    return model

    # ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    train()