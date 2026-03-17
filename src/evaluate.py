import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay,
    classification_report
)
from model import get_model
from dataset import get_dataloaders


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_best_model(device, model_path='outputs/best_model.pt'):
    model = get_model(device)
    model.load_state_dict(torch.load(
        model_path,
        map_location=device,
        weights_only=True
    ))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


# ─────────────────────────────────────────────
# RUN INFERENCE ON TEST SET
# ─────────────────────────────────────────────

def get_predictions(model, test_loader, device):
    """
    Runs model on entire test set.
    Returns true labels, predicted labels, attack probabilities.
    """
    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            # softmax converts raw scores to probabilities
            probs     = torch.softmax(outputs, dim=1)[:, 1]
            predicted = outputs.argmax(dim=1)

            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


# ─────────────────────────────────────────────
# PRINT METRICS
# ─────────────────────────────────────────────

def print_metrics(labels, preds, probs):
    """
    Prints all required assignment metrics cleanly.
    """
    acc       = (preds == labels).mean()
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)
    roc_auc   = roc_auc_score(labels, probs)
    cm        = confusion_matrix(labels, preds)

    tn, fp, fn, tp = cm.ravel()

    print("\n" + "="*55)
    print("   FINAL TEST SET EVALUATION — FlowTransformer")
    print("="*55)
    print(f"\n  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  Recall    : {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"  ROC-AUC   : {roc_auc:.4f}  ({roc_auc*100:.2f}%)")

    print(f"\n  Confusion Matrix:")
    print(f"  ┌─────────────┬─────────────┐")
    print(f"  │ TN = {tn:>6,} │ FP = {fp:>6,} │  ← Predicted Benign")
    print(f"  ├─────────────┼─────────────┤")
    print(f"  │ FN = {fn:>6,} │ TP = {tp:>6,} │  ← Predicted Attack")
    print(f"  └─────────────┴─────────────┘")
    print(f"    Actual Benign  Actual Attack")

    print(f"\n  Detailed Classification Report:")
    print(classification_report(
        labels, preds,
        target_names=['BENIGN', 'ATTACK']
    ))

    # ── Interpretation ────────────────────────
    print("="*55)
    print("   INTERPRETATION")
    print("="*55)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n  False Positive Rate : {fpr*100:.2f}%")
    print(f"  → {fp:,} benign flows incorrectly flagged as attacks")
    print(f"  → Causes unnecessary alerts for security analysts")

    print(f"\n  False Negative Rate : {fnr*100:.2f}%")
    print(f"  → {fn:,} real attacks went undetected")
    print(f"  → Most critical metric in cybersecurity")

    if fnr < 0.01:
        print(f"  → Excellent: model misses less than 1% of attacks ✅")
    elif fnr < 0.05:
        print(f"  → Good: model misses less than 5% of attacks ✅")
    else:
        print(f"  → Needs improvement: too many missed attacks ⚠️")

    if roc_auc > 0.99:
        print(f"\n  ROC-AUC > 0.99 → near perfect discrimination ✅")
    elif roc_auc > 0.95:
        print(f"\n  ROC-AUC > 0.95 → excellent discrimination ✅")

    print(f"\n  Note: High accuracy on CICIDS2017 is expected.")
    print(f"  This is a controlled lab dataset with clean,")
    print(f"  well-separated attack signatures. Real-world")
    print(f"  performance would be lower due to noisy traffic.")

    return {
        'accuracy'  : float(acc),
        'precision' : float(precision),
        'recall'    : float(recall),
        'f1'        : float(f1),
        'roc_auc'   : float(roc_auc),
        'TN'        : int(tn),
        'FP'        : int(fp),
        'FN'        : int(fn),
        'TP'        : int(tp),
        'fpr'       : float(fpr),
        'fnr'       : float(fnr)
    }


# ─────────────────────────────────────────────
# SAVE ALL PLOTS
# ─────────────────────────────────────────────

def save_results(metrics, out_dir='outputs/'):
    path = f'{out_dir}evaluation_results.json'
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {path}")


# ─────────────────────────────────────────────
# SAVE RESULTS JSON
# ─────────────────────────────────────────────

def save_plots(labels, preds, probs, out_dir='outputs/'):
    """
    Saves confusion matrix and ROC curve as clean plots.
    """

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # ── Confusion Matrix ──────────────────────
    ax1 = fig.add_subplot(gs[0])
    cm  = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['BENIGN', 'ATTACK']
    )
    disp.plot(ax=ax1, colorbar=True, cmap='Blues')
    ax1.set_title('Confusion Matrix — Test Set', fontsize=13)

    # ── ROC Curve ─────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    RocCurveDisplay.from_predictions(
        labels, probs,
        name='FlowTransformer',
        ax=ax2,
        color='darkorange'
    )
    ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax2.set_title('ROC Curve — Test Set', fontsize=13)
    ax2.legend(loc='lower right')

    plt.suptitle(
        'FlowTransformer — Network Intrusion Detection\n'
        'CICIDS2017 Dataset',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(f'{out_dir}evaluation_plots.png', dpi=150)
    plt.close()
    print(f"\nEvaluation plots saved to {out_dir}evaluation_plots.png")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':

    print("="*55)
    print("   FlowTransformer — Evaluation Script")
    print("="*55)

    # setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # load model
    model = load_best_model(device)

    # load test data
    print("\nLoading test data...")
    _, _, test_loader = get_dataloaders(batch_size=64)

    # get predictions
    print("Running inference on test set...")
    labels, preds, probs = get_predictions(model, test_loader, device)
    print(f"Total test sequences evaluated: {len(labels):,}")

    # print all metrics
    metrics = print_metrics(labels, preds, probs)

    # save plots
    save_plots(labels, preds, probs)

    # save results
    save_results(metrics)

    print("\n✅ Evaluation complete!")
    print("   Check outputs/ for:")
    print("   → evaluation_plots.png  (confusion matrix + ROC curve)")
    print("   → evaluation_results.json (all metrics)")