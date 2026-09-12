"""
Main training script.

Usage:
    python train.py                  # auto-detect device (GPU if available)
    python train.py --device cpu     # force CPU
    python train.py --device cuda    # force GPU
"""

import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

import config
from data import build_dataloaders, compute_class_weights
from model import BUSMultiTaskCNN
from losses import MultiTaskLoss
from metrics import compute_metrics_a, compute_metrics_b
from utils import set_seed, save_checkpoint, load_checkpoint, AverageMeter



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None,
                        help="Override device: 'cpu' or 'cuda'")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    loss_a_meter = AverageMeter()
    loss_b_meter = AverageMeter()

    for images, label_a, label_b in loader:
        images = images.to(device, non_blocking=True)
        label_a = label_a.to(device, non_blocking=True)
        label_b = label_b.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits_a, logits_b = model(images)
        total_loss, loss_a, loss_b = criterion(
            logits_a, logits_b, label_a, label_b
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n = images.size(0)
        loss_meter.update(total_loss.item(), n)
        loss_a_meter.update(loss_a.item(), n)
        loss_b_meter.update(loss_b.item(), n)

    return loss_meter.avg, loss_a_meter.avg, loss_b_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device, threshold_b: float = 0.5):
    model.eval()
    loss_meter = AverageMeter()
    loss_a_meter = AverageMeter()
    loss_b_meter = AverageMeter()

    all_pred_a, all_true_a = [], []
    all_prob_b, all_true_b = [], []

    for images, label_a, label_b in loader:
        images = images.to(device, non_blocking=True)
        label_a = label_a.to(device, non_blocking=True)
        label_b = label_b.to(device, non_blocking=True)

        logits_a, logits_b = model(images)
        total_loss, loss_a, loss_b = criterion(
            logits_a, logits_b, label_a, label_b
        )

        n = images.size(0)
        loss_meter.update(total_loss.item(), n)
        loss_a_meter.update(loss_a.item(), n)
        loss_b_meter.update(loss_b.item(), n)

        pred_a = logits_a.argmax(dim=1).cpu().numpy()
        all_pred_a.append(pred_a)
        all_true_a.append(label_a.cpu().numpy())

        prob_b = torch.sigmoid(logits_b).squeeze(1).cpu().numpy()
        all_prob_b.append(prob_b)
        all_true_b.append(label_b.cpu().numpy())

    y_true_a = np.concatenate(all_true_a)
    y_pred_a = np.concatenate(all_pred_a)
    y_true_b = np.concatenate(all_true_b)
    y_prob_b = np.concatenate(all_prob_b)

    metrics_a = compute_metrics_a(y_true_a, y_pred_a)
    metrics_b = compute_metrics_b(y_true_b, y_prob_b, threshold=threshold_b)

    return {
        "loss": loss_meter.avg,
        "loss_a": loss_a_meter.avg,
        "loss_b": loss_b_meter.avg,
        **metrics_a,
        **metrics_b,
    }

# ---------------------------------------------------------------------------
# Threshold tuning for Head B
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_val_probs(model, loader, device):
    """
    Run the model over a loader and return (labels_b, probs_b) as numpy arrays.
    Used to pick the best decision threshold for Head B.
    """
    model.eval()
    all_labels, all_probs = [], []
    for images, _, label_b in loader:
        images = images.to(device, non_blocking=True)
        _, logits_b = model(images)
        probs = torch.sigmoid(logits_b).squeeze(1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(label_b.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def find_best_threshold(y_true, y_prob, step: float = 0.01):
    """
    Sweep thresholds in (0, 1) and return the one that maximises F1 for Head B.
    """
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(step, 1.0, step):
        pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ----- Device -----
    if args.device is not None:
        config.DEVICE_OVERRIDE = args.device
    device = config.get_device()
    print(f"[Device] Using: {device}")

    # ----- Reproducibility -----
    set_seed(config.SEED)

    # ----- Data -----
    train_loader, val_loader, test_loader = build_dataloaders()
    print(f"[Data] Train batches: {len(train_loader)} | "
          f"Val batches: {len(val_loader)} | "
          f"Test batches: {len(test_loader)}")

    # ----- Class weights (computed once from the training split) -----
    weights_a, pos_weight_b = compute_class_weights(train_loader.dataset)
    print(f"[Data] Head A class weights: {weights_a.tolist()}")
    print(f"[Data] Head B pos_weight:   {pos_weight_b.item():.2f}")

    # ----- Model -----
    model = BUSMultiTaskCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {n_params:,}")

    # ----- Loss, optimiser, scheduler -----
    criterion = MultiTaskLoss(
        alpha=config.ALPHA,
        beta=config.BETA,
        # weight_a=None,
        weight_a=weights_a.to(device),
        pos_weight_b=pos_weight_b.to(device),
    )

    backbone_params = list(model.encoder.parameters())
    head_params = (list(model.head_a.parameters())
                + list(model.head_b.parameters()))

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": config.LR * 0.1},   # 1e-5
            {"params": head_params,     "lr": config.LR},          # 1e-4
        ],
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=config.PATIENCE_SCHEDULER,
    )

    # ----- Training loop -----
    best_val_loss = float("inf")
    best_score = -float("inf")
    epochs_without_improvement = 0
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_la, train_lb = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train loss: {train_loss:.4f} (A:{train_la:.4f} B:{train_lb:.4f}) | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"AccA: {val_metrics['acc_a']:.3f} F1A: {val_metrics['f1_macro_a']:.3f} | "
            f"AccB: {val_metrics['acc_b']:.3f} F1B: {val_metrics['f1_b']:.3f} "
            f"AUC: {val_metrics['auc_b']:.3f}"
        )

        # ----- Checkpointing -----
        score = val_metrics["acc_a"] + val_metrics["auc_b"]

        scheduler.step(score)

        if score > best_score:
            best_score = score
            epochs_without_improvement = 0

            save_checkpoint(
                args.checkpoint, model, optimizer, scheduler,
                epoch, best_score,
            )
            print(f"  -> New best score: {best_score:.4f}. Checkpoint saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.PATIENCE_EARLY_STOP:
                print(f"Early stopping after {epoch+1} epochs.")
                break

    # ----- Final evaluation (load best checkpoint) -----
    print("\n[Test] Loading best checkpoint...")
    load_checkpoint(args.checkpoint, model, device=device)

    # 1) Pick the best Head B threshold on the validation set.
    best_thr = 0.5
    print("[Threshold] Fixed at 0.50 (no tuning — using AUC as primary metric)")

    # 2) Evaluate test with the tuned threshold.
    test_metrics = validate(
        model, test_loader, criterion, device, threshold_b=best_thr
    )
    print(
        f"Test | "
        f"Loss: {test_metrics['loss']:.4f} | "
        f"AccA: {test_metrics['acc_a']:.3f} F1A: {test_metrics['f1_macro_a']:.3f} | "
        f"AccB: {test_metrics['acc_b']:.3f} F1B: {test_metrics['f1_b']:.3f} "
        f"AUC: {test_metrics['auc_b']:.3f} "
        f"(thr={best_thr:.2f})"
    )


if __name__ == "__main__":
    main()