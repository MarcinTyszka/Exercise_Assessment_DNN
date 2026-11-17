import os
import argparse
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from feature_extraction import create_or_load_dataset, NUM_FEATURES, EXERCISE_CLASSES
from build_dataset import balance_dataset_per_exercise
from model import PoseBaseMLP, PoseAssessmentLoss


def get_safe_device():
    # Selects CUDA if supported by the current PyTorch installation, otherwise CPU
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            print("CUDA compute capability mismatch detected, falling back to CPU.")
    return torch.device("cpu")


def setup_text_logger(log_filepath):
    # Initializes searchable plain-text training log file
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    with open(log_filepath, "w", encoding="utf-8") as f:
        f.write(f"DNNDA Phase 2 Training Log - Started at {datetime.now().isoformat()}\n")
        f.write("-" * 80 + "\n")


def log_message(log_filepath, message):
    # Prints message to console and appends it to the plain-text log
    print(message)
    with open(log_filepath, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def save_checkpoint(filepath, model, optimizer, epoch, loss, hparams, feat_mean, feat_std, test_indices=None):
    # Saves model weights, optimizer state, hyperparameters, normalization stats, and held-out test indices
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "hparams": hparams,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "test_indices": test_indices
    }, filepath)


def load_dataset_with_groups(npz_path="data/pose_dataset.npz"):
    # Loads features, targets, and video_ids (falls back to frame-level IDs if absent)
    X, y_ex, y_form = create_or_load_dataset(npz_path)
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        if "video_ids" in data and len(data["video_ids"]) == len(X):
            return X, y_ex, y_form, data["video_ids"]
    return X, y_ex, y_form, np.arange(len(X), dtype=np.int64)


def stratified_group_split(y_ex, y_form, video_ids, seed=42):
    # Splits dataset 70% train / 15% val / 15% test grouped by video_id per (exercise, form) stratum
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    warnings = []

    for ex_idx, ex_name in enumerate(EXERCISE_CLASSES):
        for form_val, form_name in [(1.0, "correct"), (0.0, "error")]:
            stratum_mask = (y_ex == ex_idx) & (y_form == form_val)
            stratum_indices = np.where(stratum_mask)[0]
            if len(stratum_indices) == 0:
                continue

            unique_vids = np.unique(video_ids[stratum_indices])
            unique_vids = rng.permutation(unique_vids)
            n_vids = len(unique_vids)

            if n_vids >= 3:
                n_train_v = max(1, int(round(0.70 * n_vids)))
                n_val_v = max(1, int(round(0.15 * n_vids)))
                if n_train_v + n_val_v >= n_vids:
                    n_train_v = max(1, n_vids - 2)
                    n_val_v = 1

                train_vids = set(unique_vids[:n_train_v])
                val_vids = set(unique_vids[n_train_v:n_train_v + n_val_v])
                test_vids = set(unique_vids[n_train_v + n_val_v:])

                for idx in stratum_indices:
                    vid = video_ids[idx]
                    if vid in train_vids:
                        train_idx.append(idx)
                    elif vid in val_vids:
                        val_idx.append(idx)
                    else:
                        test_idx.append(idx)
            elif n_vids == 2:
                warnings.append(
                    f"Only 2 videos for {ex_name}/{form_name}: assigned 1 to train, 1 to val (0 to test)."
                )
                for idx in stratum_indices:
                    if video_ids[idx] == unique_vids[0]:
                        train_idx.append(idx)
                    else:
                        val_idx.append(idx)
            else:
                warnings.append(
                    f"Only 1 video for {ex_name}/{form_name}: falling back to frame split for this bucket."
                )
                perm_frames = rng.permutation(stratum_indices)
                nt = max(1, int(0.70 * len(perm_frames)))
                nv = max(1, int(0.15 * len(perm_frames)))
                train_idx.extend(perm_frames[:nt])
                val_idx.extend(perm_frames[nt:nt + nv])
                test_idx.extend(perm_frames[nt + nv:])

    train_idx = rng.permutation(np.array(train_idx, dtype=np.int64))
    val_idx = np.array(sorted(val_idx), dtype=np.int64)
    test_idx = np.array(sorted(test_idx), dtype=np.int64)
    return train_idx, val_idx, test_idx, warnings


def compute_metrics(exercise_logits, form_logit, y_ex, y_form):
    # Computes accuracy for exercise recognition and binary form evaluation
    pred_ex = torch.argmax(exercise_logits, dim=1)
    ex_acc = (pred_ex == y_ex).float().mean().item()

    pred_form = (torch.sigmoid(form_logit) >= 0.5).float()
    form_acc = (pred_form == y_form).float().mean().item()
    return ex_acc, form_acc


def evaluate_split(model, loader, criterion, device):
    # Evaluates loss, accuracy, and binary form precision/recall/F1 across a DataLoader
    model.eval()
    total_loss = 0.0
    total_samples = 0
    ex_correct = 0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for bx, by_ex, by_form in loader:
            bx, by_ex, by_form = bx.to(device), by_ex.to(device), by_form.to(device)
            ex_logits, form_logit = model(bx)
            loss, _, _ = criterion(ex_logits, form_logit, by_ex, by_form)

            batch_size = bx.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            pred_ex = torch.argmax(ex_logits, dim=1)
            ex_correct += (pred_ex == by_ex).sum().item()

            pred_form = (torch.sigmoid(form_logit) >= 0.5).float()
            tp += ((pred_form == 1.0) & (by_form == 1.0)).sum().item()
            fp += ((pred_form == 1.0) & (by_form == 0.0)).sum().item()
            tn += ((pred_form == 0.0) & (by_form == 0.0)).sum().item()
            fn += ((pred_form == 0.0) & (by_form == 1.0)).sum().item()

    denom = max(total_samples, 1)
    avg_loss = total_loss / denom
    ex_acc = ex_correct / denom
    form_acc = (tp + tn) / denom
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-8)

    return {
        "loss": avg_loss,
        "ex_acc": ex_acc,
        "form_acc": form_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def run_training(args):
    device = get_safe_device()
    mode_name = "sanity_check" if args.sanity_check else "full_train"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = f"logs/{mode_name}_{timestamp}.txt"
    setup_text_logger(log_file)

    hparams = {
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "sanity_check": args.sanity_check
    }
    log_message(log_file, f"Device: {device} | Hyperparameters: {hparams}")

    X, y_ex, y_form, video_ids = load_dataset_with_groups()
    train_idx, val_idx, test_idx, split_warnings = stratified_group_split(y_ex, y_form, video_ids, seed=42)

    for w in split_warnings:
        log_message(log_file, f"[Split Warning] {w}")

    X_train_raw, y_ex_train_raw, y_form_train_raw = X[train_idx], y_ex[train_idx], y_form[train_idx]
    X_val, y_ex_val, y_form_val = X[val_idx], y_ex[val_idx], y_form[val_idx]
    X_test, y_ex_test, y_form_test = X[test_idx], y_ex[test_idx], y_form[test_idx]

    # Balance correct/error classes strictly inside the training split to prevent data leakage
    X_train, y_ex_train, y_form_train = balance_dataset_per_exercise(
        X_train_raw, y_ex_train_raw, y_form_train_raw, seed=42
    )

    log_message(
        log_file,
        f"Split sizes -> Train: {len(X_train)} (raw {len(X_train_raw)}) | "
        f"Val: {len(X_val)} | Test: {len(X_test)}"
    )

    # Compute standardization statistics exclusively on the training split
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0) + 1e-5

    X_train_norm = (X_train - feat_mean) / feat_std
    X_val_norm = (X_val - feat_mean) / feat_std

    train_set = TensorDataset(torch.tensor(X_train_norm), torch.tensor(y_ex_train), torch.tensor(y_form_train))
    val_set = TensorDataset(torch.tensor(X_val_norm), torch.tensor(y_ex_val), torch.tensor(y_form_val))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=not args.sanity_check)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    test_loader = None
    if len(X_test) > 0:
        X_test_norm = (X_test - feat_mean) / feat_std
        test_set = TensorDataset(torch.tensor(X_test_norm), torch.tensor(y_ex_test), torch.tensor(y_form_test))
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = PoseBaseMLP(
        input_dim=NUM_FEATURES,
        hidden_dim=args.hidden_dim,
        num_exercises=len(EXERCISE_CLASSES)
    ).to(device)

    num_pos = max(float((y_form_train == 1.0).sum()), 1.0)
    num_neg = max(float((y_form_train == 0.0).sum()), 1.0)
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    criterion = PoseAssessmentLoss(pos_weight=pos_weight)

    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    writer = SummaryWriter(log_dir=f"runs/{mode_name}_{timestamp}")
    single_batch = next(iter(train_loader))

    best_tracked_loss = float("inf")
    final_ex_acc = 0.0
    final_form_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = [single_batch] if args.sanity_check else train_loader

        epoch_loss, epoch_ex_acc, epoch_form_acc = 0.0, 0.0, 0.0

        for bx, by_ex, by_form in batches:
            bx, by_ex, by_form = bx.to(device), by_ex.to(device), by_form.to(device)

            optimizer.zero_grad()
            ex_logits, form_logit = model(bx)
            loss, _, _ = criterion(ex_logits, form_logit, by_ex, by_form)
            loss.backward()
            optimizer.step()

            ex_acc, form_acc = compute_metrics(ex_logits, form_logit, by_ex, by_form)
            epoch_loss += loss.item()
            epoch_ex_acc += ex_acc
            epoch_form_acc += form_acc

        num_batches = len(batches)
        epoch_loss /= num_batches
        epoch_ex_acc /= num_batches
        epoch_form_acc /= num_batches

        final_ex_acc = epoch_ex_acc
        final_form_acc = epoch_form_acc

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train_exercise", epoch_ex_acc, epoch)
        writer.add_scalar("Accuracy/train_form", epoch_form_acc, epoch)

        tracked_loss = epoch_loss
        val_info = ""

        if not args.sanity_check and len(val_loader) > 0:
            val_metrics = evaluate_split(model, val_loader, criterion, device)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/val_exercise", val_metrics["ex_acc"], epoch)
            writer.add_scalar("Accuracy/val_form", val_metrics["form_acc"], epoch)
            writer.add_scalar("F1/val_form", val_metrics["f1"], epoch)

            tracked_loss = val_metrics["loss"]
            val_info = (
                f" | Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Ex Acc: {val_metrics['ex_acc']:.4f} | "
                f"Val Form Acc: {val_metrics['form_acc']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

        log_line = (
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Ex Acc: {epoch_ex_acc:.4f} | "
            f"Form Acc: {epoch_form_acc:.4f}{val_info}"
        )
        log_message(log_file, log_line)

        if tracked_loss < best_tracked_loss:
            best_tracked_loss = tracked_loss
            if args.sanity_check:
                save_checkpoint(
                    "checkpoints/sanity_best.pt", model, optimizer, epoch,
                    best_tracked_loss, hparams, feat_mean, feat_std, test_indices=test_idx
                )
            else:
                save_checkpoint(
                    "checkpoints/best_model.pt", model, optimizer, epoch,
                    best_tracked_loss, hparams, feat_mean, feat_std, test_indices=test_idx
                )
                save_checkpoint(
                    "checkpoints/latest_demo_model.pt", model, optimizer, epoch,
                    best_tracked_loss, hparams, feat_mean, feat_std, test_indices=test_idx
                )

    hparam_metrics = {
        "hparam/best_loss": best_tracked_loss,
        "hparam/final_exercise_acc": final_ex_acc,
        "hparam/final_form_acc": final_form_acc
    }

    if not args.sanity_check and test_loader is not None and os.path.exists("checkpoints/best_model.pt"):
        best_ckpt = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_metrics = evaluate_split(model, test_loader, criterion, device)

        writer.add_scalar("Loss/test", test_metrics["loss"], best_ckpt["epoch"])
        writer.add_scalar("Accuracy/test_exercise", test_metrics["ex_acc"], best_ckpt["epoch"])
        writer.add_scalar("Accuracy/test_form", test_metrics["form_acc"], best_ckpt["epoch"])
        writer.add_scalar("F1/test_form", test_metrics["f1"], best_ckpt["epoch"])

        hparam_metrics["hparam/test_exercise_acc"] = test_metrics["ex_acc"]
        hparam_metrics["hparam/test_form_acc"] = test_metrics["form_acc"]
        hparam_metrics["hparam/test_form_f1"] = test_metrics["f1"]

        log_message(log_file, "-" * 80)
        log_message(
            log_file,
            f"Held-Out Test Evaluation (Best Epoch {best_ckpt['epoch']}): "
            f"Test Loss: {test_metrics['loss']:.4f} | "
            f"Test Ex Acc: {test_metrics['ex_acc']:.4f} | "
            f"Test Form Acc: {test_metrics['form_acc']:.4f} | "
            f"Test Form F1: {test_metrics['f1']:.4f} "
            f"(P: {test_metrics['precision']:.4f}, R: {test_metrics['recall']:.4f})"
        )

    writer.add_hparams(hparam_dict=hparams, metric_dict=hparam_metrics)
    writer.close()
    log_message(log_file, f"Training finished. Best Loss: {best_tracked_loss:.6f}. Checkpoint saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNNDA Phase 2 Training & Sanity Check")
    parser.add_argument("--sanity_check", action="store_true", help="Overfit on a single batch")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"])
    args = parser.parse_args()

    run_training(args)