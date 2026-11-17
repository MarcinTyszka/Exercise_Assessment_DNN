import os
import argparse
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from feature_extraction import create_or_load_dataset, NUM_FEATURES, EXERCISE_CLASSES
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


def save_checkpoint(filepath, model, optimizer, epoch, loss, hparams, feat_mean, feat_std):
    # Saves model weights, optimizer state, hyperparameters, and normalization stats
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "hparams": hparams,
        "feat_mean": feat_mean,
        "feat_std": feat_std
    }, filepath)


def compute_metrics(exercise_logits, form_logit, y_ex, y_form):
    # Computes accuracy for exercise recognition and binary form evaluation
    pred_ex = torch.argmax(exercise_logits, dim=1)
    ex_acc = (pred_ex == y_ex).float().mean().item()

    pred_form = (torch.sigmoid(form_logit) >= 0.5).float()
    form_acc = (pred_form == y_form).float().mean().item()
    return ex_acc, form_acc


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

    X, y_ex, y_form = create_or_load_dataset()

    # Shuffle indices with fixed seed before splitting 70% train / 15% val / 15% test
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    X, y_ex, y_form = X[perm], y_ex[perm], y_form[perm]

    n_total = len(X)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    X_train, y_ex_train, y_form_train = X[:n_train], y_ex[:n_train], y_form[:n_train]
    X_val, y_ex_val, y_form_val = X[n_train:n_train + n_val], y_ex[n_train:n_train + n_val], y_form[n_train:n_train + n_val]

    # Compute standardization statistics on training split
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0) + 1e-5

    X_train_norm = (X_train - feat_mean) / feat_std
    X_val_norm = (X_val - feat_mean) / feat_std

    train_set = TensorDataset(torch.tensor(X_train_norm), torch.tensor(y_ex_train), torch.tensor(y_form_train))
    val_set = TensorDataset(torch.tensor(X_val_norm), torch.tensor(y_ex_val), torch.tensor(y_form_val))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=not args.sanity_check)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

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

        if not args.sanity_check:
            model.eval()
            val_loss, val_ex_acc, val_form_acc = 0.0, 0.0, 0.0
            with torch.no_grad():
                for vx, vy_ex, vy_form in val_loader:
                    vx, vy_ex, vy_form = vx.to(device), vy_ex.to(device), vy_form.to(device)
                    v_ex_logits, v_form_logit = model(vx)
                    v_loss, _, _ = criterion(v_ex_logits, v_form_logit, vy_ex, vy_form)
                    v_ex, v_form = compute_metrics(v_ex_logits, v_form_logit, vy_ex, vy_form)
                    val_loss += v_loss.item()
                    val_ex_acc += v_ex
                    val_form_acc += v_form

            val_loss /= max(len(val_loader), 1)
            val_ex_acc /= max(len(val_loader), 1)
            val_form_acc /= max(len(val_loader), 1)

            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val_exercise", val_ex_acc, epoch)
            writer.add_scalar("Accuracy/val_form", val_form_acc, epoch)

            tracked_loss = val_loss
            val_info = f" | Val Loss: {val_loss:.4f} | Val Ex Acc: {val_ex_acc:.4f} | Val Form Acc: {val_form_acc:.4f}"

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
                save_checkpoint("checkpoints/sanity_best.pt", model, optimizer, epoch, best_tracked_loss, hparams, feat_mean, feat_std)
            else:
                save_checkpoint("checkpoints/best_model.pt", model, optimizer, epoch, best_tracked_loss, hparams, feat_mean, feat_std)
                save_checkpoint("checkpoints/latest_demo_model.pt", model, optimizer, epoch, best_tracked_loss, hparams, feat_mean, feat_std)

    writer.add_hparams(
        hparam_dict=hparams,
        metric_dict={
            "hparam/best_loss": best_tracked_loss,
            "hparam/final_exercise_acc": final_ex_acc,
            "hparam/final_form_acc": final_form_acc
        }
    )
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