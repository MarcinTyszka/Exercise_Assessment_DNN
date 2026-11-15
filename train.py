import os
import argparse
from datetime import datetime
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from feature_extraction import create_or_load_dataset, NUM_FEATURES, EXERCISE_CLASSES
from model import PoseBaseMLP, PoseAssessmentLoss


def setup_text_logger(log_filepath):
    # Creates directory for plain text logs and writes header information
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    with open(log_filepath, "w", encoding="utf-8") as f:
        f.write(f"DNNDA Phase 2 Training Log - Started at {datetime.now().isoformat()}\n")
        f.write("-" * 80 + "\n")


def log_message(log_filepath, message):
    # Prints message to console and appends it to the searchable plain text file
    print(message)
    with open(log_filepath, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def save_checkpoint(filepath, model, optimizer, epoch, loss, hparams):
    # Saves model state, optimizer state, and hyperparameters to a checkpoint file
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "hparams": hparams
    }, filepath)


def compute_metrics(exercise_logits, form_logit, y_ex, y_form):
    # Calculates exercise classification accuracy and binary form accuracy
    pred_ex = torch.argmax(exercise_logits, dim=1)
    ex_acc = (pred_ex == y_ex).float().mean().item()

    pred_form = (torch.sigmoid(form_logit) >= 0.5).float()
    form_acc = (pred_form == y_form).float().mean().item()
    return ex_acc, form_acc


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Load dataset and split 70% train, 15% val, 15% test
    X, y_ex, y_form = create_or_load_dataset()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y_ex), torch.tensor(y_form))

    n_total = len(dataset)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, _ = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=not args.sanity_check)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # Initialize Base Model, Loss, and Optimizer (SGD or Adam without weight decay)
    model = PoseBaseMLP(
        input_dim=NUM_FEATURES,
        hidden_dim=args.hidden_dim,
        num_exercises=len(EXERCISE_CLASSES)
    ).to(device)
    criterion = PoseAssessmentLoss()

    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    writer = SummaryWriter(log_dir=f"runs/{mode_name}_{timestamp}")

    # Isolate a single batch if running the Phase 2 sanity check
    single_batch = next(iter(train_loader))

    best_loss = float("inf")
    final_ex_acc = 0.0
    final_form_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = [single_batch] if args.sanity_check else train_loader

        epoch_loss = 0.0
        epoch_ex_acc = 0.0
        epoch_form_acc = 0.0

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

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train_exercise", epoch_ex_acc, epoch)
        writer.add_scalar("Accuracy/train_form", epoch_form_acc, epoch)

        # Evaluate on validation set when running full training
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

            val_loss /= len(val_loader)
            val_ex_acc /= len(val_loader)
            val_form_acc /= len(val_loader)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val_exercise", val_ex_acc, epoch)
            writer.add_scalar("Accuracy/val_form", val_form_acc, epoch)
            val_info = f" | Val Loss: {val_loss:.4f} | Val Ex Acc: {val_ex_acc:.4f}"

        log_line = (
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Ex Acc: {epoch_ex_acc:.4f} | "
            f"Form Acc: {epoch_form_acc:.4f}{val_info}"
        )
        log_message(log_file, log_line)

        # Save checkpoint when loss improves
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_name = "sanity_best.pt" if args.sanity_check else "best_model.pt"
            save_checkpoint(f"checkpoints/{ckpt_name}", model, optimizer, epoch, best_loss, hparams)
            # Also save default checkpoint for the Streamlit demo app
            save_checkpoint("checkpoints/latest_demo_model.pt", model, optimizer, epoch, best_loss, hparams)

    # Log Hyperparameters and final metrics to TensorBoard HParams Dashboard
    writer.add_hparams(
        hparam_dict=hparams,
        metric_dict={
            "hparam/best_loss": best_loss,
            "hparam/final_exercise_acc": final_ex_acc,
            "hparam/final_form_acc": final_form_acc
        }
    )
    writer.close()
    log_message(log_file, f"Training finished. Best Loss: {best_loss:.6f}. Checkpoint saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNNDA Phase 2 Training & Sanity Check")
    parser.add_argument("--sanity_check", action="store_true", help="Overfit on a single batch")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"])
    args = parser.parse_args()

    run_training(args)