"""
train.py - Train ONE model (CNN or ViT) on CIFAR-10 and save its results.

Usage:
    python train.py --model cnn
    python train.py --model vit

Both models are trained with exactly the same recipe (data augmentation,
optimizer, learning rate, epochs), so any difference in the results comes
from the architecture itself.

Outputs (in the --out folder, default "results/"):
    <model>.pt    the trained weights
    <model>.json  accuracy per epoch, timing, parameter count, FLOPs, etc.
"""

import argparse
import copy
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from torch.utils.flop_counter import FlopCounterMode
    _FLOP_COUNTER_AVAILABLE = True
except ImportError:
    _FLOP_COUNTER_AVAILABLE = False

from models import SmallResNet, ViT


# The 10 CIFAR-10 classes, in label order.
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Mean and standard deviation of CIFAR-10 pixels (per RGB channel).
# Used to normalise the images so every channel has mean 0 and std 1.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


# -----------------------------------------------------------------------------
# Setup helpers
# -----------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "vit"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", default="./results")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def get_device():
    """Use a GPU if there is one, otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_transform():
    """Test images: only convert to a tensor and normalise."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_data(data_dir, batch_size):
    """Download CIFAR-10 (the first time) and build the data loaders."""

    # Training images get simple random augmentation so the model sees a
    # slightly different version of each image every epoch.
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # random shift of up to 4 px
        transforms.RandomHorizontalFlip(),      # random left-right flip
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_set = datasets.CIFAR10(data_dir, train=True, download=True,
                                 transform=train_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=test_transform())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=2)
    return train_loader, test_loader


def build_model(name):
    if name == "cnn":
        return SmallResNet(num_classes=10)
    return ViT(num_classes=10)


# -----------------------------------------------------------------------------
# Efficiency measurements
# -----------------------------------------------------------------------------

def count_params(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model):
    """
    Floating-point operations for ONE 32x32 image (forward pass only).
    Returns None on PyTorch < 2.0 where FlopCounterMode is unavailable.
    """
    if not _FLOP_COUNTER_AVAILABLE:
        return None
    m = copy.deepcopy(model).cpu().train()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        m(torch.randn(1, 3, 32, 32))
    return counter.get_total_flops()


def sync(device):
    """GPUs run asynchronously; wait for them to finish before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_throughput(model, device, batch_size=256, n_batches=20):
    """How many images per second the model can classify (inference)."""
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    with torch.no_grad():
        for _ in range(5):            # warm-up runs (not timed)
            model(x)
        sync(device)
        start = time.perf_counter()
        for _ in range(n_batches):
            model(x)
        sync(device)
    elapsed = time.perf_counter() - start
    return batch_size * n_batches / elapsed


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training set. Returns (average loss, accuracy)."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)               # forward pass
        loss = criterion(outputs, labels)

        optimizer.zero_grad()                 # clear old gradients
        loss.backward()                       # compute new gradients
        optimizer.step()                      # update the weights

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)

    return total_loss / seen, correct / seen


def evaluate(model, loader, criterion, device):
    """
    Evaluate on the test set.
    Returns (average loss, accuracy, all predictions, all true labels).
    """
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():                     # no gradients needed here
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            seen += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    return (total_loss / seen, correct / seen,
            torch.cat(all_preds), torch.cat(all_labels))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = get_args()
    torch.manual_seed(args.seed)              # make runs repeatable
    device = get_device()
    print(f"Training {args.model.upper()} on {device}")

    train_loader, test_loader = get_data(args.data, args.batch_size)

    model = build_model(args.model)
    n_params = count_params(model)
    flops = count_flops(model)
    flops_str = f"{flops / 1e6:.1f} M" if flops is not None else "n/a"
    print(f"Parameters: {n_params / 1e6:.2f} M   FLOPs per image: {flops_str}")
    model = model.to(device)

    # Same training recipe for both models:
    #  - cross-entropy loss with a little label smoothing
    #  - AdamW optimizer (Adam + weight decay)
    #  - cosine schedule: learning rate falls smoothly to 0 by the last epoch
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = []
    train_time = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        sync(device)
        epoch_time = time.perf_counter() - start
        train_time += epoch_time

        test_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
            "epoch_time_s": epoch_time,
        })
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {train_loss:.3f} acc {train_acc:.3f} | "
              f"test loss {test_loss:.3f} acc {test_acc:.3f} | "
              f"{epoch_time:.1f}s")

    # ---- Final evaluation: per-class accuracy and confusion matrix ----
    _, final_acc, preds, labels = evaluate(
        model, test_loader, criterion, device)

    # confusion[i, j] = number of class-i images predicted as class j
    confusion = torch.zeros(10, 10, dtype=torch.long)
    for t, p in zip(labels, preds):
        confusion[t, p] += 1
    per_class_acc = (confusion.diag().float() /
                     confusion.sum(dim=1).float()).tolist()

    throughput = measure_throughput(model, device)
    peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6
                   if device.type == "cuda" else None)

    results = {
        "model": args.model,
        "device": str(device),
        "params": n_params,
        "flops_per_image": flops,
        "epochs": args.epochs,
        "final_test_acc": final_acc,
        "best_test_acc": max(h["test_acc"] for h in history),
        "total_train_time_s": train_time,
        "avg_epoch_time_s": train_time / args.epochs,
        "inference_images_per_s": throughput,
        "peak_train_memory_mb": peak_mem_mb,
        "per_class_acc": dict(zip(CLASSES, per_class_acc)),
        "confusion_matrix": confusion.tolist(),
        "history": history,
    }

    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out, f"{args.model}.pt"))
    with open(os.path.join(args.out, f"{args.model}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal test accuracy: {final_acc:.4f}")
    print(f"Saved results to {args.out}/{args.model}.json")


if __name__ == "__main__":
    main()