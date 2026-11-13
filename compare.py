"""
compare.py - Compare the trained CNN and ViT. Run this AFTER train.py has
been run for both models.

Usage:
    python compare.py

It prints a summary table and saves these figures to results/:
    curves.png      training loss, test loss and test accuracy per epoch
    per_class.png   test accuracy for each of the 10 classes
    confusion.png   confusion matrices of both models
    tsne.png        2-D t-SNE map of the features each model learned
                    (shows how well each model separates the classes)
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")                       # save figures without a screen
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from train import CLASSES, build_model, get_device, test_transform

NAMES = {"cnn": "CNN (ResNet)", "vit": "Transformer (ViT)"}


def load_results(folder):
    results = {}
    for name in ["cnn", "vit"]:
        with open(os.path.join(folder, f"{name}.json")) as f:
            results[name] = json.load(f)
    return results


# -----------------------------------------------------------------------------
# 1. Summary table
# -----------------------------------------------------------------------------

def print_table(results):
    cnn, vit = results["cnn"], results["vit"]

    def mem(r):
        m = r["peak_train_memory_mb"]
        return f"{m:.0f}" if m is not None else "n/a"

    def flops(r):
        f = r["flops_per_image"]
        return f"{f / 1e6:.0f}" if f is not None else "n/a"

    rows = [
        ("Parameters (M)", f"{cnn['params'] / 1e6:.2f}",
         f"{vit['params'] / 1e6:.2f}"),
        ("FLOPs per image (M)", flops(cnn), flops(vit)),
        ("Best test accuracy (%)", f"{100 * cnn['best_test_acc']:.2f}",
         f"{100 * vit['best_test_acc']:.2f}"),
        ("Final test accuracy (%)", f"{100 * cnn['final_test_acc']:.2f}",
         f"{100 * vit['final_test_acc']:.2f}"),
        ("Final train accuracy (%)",
         f"{100 * cnn['history'][-1]['train_acc']:.2f}",
         f"{100 * vit['history'][-1]['train_acc']:.2f}"),
        ("Total training time (min)",
         f"{cnn['total_train_time_s'] / 60:.1f}",
         f"{vit['total_train_time_s'] / 60:.1f}"),
        ("Time per epoch (s)", f"{cnn['avg_epoch_time_s']:.1f}",
         f"{vit['avg_epoch_time_s']:.1f}"),
        ("Inference (images/s)", f"{cnn['inference_images_per_s']:.0f}",
         f"{vit['inference_images_per_s']:.0f}"),
        ("Peak GPU memory (MB)", mem(cnn), mem(vit)),
    ]

    print(f"\n{'Metric':<28}{'CNN (ResNet)':>15}{'ViT':>15}")
    print("-" * 58)
    for label, a, b in rows:
        print(f"{label:<28}{a:>15}{b:>15}")
    print(f"(device: {cnn['device']})\n")


# -----------------------------------------------------------------------------
# 2. Learning curves
# -----------------------------------------------------------------------------

def plot_curves(results, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [("train_loss", "Training loss"),
              ("test_loss", "Test loss"),
              ("test_acc", "Test accuracy")]

    for ax, (key, title) in zip(axes, panels):
        for name, r in results.items():
            epochs = [h["epoch"] for h in r["history"]]
            values = [h[key] for h in r["history"]]
            ax.plot(epochs, values, label=NAMES[name])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out, "curves.png"), dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 3. Per-class accuracy
# -----------------------------------------------------------------------------

def plot_per_class(results, out):
    x = np.arange(len(CLASSES))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 4))

    for i, (name, r) in enumerate(results.items()):
        accs = [r["per_class_acc"][c] for c in CLASSES]
        ax.bar(x + (i - 0.5) * width, accs, width, label=NAMES[name])

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=30)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class test accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out, "per_class.png"), dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. Confusion matrices
# -----------------------------------------------------------------------------

def plot_confusion(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (name, r) in zip(axes, results.items()):
        cm = np.array(r["confusion_matrix"])
        cm = cm / cm.sum(axis=1, keepdims=True)    # rows sum to 1
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(NAMES[name])
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CLASSES, rotation=90)
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        # Write the value in each cell.
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")

    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion.png"), dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. Representation ability: t-SNE of the learned features
# -----------------------------------------------------------------------------

def extract_features(model, loader, device):
    """Run the model on the images and collect the 256-d feature vectors."""
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            feats.append(model.features(images.to(device)).cpu())
            labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def plot_tsne(results_dir, data_dir, out, n_images=2000):
    """
    t-SNE squeezes the 256-d features down to 2-D so they can be plotted.
    If a model has learned good representations, images of the same class
    form tight, well-separated clusters.

    We also report the silhouette score (from -1 to 1, higher = better
    separated classes) as a number to go with the picture.
    """
    device = get_device()
    test_set = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=test_transform())
    subset = Subset(test_set, range(n_images))    # first N test images
    loader = DataLoader(subset, batch_size=256, shuffle=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, name in zip(axes, ["cnn", "vit"]):
        model = build_model(name)
        weights = torch.load(os.path.join(results_dir, f"{name}.pt"),
                             map_location="cpu")
        model.load_state_dict(weights)
        model.to(device)

        feats, labels = extract_features(model, loader, device)
        score = silhouette_score(feats, labels)
        points = TSNE(n_components=2, init="pca",
                      random_state=0).fit_transform(feats)

        for c in range(10):
            mask = labels == c
            ax.scatter(points[mask, 0], points[mask, 1], s=4,
                       label=CLASSES[c])
        ax.set_title(f"{NAMES[name]}  (silhouette = {score:.3f})")
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"{NAMES[name]}: silhouette score of features = {score:.3f}")

    axes[1].legend(markerscale=4, loc="center left",
                   bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tsne.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="./results")
    parser.add_argument("--data", default="./data")
    args = parser.parse_args()

    results = load_results(args.results)
    print_table(results)
    plot_curves(results, args.results)
    plot_per_class(results, args.results)
    plot_confusion(results, args.results)
    plot_tsne(args.results, args.data, args.results)
    print(f"Figures saved in {args.results}/")


if __name__ == "__main__":
    main()