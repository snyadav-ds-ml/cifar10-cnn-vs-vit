# CNN vs Vision Transformer on CIFAR-10

Trains a small ResNet (CNN) and a small Vision Transformer (ViT) on CIFAR-10 under identical conditions and compares their accuracy, efficiency, and learned representations.

Both models are sized to ~3 M parameters so the comparison is architecture-driven, not scale-driven.

---

## Project structure

```
project3/
├── models.py        # SmallResNet and ViT architectures
├── train.py         # Training script (one model at a time)
├── compare.py       # Comparison plots and summary table
├── app.py           # Streamlit interactive demo
├── requirements.txt
└── results/         # Created automatically by train.py
    ├── cnn.pt / vit.pt        trained weights
    ├── cnn.json / vit.json    metrics and history
    ├── curves.png             learning curves
    ├── per_class.png          per-class accuracy
    ├── confusion.png          confusion matrices
    └── tsne.png               t-SNE feature maps
```

---

## Environment setup

Pick **one** of the two options below.

### Option A — conda

```bash
conda create -n cv3 python=3.11 -y
conda activate cv3
pip install -r requirements.txt
```

### Option B — venv

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU note:** the commands above install the CPU build of PyTorch.  
> For CUDA support visit https://pytorch.org/get-started/locally/ and install the
> matching `torch` + `torchvision` wheel before running `pip install -r requirements.txt`
> (or replace those two lines in requirements.txt with the CUDA variants).  
> On Apple Silicon the MPS backend is used automatically with no extra steps.

---

## Step 1 — Train the models

CIFAR-10 (~170 MB) is downloaded automatically on the first run.

```bash
# Train the CNN  (~5 min on CPU, ~1 min on GPU)
python train.py --model cnn

# Train the ViT  (~8 min on CPU, ~2 min on GPU)
python train.py --model vit
```

Both scripts save `results/<model>.pt` (weights) and `results/<model>.json` (metrics).

**Optional flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 30 | Number of training epochs |
| `--batch-size` | 128 | Mini-batch size |
| `--lr` | 1e-3 | Initial learning rate (cosine decay) |
| `--data` | `./data` | Directory for the CIFAR-10 download |
| `--out` | `./results` | Directory for weights and metrics |
| `--seed` | 0 | Random seed for reproducibility |

Example — longer run with a custom seed:

```bash
python train.py --model vit --epochs 50 --seed 42
```

---

## Step 2 — Compare the models

Requires both `results/cnn.pt` and `results/vit.pt` to exist.

```bash
python compare.py
```

Prints a summary table to the terminal and saves four figures to `results/`:

| File | Contents |
|------|----------|
| `curves.png` | Training loss, test loss, and test accuracy per epoch |
| `per_class.png` | Test accuracy for each of the 10 CIFAR-10 classes |
| `confusion.png` | Normalised confusion matrices |
| `tsne.png` | 2-D t-SNE map of the 256-d features (+ silhouette score) |

---

## Step 3 — Interactive demo (Streamlit app)

Requires both trained models.

```bash
streamlit run app.py
```

Open the URL printed in the terminal (default `http://localhost:8501`).

**What you can do in the app:**

- Upload any image and see CNN and ViT predictions side by side.
- Pick a random image straight from the CIFAR-10 test set.

> Both models were trained on 32×32 images, so every uploaded image is
> resized to 32×32 before classification. Photos with one centred object
> (a car, a dog, a ship…) work best.

**Testing with the included cat images:**

Two real cat photos are included in the project root for quick testing:

| File | Description |
|------|-------------|
| `selective-shot-red-mackerel-tabby-cat-looking-camera-with-green-background.jpg` | Tabby cat on green background |
| `grey-kitty-with-monochrome-wall-her.jpg` | Grey cat against a plain wall |

Upload either file in the app to see how both models handle a real-world cat photo.
Cat is the hardest class for both models — the CNN reaches 82% and the ViT only 52%
on this class — so these images are a good stress test.
The ViT is likely to confuse the cat with a dog; the CNN should classify it correctly.

---

## CIFAR-10 classes

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`
