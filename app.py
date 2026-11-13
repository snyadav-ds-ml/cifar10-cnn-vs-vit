"""
app.py - A small web app to try both trained models on your own images.

Upload an image (or pick a random CIFAR-10 test image) and see the CNN's and
the ViT's predictions side by side.

Run it AFTER training both models (it needs results/cnn.pt and results/vit.pt):

    streamlit run app.py

Then open the link it prints (usually http://localhost:8501).

Note: both models were trained on tiny 32x32 CIFAR-10 images, so every image
is shrunk to 32x32 before it is classified. Photos that look like CIFAR-10
(one centred object: a car, a dog, a ship...) work best.
"""

import os
import random
import time

import streamlit as st
import torch
from PIL import Image
from torchvision import datasets, transforms

from train import CLASSES, MEAN, STD, build_model

RESULTS_DIR = "results"
DATA_DIR = "data"
NAMES = {"cnn": "CNN (ResNet)", "vit": "Transformer (ViT)"}


# -----------------------------------------------------------------------------
# Loading the models (cached so it only happens once, not on every click)
# -----------------------------------------------------------------------------

@st.cache_resource
def load_models():
    """Load both trained models onto the CPU in evaluation mode."""
    models = {}
    for name in ["cnn", "vit"]:
        path = os.path.join(RESULTS_DIR, f"{name}.pt")
        if not os.path.exists(path):
            return None, f"Missing {path}. Train both models first (see README)."
        model = build_model(name)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()                      # turn off dropout, use BN averages
        models[name] = model
    return models, None


# -----------------------------------------------------------------------------
# Image preprocessing: exactly the same as the test images during training
# -----------------------------------------------------------------------------

# 1. Shrink the shorter side to 32 px, 2. cut out the centre 32x32 square.
shrink = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
])

# 3. Convert to a tensor and normalise with the CIFAR-10 mean / std.
to_model_input = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def predict(model, image_32):
    """
    Classify one 32x32 PIL image.
    Returns (probabilities for the 10 classes, time taken in milliseconds).
    """
    x = to_model_input(image_32).unsqueeze(0)     # add batch dim: (1, 3, 32, 32)
    with torch.no_grad():
        start = time.perf_counter()
        logits = model(x)
        elapsed_ms = (time.perf_counter() - start) * 1000
    probs = torch.softmax(logits, dim=1)[0]       # scores -> probabilities
    return probs, elapsed_ms


@st.cache_resource
def load_cifar_test():
    """Load the CIFAR-10 test set once and cache it."""
    return datasets.CIFAR10(DATA_DIR, train=False, download=True)


def random_cifar_image():
    """Pick a random image (and its true label) from the CIFAR-10 test set."""
    test_set = load_cifar_test()
    index = random.randrange(len(test_set))
    image, label = test_set[index]                # a 32x32 PIL image + int
    return image, CLASSES[label]


def show_prediction(column, name, probs, elapsed_ms, top_k=5):
    """Draw one model's result inside a Streamlit column."""
    with column:
        st.subheader(NAMES[name])
        best = int(probs.argmax())
        st.metric("Prediction", CLASSES[best], f"{100 * probs[best]:.1f}% confident",
                  delta_color="off")

        st.write(f"**Top {top_k} classes**")
        top_probs, top_idx = probs.topk(top_k)
        for p, i in zip(top_probs.tolist(), top_idx.tolist()):
            st.write(f"{CLASSES[i]} — {100 * p:.1f}%")
            st.progress(p)

        st.caption(f"Inference time: {elapsed_ms:.1f} ms on CPU")


# -----------------------------------------------------------------------------
# The page
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="CNN vs ViT", layout="wide")
    st.title("CNN vs. Vision Transformer — CIFAR-10 classifier")
    st.write("Upload an image, or pick a random image from the CIFAR-10 **test set** "
             "(the same 10,000 images used to measure accuracy). "
             "Both models classify it into one of: " + ", ".join(CLASSES) + ".")

    models, error = load_models()
    if error:
        st.error(error)
        st.stop()

    # ---- Choose the input image ----
    uploaded = st.file_uploader("Upload your own image", type=["png", "jpg", "jpeg", "webp"])
    st.write("**— or —**")
    use_random = st.button(
        "Pick a random CIFAR-10 test image",
        help="Loads a random image from the 10,000 CIFAR-10 test images. "
             "The true class label is shown so you can check whether each model is correct.",
    )

    # Remember the random image between clicks (Streamlit reruns the script
    # from the top every time something changes).
    if use_random:
        try:
            st.session_state["random"] = random_cifar_image()
        except Exception as e:
            st.error(f"Could not load CIFAR-10 from ./{DATA_DIR}: {e}")

    true_label = None
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
    elif "random" in st.session_state:
        image, true_label = st.session_state["random"]
    else:
        st.info("Waiting for an image...")
        st.stop()

    image_32 = shrink(image)

    # ---- Show the input ----
    left, right = st.columns(2)
    with left:
        st.write("**Your image**")
        st.image(image, width=256)
    with right:
        st.write("**What the models see (32×32, enlarged)**")
        # Enlarge with NEAREST so the pixels stay sharp instead of blurry.
        st.image(image_32.resize((256, 256), Image.NEAREST), width=256)
    if true_label:
        st.info(f"True label (from CIFAR-10 test set): **{true_label}**")

    st.divider()

    # ---- Run both models and show the results side by side ----
    results = {name: predict(model, image_32) for name, model in models.items()}

    col_cnn, col_vit = st.columns(2)
    show_prediction(col_cnn, "cnn", *results["cnn"])
    show_prediction(col_vit, "vit", *results["vit"])

    # ---- Do the two models agree? ----
    pred_cnn = CLASSES[int(results["cnn"][0].argmax())]
    pred_vit = CLASSES[int(results["vit"][0].argmax())]
    st.divider()
    if pred_cnn == pred_vit:
        st.success(f"Both models agree: **{pred_cnn}**")
    else:
        st.warning(f"The models disagree: CNN says **{pred_cnn}**, "
                   f"ViT says **{pred_vit}**")
    if true_label:
        st.write(f"CNN correct: {'yes' if pred_cnn == true_label else 'no'} · "
                 f"ViT correct: {'yes' if pred_vit == true_label else 'no'}")


if __name__ == "__main__":
    main()