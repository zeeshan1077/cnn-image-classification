"""
CNN fundamentals demo (no deep learning library needed, only numpy).

Shows, on a real image from the dataset:
  1. How an image is stored as numbers (pixel matrix)
  2. What a convolution with a filter/kernel does (feature maps)
  3. Effect of padding, stride and pooling on the output size

Run:  python src/conv_demo.py
Output: outputs/plots/demo_*.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PLOT_DIR = ROOT / "outputs" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# Sample image: dataset ki pehli image (agar CSV maujood ho)
# ---------------------------------------------------------------
def load_sample_image():
    files = [f for f in DATA_DIR.glob("*.csv")
             if "train" in f.name.lower() and "sample" not in f.name.lower()]
    if files:
        df = pd.read_csv(sorted(files)[0], nrows=5)
        if df.shape[1] == 785:
            return df.iloc[0, 1:].to_numpy(dtype=np.float32).reshape(28, 28)
    # CSV nahi mili to ek simple nakli image (sirf demo ke liye)
    img = np.zeros((28, 28), dtype=np.float32)
    img[6:22, 8:20] = 200
    img[6:12, 4:24] = 200
    return img


# ---------------------------------------------------------------
# Convolution / pooling ko haath se implement kiya (taake samajh aaye)
# ---------------------------------------------------------------
def conv2d(img, kernel, stride=1, padding=0):
    """Ek channel ki image par ek kernel slide karta hai.
    (Deep learning libraries technically cross-correlation karti hain,
    jo yahi cheez hai - kernel flip nahi hota.)"""
    if padding > 0:
        img = np.pad(img, padding, mode="constant", constant_values=0)
    kh, kw = kernel.shape
    h, w = img.shape
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    out = np.zeros((oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            patch = img[i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def pool2d(img, size=2, stride=2, mode="max"):
    h, w = img.shape
    oh = (h - size) // stride + 1
    ow = (w - size) // stride + 1
    out = np.zeros((oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            patch = img[i * stride:i * stride + size, j * stride:j * stride + size]
            out[i, j] = patch.max() if mode == "max" else patch.mean()
    return out


def out_size(w, k, p, s):
    """Formula: (W - K + 2P) / S + 1"""
    return (w - k + 2 * p) // s + 1


img = load_sample_image()
print("Sample image shape:", img.shape, "| pixel range:", img.min(), "to", img.max())

# ---------------------------------------------------------------
# 1) Image = numbers ka matrix
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Image (28x28 grayscale)")
axes[0].axis("off")
crop = img[10:18, 10:18]
axes[1].imshow(crop, cmap="gray", vmin=0, vmax=255)
for i in range(crop.shape[0]):
    for j in range(crop.shape[1]):
        v = int(crop[i, j])
        axes[1].text(j, i, str(v), ha="center", va="center", fontsize=8,
                     color="white" if v < 128 else "black")
axes[1].set_title("Same image: an 8x8 crop shown as pixel values (0=black, 255=white)")
axes[1].set_xticks([]); axes[1].set_yticks([])
plt.tight_layout()
plt.savefig(PLOT_DIR / "demo_1_image_as_numbers.png", dpi=110)
plt.close()

# ---------------------------------------------------------------
# 2) Kernels (filters) -> feature maps
# ---------------------------------------------------------------
kernels = {
    "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    "Blur (box)": np.ones((3, 3), dtype=np.float32) / 9.0,
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "Sobel-X (vertical edges)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "Sobel-Y (horizontal edges)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
}
fig, axes = plt.subplots(1, len(kernels) + 1, figsize=(3 * (len(kernels) + 1), 3.4))
axes[0].imshow(img, cmap="gray"); axes[0].set_title("Input"); axes[0].axis("off")
for ax, (name, k) in zip(axes[1:], kernels.items()):
    fmap = conv2d(img, k, stride=1, padding=1)
    ax.imshow(fmap, cmap="gray")
    ax.set_title(name, fontsize=9)
    ax.axis("off")
    print(f"\nKernel: {name}\n{k}")
plt.suptitle("Convolution with different 3x3 kernels -> different feature maps", y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / "demo_2_kernels_feature_maps.png", dpi=110, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 3) Padding, stride, pooling
# ---------------------------------------------------------------
k = kernels["Sobel-X (vertical edges)"]
base = conv2d(img, k, stride=1, padding=0)
same = conv2d(img, k, stride=1, padding=1)
strided = conv2d(img, k, stride=2, padding=1)
mp = pool2d(same, 2, 2, "max")
ap = pool2d(same, 2, 2, "avg")

panels = [
    (base, f"No padding, stride 1\noutput {base.shape[0]}x{base.shape[1]}"),
    (same, f"Padding 1, stride 1 ('same')\noutput {same.shape[0]}x{same.shape[1]}"),
    (strided, f"Padding 1, stride 2\noutput {strided.shape[0]}x{strided.shape[1]}"),
    (mp, f"Max-pool 2x2 of 'same' map\noutput {mp.shape[0]}x{mp.shape[1]}"),
    (ap, f"Avg-pool 2x2 of 'same' map\noutput {ap.shape[0]}x{ap.shape[1]}"),
]
fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
for ax, (m, t) in zip(axes, panels):
    ax.imshow(m, cmap="gray")
    ax.set_title(t, fontsize=9)
    ax.axis("off")
plt.suptitle("Padding, stride and pooling change the output size", y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / "demo_3_stride_padding_pooling.png", dpi=110, bbox_inches="tight")
plt.close()

print("\nOutput size formula: (W - K + 2P) / S + 1   with W=28, K=3")
print(f"{'Padding':>8} {'Stride':>7} {'Output':>8}")
for p, s in [(0, 1), (1, 1), (1, 2), (0, 2)]:
    o = out_size(28, 3, p, s)
    print(f"{p:>8} {s:>7} {o:>5}x{o}")

print("\nDone. Plots saved in outputs/plots/ (demo_1..3)")
