"""
Custom CNN image classification on Fashion-MNIST (Kaggle CSV format), PyTorch.

Run:
    python src/train.py --quick     (chota test run, ~2 min - pehle yeh chalao)
    python src/train.py             (full run)
    python src/train.py --epochs 4  (agar laptop slow ho)

Outputs (outputs/ folder):
    results.json, best_cnn.pt, plots/*.png
"""
import argparse
import json
import math
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report

# ------------------------------------------------------------------
# Paths (kahin se bhi chalao, ye folder khud dhoondh lega)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots"
for d in (DATA_DIR, OUT_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="small fast test run")
parser.add_argument("--epochs", type=int, default=None)
args = parser.parse_args()

SEED = 42
BATCH_SIZE = 128
EPOCHS = args.epochs if args.epochs else (2 if args.quick else 6)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE} | Epochs per experiment: {EPOCHS} | Quick mode: {args.quick}")

FASHION_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# ==================================================================
# STEP 1: DATA LOAD (Kaggle CSV: pehla column = label, baaki 784 = pixels)
# ==================================================================
def find_csv(keyword):
    cands = [f for f in DATA_DIR.glob("*.csv")
             if keyword in f.name.lower() and "sample" not in f.name.lower()]
    if not cands:
        raise SystemExit(
            f"ERROR: data/ folder mein '{keyword}' wali CSV nahi mili.\n"
            f"Kaggle 'Fashion MNIST' dataset download karke fashion-mnist_train.csv aur "
            f"fashion-mnist_test.csv ko data/ folder mein rakho.\nData folder: {DATA_DIR}")
    return sorted(cands)[0]


def load_csv(path):
    df = pd.read_csv(path)
    if df.shape[1] != 785:
        raise SystemExit(f"ERROR: {path.name} mein {df.shape[1]} columns hain, 785 hone chahiye "
                         f"(1 label + 784 pixels). Ye Fashion-MNIST CSV nahi lag rahi.")
    y = torch.tensor(df.iloc[:, 0].to_numpy(dtype=np.int64))
    X = torch.tensor(df.iloc[:, 1:].to_numpy(dtype=np.float32)) / 255.0   # 0-255 -> 0-1
    return X.view(-1, 1, 28, 28), y


train_path, test_path = find_csv("train"), find_csv("test")
print(f"Loading {train_path.name} and {test_path.name} ...")
X_all, y_all = load_csv(train_path)
X_test, y_test = load_csv(test_path)

# Train / Validation split (test set alag rehta hai, sirf aakhir mein use hoga)
g = torch.Generator().manual_seed(SEED)
perm = torch.randperm(len(X_all), generator=g)
if args.quick:
    perm = perm[:min(10000, len(perm))]
n_val = int(0.1 * len(perm)) if not args.quick else int(0.2 * len(perm))
val_idx, train_idx = perm[:n_val], perm[n_val:]
X_train, y_train = X_all[train_idx].to(DEVICE), y_all[train_idx].to(DEVICE)
X_val, y_val = X_all[val_idx].to(DEVICE), y_all[val_idx].to(DEVICE)
X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

N_CLASSES = int(y_all.max().item()) + 1
CLASS_NAMES = FASHION_NAMES if N_CLASSES == 10 else [str(i) for i in range(N_CLASSES)]

# Normalization: training data ka mean/std (test data se kuch nahi liya -> no leakage)
MEAN, STD = X_train.mean().item(), X_train.std().item()


def norm(x):
    return (x - MEAN) / STD


# ==================================================================
# STEP 2: DATA AUGMENTATION (random rotation + shift + horizontal flip)
# ==================================================================
def augment(x, max_rot_deg=10, max_shift_px=2):
    B, _, H, W = x.shape
    dev = x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(max_rot_deg)
    tx = (torch.rand(B, device=dev) * 2 - 1) * (2.0 * max_shift_px / W)
    ty = (torch.rand(B, device=dev) * 2 - 1) * (2.0 * max_shift_px / H)
    flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1     # +1 ya -1
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = cos * flip
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin * flip
    theta[:, 1, 1] = cos
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


# ==================================================================
# STEP 3: 3 CNN ARCHITECTURES
# ==================================================================
class CNN1Block(nn.Module):
    """1 conv block: Conv(16) -> ReLU -> MaxPool -> FC"""
    def __init__(self, n_classes, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))          # 28 -> 14
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 14 * 14, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class CNN2Block(nn.Module):
    """2 conv blocks: Conv(16)-Pool -> Conv(32)-Pool -> FC"""
    def __init__(self, n_classes, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),          # 28 -> 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))         # 14 -> 7
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class CNN3BlockBN(nn.Module):
    """3 conv blocks + BatchNorm + Dropout: 32-64-128 filters"""
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                    nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*block(1, 32), *block(32, 64), *block(64, 128))  # 28->14->7->3
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(128 * 3 * 3, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


ARCHS = {"cnn_1block": CNN1Block, "cnn_2block": CNN2Block, "cnn_3block_bn": CNN3BlockBN}

# ==================================================================
# STEP 4: EXPERIMENTS (alag architecture + alag training settings)
# ==================================================================
EXPERIMENTS = [
    dict(name="E1_1block", arch="cnn_1block", optimizer="adam", lr=1e-3, augment=False, dropout=0.0),
    dict(name="E2_2block", arch="cnn_2block", optimizer="adam", lr=1e-3, augment=False, dropout=0.0),
    dict(name="E3_2block_SGD", arch="cnn_2block", optimizer="sgd", lr=1e-2, augment=False, dropout=0.0),
    dict(name="E4_2block_aug", arch="cnn_2block", optimizer="adam", lr=1e-3, augment=True, dropout=0.0),
    dict(name="E5_3block_BN_drop_aug", arch="cnn_3block_bn", optimizer="adam", lr=1e-3, augment=True, dropout=0.3),
]


@torch.no_grad()
def predict_logits(model, X, bs=1000):
    model.eval()
    return torch.cat([model(norm(X[i:i + bs])) for i in range(0, len(X), bs)])


def evaluate(model, X, y):
    logits = predict_logits(model, X)
    return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def run_experiment(cfg):
    torch.manual_seed(SEED)
    model = ARCHS[cfg["arch"]](N_CLASSES, cfg["dropout"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    if cfg["optimizer"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    else:
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best = {"val_acc": -1.0, "epoch": 0, "state": None}
    t0 = time.time()
    n = len(X_train)
    print(f"\n=== {cfg['name']} | {cfg['arch']} | {cfg['optimizer']} lr={cfg['lr']} | "
          f"augment={cfg['augment']} | params={n_params:,}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(n, device=DEVICE)
        tot_loss, correct = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            xb, yb = X_train[idx], y_train[idx]
            if cfg["augment"]:
                xb = augment(xb)
            out = model(norm(xb))
            loss = F.cross_entropy(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
        val_loss, val_acc = evaluate(model, X_val, y_val)
        hist["train_loss"].append(tot_loss / n)
        hist["train_acc"].append(correct / n)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        if val_acc > best["val_acc"]:
            best.update(val_acc=val_acc, epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f"  epoch {epoch}/{EPOCHS} | train acc {correct / n:.4f} | "
              f"val acc {val_acc:.4f} | val loss {val_loss:.4f} | {time.time() - t0:.0f}s")

    model.load_state_dict(best["state"])          # best epoch ke weights wapas
    result = dict(cfg, params=n_params, best_val_acc=best["val_acc"], best_epoch=best["epoch"],
                  final_train_acc=hist["train_acc"][-1], final_val_acc=hist["val_acc"][-1],
                  train_time_sec=round(time.time() - t0, 1), history=hist)
    return model, result


models, results = {}, []
for cfg in EXPERIMENTS:
    m, r = run_experiment(cfg)
    models[cfg["name"]] = m
    results.append(r)

best_res = max(results, key=lambda r: r["best_val_acc"])
best_name = best_res["name"]
best_model = models[best_name]
print(f"\nBest experiment (by validation accuracy): {best_name} ({best_res['best_val_acc']:.4f})")

# ==================================================================
# STEP 5: FINAL EVALUATION - test set (sirf ek dafa, sirf best model)
# ==================================================================
logits = predict_logits(best_model, X_test)
probs = F.softmax(logits, dim=1)
conf, pred = probs.max(1)
y_np, pred_np, conf_np = y_test.cpu().numpy(), pred.cpu().numpy(), conf.cpu().numpy()
test_acc = float((pred_np == y_np).mean())
labels = list(range(N_CLASSES))
cm = confusion_matrix(y_np, pred_np, labels=labels)
rep = classification_report(y_np, pred_np, labels=labels, target_names=CLASS_NAMES,
                            output_dict=True, zero_division=0)
wrong = np.where(pred_np != y_np)[0]

off = cm.copy()
np.fill_diagonal(off, 0)
top_conf = []
for flat in np.argsort(off, axis=None)[::-1][:6]:
    i, j = divmod(int(flat), N_CLASSES)
    if off[i, j] > 0:
        top_conf.append(dict(true=CLASS_NAMES[i], pred=CLASS_NAMES[j], count=int(off[i, j])))

print(f"TEST accuracy: {test_acc:.4f} | macro F1: {rep['macro avg']['f1-score']:.4f} | "
      f"wrong: {len(wrong)}/{len(y_np)}")
print("Most confused pairs (true -> predicted):")
for t in top_conf:
    print(f"   {t['true']} -> {t['pred']}: {t['count']}")

# ==================================================================
# STEP 6: PLOTS
# ==================================================================
def img_of(x):
    return x.squeeze().detach().cpu().numpy()


# 6a) Sample images (har class ki ek image) + class distribution
ytr_np = y_train.cpu().numpy()
fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for c, ax in zip(range(N_CLASSES), axes.ravel()):
    idx = int(np.where(ytr_np == c)[0][0])
    ax.imshow(img_of(X_train[idx]), cmap="gray")
    ax.set_title(CLASS_NAMES[c], fontsize=9)
    ax.axis("off")
plt.suptitle("One sample image per class (28x28 grayscale)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_sample_images.png", dpi=110)
plt.close()

plt.figure(figsize=(8, 4))
counts = np.bincount(ytr_np, minlength=N_CLASSES)
plt.bar(range(N_CLASSES), counts)
plt.xticks(range(N_CLASSES), CLASS_NAMES, rotation=45, ha="right")
plt.ylabel("Images in training split")
plt.title("Class distribution (training split)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_class_distribution.png", dpi=110)
plt.close()

# 6b) Augmentation examples
torch.manual_seed(1)
sample = X_test[:6]
fig, axes = plt.subplots(3, 6, figsize=(10, 5.4))
rows = [("Original", sample), ("Augmented #1", augment(sample)), ("Augmented #2", augment(sample))]
for r, (title, batch) in enumerate(rows):
    for c in range(6):
        axes[r, c].imshow(img_of(batch[c]), cmap="gray")
        axes[r, c].axis("off")
    axes[r, 0].set_title(title, fontsize=9, loc="left")
plt.suptitle("Data augmentation: random rotation (+-10 deg), shift (+-2 px), horizontal flip")
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_augmentation_examples.png", dpi=110)
plt.close()

# 6c) Training curves
colors = plt.cm.tab10.colors
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for k, r in enumerate(results):
    ep = range(1, len(r["history"]["val_acc"]) + 1)
    axes[0].plot(ep, r["history"]["val_acc"], color=colors[k], marker="o", label=f"{r['name']} (val)")
    axes[0].plot(ep, r["history"]["train_acc"], color=colors[k], linestyle="--", alpha=0.6)
    axes[1].plot(ep, r["history"]["val_loss"], color=colors[k], marker="o", label=r["name"])
axes[0].set_title("Accuracy per epoch (solid = validation, dashed = training)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy"); axes[0].legend(fontsize=7)
axes[1].set_title("Validation loss per epoch")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(PLOT_DIR / "04_training_curves.png", dpi=110)
plt.close()

# 6d) Experiment comparison
plt.figure(figsize=(8, 4.2))
names = [r["name"] for r in results]
vals = [r["best_val_acc"] for r in results]
plt.barh(names[::-1], vals[::-1], color=[colors[i] for i in range(len(names))][::-1])
plt.xlim(max(0.0, min(vals) - 0.03), min(1.0, max(vals) + 0.02))
for i, v in enumerate(vals[::-1]):
    plt.text(v, i, f" {v:.4f}", va="center", fontsize=8)
plt.xlabel("Best validation accuracy")
plt.title("Experiment comparison")
plt.tight_layout()
plt.savefig(PLOT_DIR / "05_experiment_comparison.png", dpi=110)
plt.close()

# 6e) Confusion matrix
plt.figure(figsize=(8.5, 7.2))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(N_CLASSES), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
plt.yticks(range(N_CLASSES), CLASS_NAMES, fontsize=8)
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title(f"Confusion matrix - {best_name} (test set)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "06_confusion_matrix.png", dpi=110)
plt.close()

# 6f) Misclassified images (sabse zyada confident galtiyan)
if len(wrong) > 0:
    order = wrong[np.argsort(-conf_np[wrong])][:24]
    fig, axes = plt.subplots(4, 6, figsize=(11, 8.2))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), order):
        ax.imshow(img_of(X_test[int(idx)]), cmap="gray")
        ax.set_title(f"True: {CLASS_NAMES[y_np[idx]]}\nPred: {CLASS_NAMES[pred_np[idx]]} "
                     f"({conf_np[idx]:.0%})", fontsize=7.5, color="red")
    plt.suptitle("Misclassified test images (most confident mistakes first)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "07_misclassified_images.png", dpi=110)
    plt.close()

# 6g) Learned conv1 filters
convs = [m for m in best_model.modules() if isinstance(m, nn.Conv2d)]
w = convs[0].weight.detach().cpu().numpy()          # (out, 1, 3, 3)
n_f = w.shape[0]
cols = 8
rows_f = math.ceil(n_f / cols)
fig, axes = plt.subplots(rows_f, cols, figsize=(1.6 * cols, 1.7 * rows_f))
for ax in np.atleast_1d(axes).ravel():
    ax.axis("off")
for i in range(n_f):
    ax = np.atleast_1d(axes).ravel()[i]
    ax.imshow(w[i, 0], cmap="coolwarm")
    ax.set_title(f"f{i + 1}", fontsize=7)
plt.suptitle(f"Learned 3x3 filters of the first conv layer ({n_f} filters)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "08_conv1_filters.png", dpi=110)
plt.close()

# 6h) Feature maps (har conv layer ke baad, ReLU ke saath)
ok = np.where(pred_np == y_np)[0]
fm_idx = int(ok[0]) if len(ok) else 0
x_in = X_test[fm_idx:fm_idx + 1]
maps = {}
hooks = []
for k, m in enumerate(convs, start=1):
    hooks.append(m.register_forward_hook(
        lambda mod, inp, out, k=k: maps.__setitem__(k, torch.relu(out).detach().cpu()[0].numpy())))
best_model.eval()
with torch.no_grad():
    best_model(norm(x_in))
for h in hooks:
    h.remove()

n_show = 8
fig, axes = plt.subplots(len(maps), n_show + 1, figsize=(1.7 * (n_show + 1), 1.9 * len(maps)))
axes = np.atleast_2d(axes)
for r, k in enumerate(sorted(maps)):
    axes[r, 0].imshow(img_of(x_in), cmap="gray")
    axes[r, 0].set_title("input", fontsize=8)
    for c in range(n_show):
        ax = axes[r, c + 1]
        if c < maps[k].shape[0]:
            ax.imshow(maps[k][c], cmap="viridis")
            ax.set_title(f"conv{k} ch{c + 1}", fontsize=7)
    for ax in axes[r]:
        ax.axis("off")
plt.suptitle(f"Feature maps after each conv layer + ReLU (test image: {CLASS_NAMES[y_np[fm_idx]]})")
plt.tight_layout()
plt.savefig(PLOT_DIR / "09_feature_maps_overview.png", dpi=110)
plt.close()

for k in sorted(maps):
    ch = min(16, maps[k].shape[0])
    fig, axes = plt.subplots(2, 8, figsize=(13, 3.6))
    for ax in axes.ravel():
        ax.axis("off")
    for c in range(ch):
        ax = axes.ravel()[c]
        ax.imshow(maps[k][c], cmap="viridis")
        ax.set_title(f"ch{c + 1}", fontsize=8)
    plt.suptitle(f"Conv layer {k}: {maps[k].shape[0]} feature maps of size "
                 f"{maps[k].shape[1]}x{maps[k].shape[2]} (first {ch} shown)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"10_feature_maps_conv{k}.png", dpi=110)
    plt.close()

# ==================================================================
# STEP 7: SAVE MODEL + RESULTS
# ==================================================================
torch.save(best_model.state_dict(), OUT_DIR / "best_cnn.pt")
summary = {
    "config": dict(epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED, quick=args.quick,
                   device=str(DEVICE), n_train=len(X_train), n_val=len(X_val), n_test=len(X_test),
                   mean=MEAN, std=STD),
    "class_names": CLASS_NAMES,
    "experiments": results,
    "best_experiment": best_name,
    "test": dict(
        accuracy=test_acc, macro_f1=rep["macro avg"]["f1-score"], n_wrong=int(len(wrong)),
        per_class={c: {k: float(v) for k, v in rep[c].items()} for c in CLASS_NAMES},
        confusion_matrix=cm.tolist(), top_confusions=top_conf,
        mean_conf_correct=float(conf_np[pred_np == y_np].mean()) if (pred_np == y_np).any() else 0.0,
        mean_conf_wrong=float(conf_np[wrong].mean()) if len(wrong) else 0.0),
    "feature_map_sample": dict(index=fm_idx, cls=CLASS_NAMES[int(y_np[fm_idx])]),
    "n_conv_layers": len(convs),
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 78)
print(f"{'Experiment':<24}{'Params':>10}{'Best val acc':>14}{'Train acc':>11}{'Time(s)':>9}")
for r in results:
    print(f"{r['name']:<24}{r['params']:>10,}{r['best_val_acc']:>14.4f}"
          f"{r['final_train_acc']:>11.4f}{r['train_time_sec']:>9.0f}")
print("=" * 78)
print(f"BEST: {best_name} | TEST accuracy = {test_acc:.4f}")
print("Saved: outputs/results.json, outputs/best_cnn.pt, outputs/plots/*.png")
