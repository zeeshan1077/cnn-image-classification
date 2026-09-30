"""
Word report banata hai outputs/results.json aur outputs/plots/ se.

Pehle chalao:  python src/train.py
Phir:          python src/build_report.py
Output:        outputs/CNN_Project_Report.docx

(zaroorat: pip install python-docx)
"""
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results.json"
PLOTS = ROOT / "outputs" / "plots"
OUT_FILE = ROOT / "outputs" / "CNN_Project_Report.docx"
AUTHOR = "Zeeshan Aziz Khan"

if not RESULTS.exists():
    raise SystemExit("ERROR: outputs/results.json nahi mili. Pehle 'python src/train.py' chalao.")

R = json.loads(RESULTS.read_text())
CFG, EXPS, TEST = R["config"], R["experiments"], R["test"]
E = {e["name"]: e for e in EXPS}
BEST = E[R["best_experiment"]]
CLASSES = R["class_names"]

# ------------------------------------------------------------------
# Document helpers
# ------------------------------------------------------------------
doc = Document()
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
for s in doc.sections:
    s.left_margin = s.right_margin = Inches(0.9)
    s.top_margin = s.bottom_margin = Inches(0.9)

_fig = [0]


def heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.color.rgb = RGBColor(0, 0, 0)
    return h


def para(text="", bold=False, italic=False, size=None, align=None, color=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold, run.italic = bold, italic
    if size:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(6)
    return p


def bullet(text):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    return p


def picture(name, caption, width=6.0):
    path = PLOTS / name
    if not path.exists():
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    _fig[0] += 1
    cap = doc.add_paragraph()
    run = cap.add_run(f"Figure {_fig[0]}: {caption}")
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(90, 90, 90)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def table(headers, rows, font=9, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    if widths:
        t.autofit = False
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(str(h))
        run.bold = True
        run.font.size = Pt(font)
        run.font.color.rgb = RGBColor(255, 255, 255)
        shade(c, "1F4E79")
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = ""
            cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(font)
    if widths:
        for i, w in enumerate(widths):
            t.columns[i].width = Inches(w)
            for row in t.rows:
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return t


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def compare(a_name, b_name, what):
    """a -> b : validation accuracy ka farq, sach-sach (sign ke hisaab se lafz badalte hain)."""
    a, b = E.get(a_name), E.get(b_name)
    if not a or not b:
        return None
    d = (b["best_val_acc"] - a["best_val_acc"]) * 100
    if abs(d) < 0.005:
        verb = "left validation accuracy essentially unchanged"
        detail = f"({pct(a['best_val_acc'])} vs {pct(b['best_val_acc'])})"
    else:
        verb = "improved validation accuracy by" if d > 0 else "reduced validation accuracy by"
        detail = f"{abs(d):.2f} points ({pct(a['best_val_acc'])} -> {pct(b['best_val_acc'])})"
    noise = " This is a small difference and may be within run-to-run noise." if abs(d) < 0.3 else ""
    return f"{what} {verb} {detail}.{noise}"


# ------------------------------------------------------------------
# Title
# ------------------------------------------------------------------
t = doc.add_heading("Image Classification with a Custom CNN - Project Report", level=0)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in t.runs:
    r.font.color.rgb = RGBColor(0, 0, 0)
para(f"Prepared by {AUTHOR}  |  Internship Task Report", italic=True, size=10,
     align=WD_ALIGN_PARAGRAPH.CENTER, color=(85, 85, 85))
if CFG.get("quick"):
    para("NOTE: this report was generated from a QUICK test run (small data subset, few epochs). "
         "Re-run 'python src/train.py' without --quick for final numbers.",
         bold=True, color=(180, 0, 0))

# ------------------------------------------------------------------
# 1. Objective
# ------------------------------------------------------------------
heading("1. Objective")
para("Study the fundamentals of computer vision and convolutional neural networks (CNNs), then build a "
     "custom CNN in PyTorch for image classification on the Fashion-MNIST dataset (Kaggle). Different "
     "architectures and training settings are compared through controlled experiments, the best model is "
     "evaluated on a held-out test set, misclassified images are analysed, and the feature maps learned by "
     "the network are visualised.")

# ------------------------------------------------------------------
# 2. Fundamentals
# ------------------------------------------------------------------
heading("2. Computer Vision and CNN Fundamentals")
heading("2.1 Image representation", 2)
para("A digital image is a grid of pixels. A grayscale image is a 2-D matrix of intensity values from 0 "
     "(black) to 255 (white); a colour image has three such matrices (Red, Green, Blue channels). The "
     "images used here are 28x28 grayscale, i.e. 784 numbers per image, stored as one row of pixel columns "
     "in the Kaggle CSV. A neural network does not 'see' a picture - it only receives these numbers.")
picture("demo_1_image_as_numbers.png",
        "An image is just a matrix of numbers (right: an 8x8 crop with its pixel values).", 5.8)

heading("2.2 Preprocessing", 2)
para("Pixel values are first scaled from 0-255 to 0-1 and then standardised with the mean and standard "
     "deviation of the training set. Normalisation keeps the inputs on a similar scale, which makes "
     "gradient-based training faster and more stable. Images are reshaped to (channels x height x width) = "
     "(1 x 28 x 28) as expected by convolutional layers.")

heading("2.3 Convolution, filters (kernels) and feature maps", 2)
para("A convolutional layer slides a small matrix of weights - a filter or kernel, e.g. 3x3 - over the "
     "image. At every position it multiplies the kernel with the pixels underneath and sums the result into "
     "one number. The grid of these numbers is a feature map: it is bright where the pattern encoded by "
     "the kernel (an edge, a corner, a texture) is present. Hand-made kernels such as Sobel detect vertical "
     "or horizontal edges; in a CNN the kernel values are not designed by hand but learned from the data by "
     "back-propagation. Because the same kernel is reused at every position, a conv layer needs far fewer "
     "parameters than a fully connected layer and detects a pattern wherever it appears in the image.")
picture("demo_2_kernels_feature_maps.png",
        "The same image filtered with different 3x3 kernels produces different feature maps.", 6.3)

heading("2.4 Stride and padding", 2)
para("Stride is the number of pixels the kernel moves at each step (stride 2 halves the output size). "
     "Padding adds a border of zeros around the image so that the kernel can also be centred on edge "
     "pixels and the output does not shrink. The output size of a convolution is:")
para("output = (W - K + 2P) / S + 1", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
para("where W is the input width, K the kernel size, P the padding and S the stride. For W=28, K=3: "
     "P=0,S=1 gives 26; P=1,S=1 gives 28 ('same'); P=1,S=2 gives 14.")

heading("2.5 Pooling", 2)
para("Pooling down-samples a feature map by summarising small windows. Max-pooling (2x2, stride 2) keeps the "
     "strongest response in each window; average-pooling keeps the mean. Pooling halves the height and "
     "width, reduces computation, and gives small translation invariance (the network cares less about the "
     "exact position of a feature).")
picture("demo_3_stride_padding_pooling.png",
        "Effect of padding, stride and pooling on the size of the output feature map.", 6.5)

heading("2.6 Feature learning", 2)
para("A CNN stacks several conv-ReLU-pool blocks. Early layers learn simple, local patterns (edges, "
     "brightness changes); deeper layers combine them into more complex shapes and parts; the final fully "
     "connected layers turn these features into class scores. The important idea is that the features are "
     "learned automatically from data instead of being engineered manually.")

heading("2.7 Data augmentation", 2)
para("Augmentation creates modified copies of training images on the fly (small rotations, shifts, "
     "flips). The label stays the same, but the network sees more variety, which acts as a regulariser "
     "and reduces overfitting, particularly when the dataset is small or the model is large.")

# ------------------------------------------------------------------
# 3. Dataset
# ------------------------------------------------------------------
heading("3. Dataset")
para(f"Fashion-MNIST (Zalando Research, downloaded from Kaggle as CSV): 70,000 grayscale 28x28 images of "
     f"clothing items in {len(CLASSES)} classes ({', '.join(CLASSES)}). The official split has 60,000 "
     f"training and 10,000 test images. In this project the training file was split into "
     f"{CFG['n_train']:,} training and {CFG['n_val']:,} validation images (random split, seed {CFG['seed']}); "
     f"the {CFG['n_test']:,} test images were kept untouched until the final evaluation of the selected model.")
picture("01_sample_images.png", "One sample image from each class.", 5.8)
picture("02_class_distribution.png", "Class distribution of the training split (classes are balanced).", 5.0)

# ------------------------------------------------------------------
# 4. Preprocessing & augmentation implemented
# ------------------------------------------------------------------
heading("4. Preprocessing and Data Augmentation Used")
bullet(f"Pixels scaled to 0-1, then standardised with the training-set mean ({CFG['mean']:.4f}) and standard "
       f"deviation ({CFG['std']:.4f}); the test set uses the same values, so no test information leaks into training.")
bullet("Augmentation (applied only to training batches, only in the augmented experiments): random rotation "
       "of up to +-10 degrees, random shift of up to +-2 pixels and random horizontal flip.")
bullet("Validation and test images are never augmented.")
picture("03_augmentation_examples.png", "Original test images and two randomly augmented versions of each.", 6.0)

# ------------------------------------------------------------------
# 5. Architectures
# ------------------------------------------------------------------
heading("5. CNN Architectures")
arch_desc = {
    "cnn_1block": ("1 conv block", "Conv(1->16, 3x3) - ReLU - MaxPool -> FC(64) -> FC(classes)"),
    "cnn_2block": ("2 conv blocks", "Conv(1->16) - ReLU - Pool -> Conv(16->32) - ReLU - Pool -> FC(128) -> FC(classes)"),
    "cnn_3block_bn": ("3 conv blocks + BatchNorm + Dropout",
                      "3 x [Conv - BatchNorm - ReLU - Pool] with 32-64-128 filters -> Dropout - FC(128) - Dropout - FC(classes)"),
}
rows = []
for key, (name, layers) in arch_desc.items():
    p = next((e["params"] for e in EXPS if e["arch"] == key), "-")
    rows.append([name, layers, f"{p:,}" if isinstance(p, int) else p])
table(["Architecture", "Layers", "Parameters"], rows)
para("All convolutions use 3x3 kernels with padding 1 (so only the pooling layers shrink the feature maps, "
     "28 -> 14 -> 7 -> 3). BatchNorm normalises the activations of each layer and stabilises training; "
     "Dropout randomly switches off neurons during training to reduce overfitting.")

# ------------------------------------------------------------------
# 6. Experiments
# ------------------------------------------------------------------
heading("6. Experiments")
para(f"Five experiments were run with identical data splits, batch size ({CFG['batch_size']}), random seed and "
     f"{CFG['epochs']} epochs each. The weights of the best validation epoch were kept for every run. "
     f"The experiments change one factor at a time where possible: depth (E1 vs E2), optimizer (E2 vs E3), "
     f"augmentation (E2 vs E4) and a deeper network with BatchNorm and Dropout (E4 vs E5).")
rows = []
for e in EXPS:
    rows.append([e["name"], e["arch"], f"{e['optimizer'].upper()} {e['lr']}", "Yes" if e["augment"] else "No",
                 e["dropout"], f"{e['params']:,}", pct(e["best_val_acc"]), e["best_epoch"],
                 pct(e["final_train_acc"]), f"{e['train_time_sec']:.0f}"])
table(["Experiment", "Architecture", "Optimizer / LR", "Aug.", "Dropout", "Params", "Best val acc",
       "Best epoch", "Final train acc", "Time (s)"], rows, font=8,
      widths=[1.55, 0.95, 0.8, 0.4, 0.5, 0.65, 0.6, 0.45, 0.6, 0.4])
picture("05_experiment_comparison.png", "Best validation accuracy of each experiment.", 5.4)
picture("04_training_curves.png",
        "Training (dashed) and validation (solid) accuracy, and validation loss, per epoch.", 6.5)

heading("6.1 Observations", 2)
for s in (compare("E1_1block", "E2_2block", "Adding a second convolutional block (E1 -> E2)"),
          compare("E3_2block_SGD", "E2_2block", "Using Adam (lr 0.001) instead of SGD with momentum (lr 0.01) (E3 -> E2)"),
          compare("E2_2block", "E4_2block_aug", "Adding data augmentation (E2 -> E4)"),
          compare("E4_2block_aug", "E5_3block_BN_drop_aug",
                  "Moving to the deeper 3-block network with BatchNorm and Dropout (E4 -> E5)")):
    if s:
        bullet(s)
bullet(f"Best experiment: {BEST['name']} with {pct(BEST['best_val_acc'])} validation accuracy "
       f"(best epoch {BEST['best_epoch']} of {CFG['epochs']}).")
para("Note: differences of a few tenths of a percent come from single runs with one random seed and should "
     "not be over-interpreted; repeating each experiment with several seeds would be needed to claim a "
     "statistically reliable ranking.", italic=True)

# ------------------------------------------------------------------
# 7. Overfitting / underfitting
# ------------------------------------------------------------------
heading("7. Overfitting and Underfitting Analysis")
gaps = {e["name"]: (e["final_train_acc"] - e["final_val_acc"]) * 100 for e in EXPS}
hi = max(gaps, key=gaps.get)
lo = min(gaps, key=gaps.get)
bullet(f"Largest train-validation accuracy gap: {hi} ({gaps[hi]:.2f} points); lowest: {lo} ({gaps[lo]:.2f} points). "
       f"A large positive gap means the network fits the training images better than unseen images (overfitting). "
       f"A negative value means training accuracy is below validation accuracy: training accuracy is a running "
       f"average over the epoch (while the weights are still improving) and, for augmented runs, is measured on "
       f"harder augmented images.")
flags = [e["name"] for e in EXPS
         if e["history"]["val_loss"][-1] > 1.03 * min(e["history"]["val_loss"])]
if flags:
    bullet(f"Validation loss started to rise again before the last epoch in: {', '.join(flags)} - the classic sign "
           f"of overfitting. This is why the best-epoch weights were kept instead of the last-epoch weights.")
else:
    bullet(f"No run showed a clear rise in validation loss within {CFG['epochs']} epochs.")
g = gaps[BEST["name"]]
if g <= 0:
    bullet(f"The selected model ({BEST['name']}) has no positive train-validation gap ({g:.2f} points), "
           f"so there is no sign of overfitting in this run.")
else:
    level = "small" if g < 2 else ("moderate" if g < 5 else "large")
    bullet(f"The selected model ({BEST['name']}) has a {level} train-validation gap of {g:.2f} points.")
bullet("Underfitting would show up as low accuracy on both training and validation data; the 1-block network is "
       "the most likely candidate because it has the least capacity.")

# ------------------------------------------------------------------
# 8. Final evaluation
# ------------------------------------------------------------------
heading("8. Final Evaluation on the Test Set")
para(f"The selected model ({BEST['name']}) was evaluated once on the {CFG['n_test']:,} untouched test images: "
     f"accuracy {pct(TEST['accuracy'])}, macro-averaged F1 {pct(TEST['macro_f1'])}, "
     f"{TEST['n_wrong']:,} misclassified images.")
rows = [[c, f"{TEST['per_class'][c]['precision']:.3f}", f"{TEST['per_class'][c]['recall']:.3f}",
         f"{TEST['per_class'][c]['f1-score']:.3f}", int(TEST["per_class"][c]["support"])] for c in CLASSES]
table(["Class", "Precision", "Recall", "F1-score", "Support"], rows)
best_c = max(CLASSES, key=lambda c: TEST["per_class"][c]["f1-score"])
worst_c = min(CLASSES, key=lambda c: TEST["per_class"][c]["f1-score"])
para(f"Easiest class: {best_c} (F1 {TEST['per_class'][best_c]['f1-score']:.3f}). "
     f"Hardest class: {worst_c} (F1 {TEST['per_class'][worst_c]['f1-score']:.3f}).")
picture("06_confusion_matrix.png", "Confusion matrix of the selected model on the test set.", 5.8)

# ------------------------------------------------------------------
# 9. Misclassification analysis
# ------------------------------------------------------------------
heading("9. Analysis of Misclassified Images")
err = TEST["n_wrong"] / CFG["n_test"]
para(f"{TEST['n_wrong']:,} of {CFG['n_test']:,} test images ({pct(err)}) were misclassified. The most frequent "
     f"confusions (true class -> predicted class) are:")
if TEST["top_confusions"]:
    table(["True class", "Predicted as", "Count"],
          [[t["true"], t["pred"], t["count"]] for t in TEST["top_confusions"]])
mc, mw = TEST["mean_conf_correct"], TEST["mean_conf_wrong"]
if mw < mc:
    para(f"The network is on average less confident when it is wrong ({pct(mw, 1)}) than when it is right "
         f"({pct(mc, 1)}), which is a healthy behaviour, but it still makes some very confident mistakes "
         f"(shown first in the figure below).")
else:
    para(f"The network is about as confident when it is wrong ({pct(mw, 1)}) as when it is right "
         f"({pct(mc, 1)}), so its confidence score cannot be used to detect its own mistakes.")
picture("07_misclassified_images.png",
        "Misclassified test images, sorted so that the most confident mistakes come first.", 6.4)
para("Likely explanations (observations, not proven causes):", bold=True)
bullet("Classes with similar silhouettes are confused with each other; at 28x28 grayscale resolution the fine "
       "details that separate them (collars, buttons, sleeve length, texture) are largely lost.")
bullet("Some images are genuinely ambiguous, even for a human, because of the way the item is photographed or cropped.")
bullet("The network was trained for only a few epochs on a CPU-friendly budget; longer training, more augmentation "
       "or a deeper network could reduce some of these errors.")

# ------------------------------------------------------------------
# 10. Feature maps
# ------------------------------------------------------------------
heading("10. Visualisation of Filters and Feature Maps")
para("The first-layer filters of the selected model and the feature maps produced for one test image "
     f"(class: {R['feature_map_sample']['cls']}) after every convolution + ReLU are shown below.")
picture("08_conv1_filters.png", "Learned 3x3 filters of the first convolutional layer.", 6.0)
picture("09_feature_maps_overview.png", "Feature maps of the first 8 channels after each convolutional layer.", 6.5)
for k in range(1, R["n_conv_layers"] + 1):
    picture(f"10_feature_maps_conv{k}.png", f"Feature maps of convolutional layer {k}.", 6.5)
bullet("Feature maps of the first layer look like filtered versions of the input: they highlight edges, outlines "
       "and bright/dark regions of the garment; different filters respond to different orientations.")
bullet("Deeper feature maps are spatially smaller and harder to read visually, because each value combines many "
       "lower-level responses over a larger part of the image; they encode 'what kind of shape or part is "
       "here' rather than raw pixel intensity.")
bullet("Some channels are almost blank for a given image: those filters respond to patterns that are not "
       "present in that particular image.")

# ------------------------------------------------------------------
# 11. Limitations, conclusion, reproduce
# ------------------------------------------------------------------
heading("11. Limitations")
bullet("Fashion-MNIST is small (28x28), grayscale and clean; performance on real-world colour photographs "
       "would be lower and would need larger networks or transfer learning.")
bullet("Each experiment was run once with a single seed and a small number of epochs; small differences between "
       "experiments may be noise.")
bullet("Only a limited set of architectures and hyper-parameters was explored (no learning-rate schedule, "
       "no systematic hyper-parameter search).")
bullet("The test set was used once for the selected model only, so the reported test accuracy is an unbiased estimate, "
       "but it is a single number without a confidence interval.")

heading("12. Conclusion")
para(f"A custom CNN trained from scratch reached {pct(TEST['accuracy'])} test accuracy on Fashion-MNIST with "
     f"the '{BEST['name']}' configuration. The experiments show how network depth, optimizer choice, "
     f"augmentation, BatchNorm and Dropout affect accuracy and overfitting, and the misclassification and "
     f"feature-map analyses explain what the network learned and where it fails.")

heading("Appendix: How to reproduce")
for cmd in ("pip install torch numpy pandas matplotlib scikit-learn python-docx",
            "python src/conv_demo.py", "python src/train.py", "python src/build_report.py"):
    p = para(cmd)
    for r in p.runs:
        r.font.name = "Consolas"
        r.font.size = Pt(10)

doc.save(OUT_FILE)
print(f"Report saved: {OUT_FILE}")
