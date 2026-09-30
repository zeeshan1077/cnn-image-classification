# Computer Vision / CNN Image Classification Project

## What this does
Trains a custom Convolutional Neural Network (PyTorch) to classify clothing
images (Fashion-MNIST), runs 5 experiments comparing architectures and
training settings, evaluates the best one, analyses its mistakes, and
visualises what its filters and feature maps look like.

## 1. Get the real dataset from Kaggle
1. Go to: https://www.kaggle.com/datasets/zalando-research/fashionmnist
2. Click **Download**, unzip it
3. You need two files: `fashion-mnist_train.csv` and `fashion-mnist_test.csv`
4. Put both of them in this project's `data/` folder

Each row = 1 label (0-9) + 784 pixel columns (28x28 image flattened).
Class numbers: 0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat,
5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot.

## 2. Install dependencies
```
pip install torch numpy pandas matplotlib scikit-learn python-docx
```
(If `pip install torch` is slow/large, that is normal — it is a big library.)

## 3. Run, in this order
```
python src/conv_demo.py        # fundamentals: image as numbers, kernels, padding/stride/pooling
python src/train.py --quick    # fast ~1-2 min sanity check first
python src/train.py            # full run (5 experiments, ~10-20 min on CPU)
python src/build_report.py     # builds outputs/CNN_Project_Report.docx from the results
```

## 4. What you get in outputs/
- `results.json` — every experiment's numbers
- `best_cnn.pt` — saved weights of the best model
- `plots/` — 15+ charts: fundamentals demo, sample images, augmentation examples,
  training curves, experiment comparison, confusion matrix, misclassified images,
  learned filters, feature maps
- `CNN_Project_Report.docx` — the full written report

## Author
Zeeshan Aziz Khan
