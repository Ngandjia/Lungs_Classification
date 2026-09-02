# Chest X-Ray Classification with PyTorch

## Overview

This repository contains a computer-vision project that develops a convolutional neural network (CNN) for binary classification of chest X-ray images into **NORMAL** and **PNEUMONIA**.

The project is designed not only as a machine-learning implementation, but also as a mathematical study of the complete learning pipeline. The notebook explicitly analyzes tensor dimensions, convolutional parameter counts, nonlinear maps, optimization, regularization, and the effect of architectural modifications on generalization.

### Main technologies

- **Python**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **Matplotlib**
- **OpenCV**
- **tqdm**
- **torchsummary**

## Project structure

The notebook follows an experimental progression:

1. **Dataset exploration**
   - inspect the class distribution;
   - inspect raw image dimensions;
   - visualize representative chest X-rays.

2. **Preprocessing and data loading**
   - resize images to 224 x 224;
   - convert images to tensors;
   - normalize image channels;
   - construct mini-batches using `DataLoader`.

3. **Baseline CNN**
   - build a CNN from scratch;
   - explicitly track channels and spatial dimensions;
   - calculate the number of trainable parameters;
   - train using SGD with momentum;
   - evaluate using negative log-likelihood loss and accuracy.

4. **Data augmentation**
   - brightness/contrast/saturation/hue perturbations;
   - random horizontal flips;
   - random rotations.

5. **Batch normalization**
   - add `BatchNorm2d` throughout the convolutional architecture;
   - study the effect on optimization stability and generalization.

6. **L1 regularization**
   - modify the objective to
     \[
     L(\theta)=L_{\mathrm{NLL}}(\theta)+\lambda\|\theta\|_1;
     \]
   - investigate whether explicit parameter regularization improves generalization.

## Mathematical viewpoint

The network can be regarded as a parameterized nonlinear map

\[
f_\theta:
\mathbb R^{3\times224\times224}\rightarrow\mathbb R^2.
\]

Each convolutional layer is a local linear operator followed by a nonlinear activation. For a convolution with \(C_{\rm in}\) input channels, \(C_{\rm out}\) output channels and kernel size K x K, the number of weights (with no bias) is

\[
C_{\rm out}C_{\rm in}K^2.
\]

The baseline architecture contains **8,396 trainable convolutional parameters**.

The batch-normalized architecture has the same 8,396 convolutional parameters plus 316 trainable normalization parameters, giving **8,712 trainable parameters**.

Training minimizes an empirical risk of the form

\[
\widehat R(\theta)
=
\frac1N
\sum_{i=1}^N
\ell(f_\theta(x_i),y_i),
\]

with SGD and momentum used as the numerical optimization method.

With L1 regularization, the objective becomes

\[
\widehat R_\lambda(\theta)
=
\frac1N
\sum_{i=1}^N
\ell(f_\theta(x_i),y_i)
+
\lambda\|\theta\|_1.
\]

This gives the project a direct connection to linear algebra, multivariable calculus, probability/statistics, numerical optimization, normed spaces, and regularization theory.

## Architecture

The baseline CNN takes an RGB tensor

\[
(3,224,224)
\]

and progressively transforms it through convolution, ReLU, max pooling, 1 x 1 channel-mixing convolutions, average pooling, and a final 4 x 4 convolution.

The spatial progression for one image is:

```text
Input                  3 × 224 × 224
Conv 3→8               8 × 222 × 222
MaxPool                8 × 111 × 111
Conv 8→16              16 × 109 × 109
MaxPool                16 × 54 × 54
Conv 16→10 (1×1)       10 × 54 × 54
MaxPool                10 × 27 × 27
Conv 10→10             10 × 25 × 25
Conv 10→32 (1×1)       32 × 25 × 25
Conv 32→10 (1×1)       10 × 25 × 25
Conv 10→10             10 × 23 × 23
Conv 10→32 (1×1)       32 × 23 × 23
Conv 32→10 (1×1)       10 × 23 × 23
Conv 10→14             14 × 21 × 21
Conv 14→16             16 × 19 × 19
AvgPool 4×4            16 × 4 × 4
Conv 16→2              2 × 1 × 1
Output                 2 class log-probabilities
```

The notebook contains a detailed derivation of these dimensions and the parameter count.

## Experimental results

The following are the **recorded results in the supplied notebook**. Because the original experiment does not set global random seeds, exact values may vary if the notebook is retrained.

| Experiment | Main modification | Recorded result |
|---|---|---:|
| Baseline | CNN only | ~62.3% final test accuracy |
| Data augmentation | Appearance + geometric augmentation | ~82.0% final test accuracy |
| Batch normalization | `BatchNorm2d` after convolution blocks | ~85.0% best test accuracy |
| Batch normalization + L1 | \(\lambda=10^{-4}\) | ~87.6% best test accuracy |


The notebook measures **accuracy** :

\[
\frac{TP+TN}{TP+TN+FP+FN},
\]

## Dataset

The supplied notebook reports:

- **Training:** 5,232 images
  - NORMAL: 1,349
  - PNEUMONIA: 3,883
- **Test:** 625 images
  - NORMAL: 235
  - PNEUMONIA: 390

The classes are therefore imbalanced. Consequently, accuracy should not be considered sufficient for evaluating a medical-imaging model.

The project demonstrates an ability to move between mathematical abstraction and implementation :

- **Linear algebra:** convolutional operators and channel transformations;
- **Multivariable calculus:** gradients and backpropagation;
- **Probability/statistics:** empirical risk, mini-batch sampling, normalization, generalization;
- **Optimization:** SGD, momentum, learning-rate schedules;
- **Functional viewpoint:** composition of parameterized maps;
- **Norms and regularization:** L1 penalty and sparsity;
- **Numerical computing:** tensor operations and GPU/accelerator execution;
- **Experimental methodology:** controlled architectural comparisons.

The goal is therefore not simply to use PyTorch APIs, but to understand what mathematical operations those APIs represent.

## How to run

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Install dependencies

A typical environment can be created with:

```bash
pip install torch torchvision numpy matplotlib opencv-python tqdm torchsummary jupyter
```

### 3. Prepare the dataset

Place the dataset in the directory structure expected by `ImageFolder`:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Update the `data_path` variable in the notebook to point to the dataset location on your machine.

### 4. Launch the notebook

```bash
jupyter notebook Classification_lungs.ipynb
```

or open the notebook with JupyterLab / VS Code.

## Important limitations and future improvements

This project is an educational/research demonstration and **is not a clinical diagnostic system**.

A stronger production-quality study would add:

- patient-level train/validation/test splitting;
- leakage checks;
- class-weighted or focal loss;
- sensitivity and specificity;
- ROC-AUC and PR-AUC;
- confusion matrices;
- calibration analysis;
- external validation;
- cross-validation;
- uncertainty estimation;
- explainability methods such as Grad-CAM;
- systematic hyperparameter search;
- reproducibility through fixed random seeds;
- comparison with transfer learning using established CNN backbones.

These improvements would make the project substantially closer to a rigorous applied medical-AI workflow.

## Author

**Chamir Ngandjia**  
Mathematics / Machine Learning & Computer Vision
