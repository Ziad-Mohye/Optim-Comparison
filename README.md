# Optimization Comparison on CIFAR-10

This project benchmarks and analyzes the performance of various optimization algorithms for training a shallow neural network on the CIFAR-10 image classification task. The goal is to evaluate convergence behavior, generalization performance, and computational efficiency across optimizers.

## 📌 Project Overview

Modern deep learning heavily relies on optimization algorithms to train models efficiently. In this project, we implemented and compared the following optimizers:

- **Stochastic Gradient Descent with Warm Restarts (SGDR)**
- **Nesterov Accelerated Gradient (NAG)**
- **RMSProp**
- **Nadam**
- **Exponential Decay**
- **Step Decay**

The experiments were conducted on a shallow feedforward neural network using the **CIFAR-10** dataset to highlight optimizer behavior without the added complexity of deep models.

---

## 🔧 Tech Stack & Tools

- **Python 3.10+**
- **TensorFlow / Keras** – Neural network implementation & training
- **NumPy** – Numerical computation
- **Matplotlib & Seaborn** – Plotting and visualizations
- **Scikit-learn** – Data preprocessing & evaluation metrics
- **Google Colab / Jupyter Notebooks** – Interactive development environment

---

## 📊 Dataset

- **CIFAR-10**: A standard benchmark dataset consisting of 60,000 32×32 color images across 10 classes.
- Split: **70% train**, **15% validation**, **15% test**
- Preprocessing: Normalization to [0,1], one-hot encoded labels.

---

## ⚙️ Optimizers Implemented

| Optimizer                | Key Feature                                      |
|--------------------------|--------------------------------------------------|
| SGDR                     | Periodic restarts using cosine annealing         |
| NAG                      | Momentum with lookahead gradient                 |
| RMSProp                  | Adaptive learning rate using gradient history    |
| Nadam                   | Combines Adam and Nesterov momentum              |
| Exponential Decay        | Reduces learning rate smoothly over time        |
| Step Decay               | Sharp drops in learning rate at fixed intervals  |

---

## 📈 Evaluation Metrics

- **Training/Validation/Test Accuracy**
- **Training/Validation Loss Curves**
- **Confusion Matrices**
- **Overfitting Indicators (Loss Divergence, Accuracy Gap)**

---

## 🧪 Key Results

| Optimizer      | Test Accuracy | Notes                                       |
|----------------|---------------|---------------------------------------------|
| SGDR           | ~49%          | Moderate performance; benefits from restarts |
| NAG            | ~51%          | Fast convergence; minor overfitting         |
| RMSProp        | ~49%          | Stable but slower convergence               |
| Nadam          | ~52%          | Best accuracy; smooth learning              |
| Exp Decay      | ~49%          | Sensitive to decay rate                     |
| Step Decay     | ~49%          | Prone to instability if decay step not tuned |

---

## 📉 Visualizations

- **Loss vs Epochs** for each optimizer
- **Accuracy vs Epochs**
- **Confusion Matrices** for classification results
- **Optimizer behavior trends** (e.g., convergence speed, overfitting)

---

## 🧠 Insights & Learnings

- Optimizer performance is highly dependent on network depth, architecture, and learning rate configuration.
- Nadam consistently offered better convergence and generalization across runs.
- SGDR helped escape local minima due to periodic restarts but took longer to converge.
- Shallow networks are limited in learning spatial patterns from image data — adding convolutional layers is recommended for future work.

