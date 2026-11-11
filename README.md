# BSD3513 Introduction to Artificial Intelligence — Lab Report 4
**Topic:** Chapter 4: Neural Networks
**Institution:** Universiti Malaysia Pahang Al-Sultan Abdullah (UMPSA)

---

## 🌐 Deployed Applications & Repository Links

| # | Activation Function | File Name | Live Streamlit Deployment Link |
|---|---------------------|-----------|--------------------------------|
| **1** | **Rectified Linear Unit (ReLU)** | `relu.py` | https://ai-lab-report4-relu.streamlit.app/ |
| **2** | **Sigmoid (Logistic)** | `sigmoid.py` | https://ai-lab-report4-sigmoid.streamlit.app/ |
| **3** | **Hyperbolic Tangent (Tanh)** | `tanh.py` | https://ai-lab-report4-tanh.streamlit.app/ |

---

## 📖 Project Overview

An activation function in a neural network is a mathematical transformation applied to a neuron's output. It introduces non-linearity, allowing the model to acquire and depict intricate data patterns. In its absence, a multi-layer neural network collapses into a basic linear regression model, regardless of depth. Furthermore, activation functions govern neuron firing and facilitate backpropagation by providing gradients for weight optimization.

This repository provides three standalone Streamlit applications corresponding to the primary activation functions evaluated:
1. **Rectified Linear Unit (ReLU)**
2. **Sigmoid (Logistic Function)**
3. **Hyperbolic Tangent (Tanh)**

Each application consists of two core components:
* **Interactive Activation & Derivative Visualizer:** Dynamic calculation and curve plotting of the function alongside its first-order derivative across an adjustable input domain.
* **Toy 2D Classification Demo:** Real-time training of a Multi-Layer Perceptron (MLP) on non-linear synthetic benchmarks (`Moons` and `Circles`) to illustrate the learned decision boundary.

---

## 🔬 Mathematical Formulations & Properties

### 1. Rectified Linear Unit (ReLU)
* **Equation:**
  $$f(x) = \max(0, x)$$
* **First Derivative:**
  $$f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}$$
* **Range:** $[0, \infty)$
* **Key Characteristics:**
  * Computationally efficient and simple to calculate.
  * Avoids gradient saturation for positive activations, speeding up convergence.
  * Prone to the *Dying ReLU* phenomenon when sustained negative updates result in a persistent zero gradient.

---

### 2. Sigmoid (Logistic Function)
* **Equation:**
  $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
* **First Derivative:**
  $$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$
* **Range:** $(0, 1)$
* **Key Characteristics:**
  * Smooth, continuously differentiable S-curve suitable for binary probabilistic modeling.
  * Subject to the *vanishing gradient problem* at outer ranges ($|x| > 4$) where the maximum derivative drops to $\le 0.25$.

---

### 3. Hyperbolic Tangent (Tanh)
* **Equation:**
  $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
* **First Derivative:**
  $$\tanh'(x) = 1 - \tanh^2(x)$$
* **Range:** $(-1, 1)$
* **Key Characteristics:**
  * Zero-centered output prevents directional bias shifts during gradient descent updates.
  * Offers steeper gradients than Sigmoid (maximum gradient of $1.0$ at $x = 0$), though tail saturation remains an issue for $|x| > 3$.

---

## 📁 Repository Directory Structure

```text
Lab-Report-4/
├── relu.py              # Application 1: ReLU visualizer & MLP demo
├── sigmoid.py           # Application 2: Sigmoid visualizer & MLP demo
├── tanh.py              # Application 3: Tanh visualizer & MLP demo
├── requirements.txt     # Python dependencies for Streamlit Cloud
└── README.md            # Lab report documentation & live links
```

## 📦 Requirements (`requirements.txt`)

```text
streamlit>=1.30.0
numpy>=1.24.0
matplotlib>=3.8.0
scikit-learn>=1.3.0
```
