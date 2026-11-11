# Lab Report 4: Neural Network Activation Functions Visualizer
---

## 📖 Overview

An activation function in an artificial neural network applies a non-linear transformation to a neuron's net linear combination:
$$z = \mathbf{w} \cdot \mathbf{x} + b = \sum_{i=1}^n w_i x_i + b$$

Without non-linear activation functions, multi-layer networks collapse mathematically into simple linear regression models regardless of network depth. In addition, activation functions govern neuron firing and provide essential gradients ($\frac{\partial f}{\partial z}$) required by optimization algorithms during backpropagation.

This interactive Streamlit web application provides visualization and mathematical analysis for three key activation functions:
1. **Rectified Linear Unit (ReLU)**
2. **Sigmoid (Logistic Function)**
3. **Hyperbolic Tangent (Tanh)**

---

## ⚙️ Implemented Activation Functions

### 1. Rectified Linear Unit (ReLU)
- **Activation Function:**
  $$f(z) = \max(0, z)$$
- **First-Order Derivative:**
  $$f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}$$
- **Characteristics:**
  - Range: $[0, \infty)$
  - Mitigates vanishing gradient issues for positive inputs ($z > 0$).
  - Susceptible to the *Dying ReLU* problem when inputs remain negative ($z \le 0$).

---

### 2. Sigmoid (Logistic) Function
- **Activation Function:**
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **First-Order Derivative:**
  $$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$
- **Characteristics:**
  - Range: $(0, 1)$
  - Smooth, continuously differentiable curve ideal for binary classification outputs.
  - Suffers from vanishing gradients for extreme values ($\vert{}z\vert{} > 4$) where maximum derivative reaches at most $0.25$.

---

### 3. Hyperbolic Tangent (Tanh)
- **Activation Function:**
  $$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
- **First-Order Derivative:**
  $$\tanh'(z) = 1 - \tanh^2(z)$$
- **Characteristics:**
  - Range: $(-1, 1)$
  - Zero-centered output, which prevents bias shifts in downstream weight updates and accelerates optimization over Sigmoid.
  - Still vulnerable to gradient saturation and vanishing gradients at tails ($\vert{}z\vert{} > 3$).

---
