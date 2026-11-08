import streamlit as st

st.set_page_config(
    page_title="Neural Network Activations Visualizer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Neural Network Activation Functions Visualizer")
st.markdown("### BSD3513 Introduction to Artificial Intelligence — Lab Report 4")

st.markdown(
    """
    An activation function performs a non-linear mathematical transformation on a neuron's net weighted input:
    $$z = \sum_{i=1}^{n} w_i x_i + b$$
    
    Without non-linear activation functions, any deep neural network collapses mathematically into a single linear regression operation, regardless of depth.
    
    ---
    ### Select an Activation Function from the Sidebar:
    1. **1_ReLU**: Rectified Linear Unit ($f(z) = \max(0, z)$)
    2. **2_Sigmoid**: Logistic Sigmoid ($\sigma(z) = \\frac{1}{1 + e^{-z}}$)
    3. **3_Tanh**: Hyperbolic Tangent ($\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$)
    
    Each page provides interactive mathematical graphs, first-order derivative plots for backpropagation gradient flow, and parameter inspection.
    """
)

# Comparison summary table
st.subheader("Summary Comparison")
st.markdown(
    """
| Function | Mathematical Formula | Output Range | Zero-Centered? | Common Primary Use |
| :--- | :--- | :--- | :--- | :--- |
| **ReLU** | $f(z) = \max(0, z)$ | $[0, \infty)$ | No | Hidden layers in deep networks / CNNs |
| **Sigmoid** | $\sigma(z) = \\frac{1}{1 + e^{-z}}$ | $(0, 1)$ | No | Output layer for binary classification |
| **Tanh** | $\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | Yes | Hidden layers in RNNs / zero-centered tasks |
    """
)
