import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Neural Network Activation Visualizer",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------------------------------------
# Sidebar Navigation: Dropdown selection between the 3 apps
# ---------------------------------------------------------
st.sidebar.title("Navigation")
app_choice = st.sidebar.selectbox(
    "Select Activation Function:",
    ["1. Rectified Linear Unit (ReLU)", "2. Sigmoid (Logistic)", "3. Hyperbolic Tangent (Tanh)"]
)

st.sidebar.divider()
st.sidebar.header("Neuron Linear Inputs")
x_input = st.sidebar.number_input("Input Signal (x)", value=1.0, step=0.5)
weight = st.sidebar.number_input("Synaptic Weight (w)", value=1.5, step=0.1)
bias = st.sidebar.number_input("Neuron Bias (b)", value=-0.5, step=0.5)

# Calculate net linear input z = w * x + b
test_z = (weight * x_input) + bias

x_range = st.sidebar.slider("Graph Domain Span (±z)", min_value=2.0, max_value=20.0, value=6.0, step=1.0)
z_vals = np.linspace(-x_range, x_range, 500)


# ---------------------------------------------------------
# App 1: Rectified Linear Unit (ReLU)
# ---------------------------------------------------------
if app_choice == "1. Rectified Linear Unit (ReLU)":
    st.title("📈 Rectified Linear Unit (ReLU)")
    st.markdown(
        r"""
        **Neuron Linear Transformation:**
        $$z = w \cdot x + b$$
        **Activation Function & Derivative:**
        $$f(z) = \max(0, z), \quad f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}$$
        """
    )

    # Computations
    relu_vals = np.maximum(0, z_vals)
    relu_grads = np.where(z_vals > 0, 1.0, 0.0)

    active_out = max(0.0, float(test_z))
    active_grad = 1.0 if test_z > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Net Input (z = w·x + b)", f"{test_z:.2f}")
    col2.metric("Neuron Output f(z)", f"{active_out:.4f}")
    col3.metric("Gradient f'(z)", f"{active_grad:.1f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(z_vals, relu_vals, label=r"ReLU: $\max(0, z)$", color="#1f77b4", linewidth=2.5)
    ax.plot(z_vals, relu_grads, label=r"Derivative: $f'(z)$", color="#d62728", linestyle="--", linewidth=1.8)
    ax.scatter([test_z], [active_out], color="black", s=45, zorder=5, label=f"Neuron State ({test_z:.2f}, {active_out:.2f})")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Net Weighted Input ($z$)")
    ax.set_ylabel("Activation / Gradient")
    ax.set_title("ReLU Function and First Derivative")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    st.pyplot(fig)

    with st.expander("Theoretical Properties & Complexity"):
        st.markdown(
            """
            - **Mathematical Form:** $f(z) = \max(0, z)$
            - **Range:** $[0, \infty)$
            - **Backpropagation Support:** Avoids vanishing gradients for positive inputs because $f'(z) = 1$.
            - **Limitation:** Can lead to 'Dying ReLU' if negative inputs keep neurons inactive permanently with zero gradient.
            """
        )


# ---------------------------------------------------------
# App 2: Sigmoid (Logistic)
# ---------------------------------------------------------
elif app_choice == "2. Sigmoid (Logistic)":
    st.title("📉 Sigmoid (Logistic) Function")
    st.markdown(
        r"""
        **Neuron Linear Transformation:**
        $$z = w \cdot x + b$$
        **Activation Function & Derivative:**
        $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$
        """
    )

    # Computations
    sig_vals = 1.0 / (1.0 + np.exp(-z_vals))
    sig_grads = sig_vals * (1.0 - sig_vals)

    active_out = 1.0 / (1.0 + np.exp(-test_z))
    active_grad = active_out * (1.0 - active_out)

    col1, col2, col3 = st.columns(3)
    col1.metric("Net Input (z = w·x + b)", f"{test_z:.2f}")
    col2.metric("Neuron Output σ(z)", f"{active_out:.4f}")
    col3.metric("Gradient σ'(z)", f"{active_grad:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(z_vals, sig_vals, label=r"Sigmoid: $\sigma(z)$", color="#2ca02c", linewidth=2.5)
    ax.plot(z_vals, sig_grads, label=r"Derivative: $\sigma'(z)$", color="#d62728", linestyle="--", linewidth=1.8)
    ax.scatter([test_z], [active_out], color="black", s=45, zorder=5, label=f"Neuron State ({test_z:.2f}, {active_out:.2f})")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(1, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Net Weighted Input ($z$)")
    ax.set_ylabel("Activation / Gradient")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Sigmoid Function and First Derivative")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    st.pyplot(fig)

    with st.expander("Theoretical Properties & Complexity"):
        st.markdown(
            """
            - **Mathematical Form:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
            - **Range:** $(0, 1)$
            - **Backpropagation Support:** Continuously differentiable, yielding smooth gradients suitable for binary cross-entropy.
            - **Limitation:** Vanishing gradient problem occurs at extreme inputs ($|z| > 4$) where maximum gradient drops below $0.25$.
            """
        )


# ---------------------------------------------------------
# App 3: Hyperbolic Tangent (Tanh)
# ---------------------------------------------------------
elif app_choice == "3. Hyperbolic Tangent (Tanh)":
    st.title("〰️ Hyperbolic Tangent (Tanh)")
    st.markdown(
        r"""
        **Neuron Linear Transformation:**
        $$z = w \cdot x + b$$
        **Activation Function & Derivative:**
        $$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z)$$
        """
    )

    # Computations
    tanh_vals = np.tanh(z_vals)
    tanh_grads = 1.0 - (tanh_vals ** 2)

    active_out = np.tanh(test_z)
    active_grad = 1.0 - (active_out ** 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Net Input (z = w·x + b)", f"{test_z:.2f}")
    col2.metric("Neuron Output tanh(z)", f"{active_out:.4f}")
    col3.metric("Gradient tanh'(z)", f"{active_grad:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(z_vals, tanh_vals, label=r"$\tanh(z)$", color="#9467bd", linewidth=2.5)
    ax.plot(z_vals, tanh_grads, label=r"Derivative: $1 - \tanh^2(z)$", color="#d62728", linestyle="--", linewidth=1.8)
    ax.scatter([test_z], [active_out], color="black", s=45, zorder=5, label=f"Neuron State ({test_z:.2f}, {active_out:.2f})")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(1, color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(-1, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Net Weighted Input ($z$)")
    ax.set_ylabel("Activation / Gradient")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("Tanh Function and First Derivative")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    st.pyplot(fig)

    with st.expander("Theoretical Properties & Complexity"):
        st.markdown(
            """
            - **Mathematical Form:** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
            - **Range:** $(-1, 1)$
            - **Backpropagation Support:** Zero-centered outputs assist gradient convergence during multi-layer backpropagation.
            - **Limitation:** Gradients saturate and vanish for $|z| > 3$.
            """
        )
