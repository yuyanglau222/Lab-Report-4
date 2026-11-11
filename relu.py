import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

st.set_page_config(
    page_title="ReLU Activation Visualizer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Rectified Linear Unit (ReLU) Visualizer")
st.markdown(
    r"""
    **Neuron Linear Transformation:**
    $$z = w \cdot x + b$$
    **Activation Function & First-Order Derivative:**
    $$f(z) = \max(0, z), \quad f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}$$
    """
)

# Sidebar: Inputs & Dataset Configuration
st.sidebar.header("Neuron Linear Inputs")
x_input = st.sidebar.number_input("Input Signal (x)", value=1.0, step=0.5)
weight = st.sidebar.number_input("Synaptic Weight (w)", value=1.5, step=0.1)
bias = st.sidebar.number_input("Neuron Bias (b)", value=-0.5, step=0.5)

test_z = (weight * x_input) + bias
x_range = st.sidebar.slider("Graph Domain Span (±z)", min_value=2.0, max_value=20.0, value=6.0, step=1.0)
z_vals = np.linspace(-x_range, x_range, 500)

st.sidebar.divider()
st.sidebar.header("2D Classification Dataset")
dataset_type = st.sidebar.selectbox("Toy Dataset", ["Moons (Non-linear)", "Circles (Non-linear)"])
n_samples = st.sidebar.slider("Sample Count", 100, 1000, 300, 50)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.4, 0.15, 0.05)
max_iter = st.sidebar.slider("Max Training Epochs", 100, 1500, 500, 100)

# Calculations
relu_vals = np.maximum(0, z_vals)
relu_grads = np.where(z_vals > 0, 1.0, 0.0)
point_out = max(0.0, float(test_z))
point_grad = 1.0 if test_z > 0 else 0.0

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Net Input (z = w·x + b)", f"{test_z:.2f}")
col_m2.metric("Neuron Output f(z)", f"{point_out:.4f}")
col_m3.metric("Gradient f'(z)", f"{point_grad:.1f}")

tab1, tab2 = st.tabs(["📊 Activation & Derivative Plot", "🧪 Toy Dataset Decision Boundary"])

with tab1:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(z_vals, relu_vals, label=r"ReLU: $\max(0, z)$", color="#1f77b4", linewidth=2.5)
    ax.plot(z_vals, relu_grads, label=r"Derivative: $f'(z)$", color="#d62728", linestyle="--", linewidth=1.8)
    ax.scatter([test_z], [point_out], color="black", s=45, zorder=5, label=f"Neuron State ({test_z:.2f}, {point_out:.2f})")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Net Weighted Input ($z$)")
    ax.set_ylabel("Activation / Gradient")
    ax.set_ylim(-0.5, x_range * 0.9)
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

with tab2:
    st.markdown("#### Non-Linear Boundary Classification using `RELU`")
    if dataset_type.startswith("Moons"):
        X, y = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)
    else:
        X, y = make_circles(n_samples=n_samples, noise=noise_level, factor=0.5, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = MLPClassifier(hidden_layer_sizes=(16, 16), activation="relu", max_iter=max_iter, random_state=42)
    clf.fit(X_scaled, y)
    acc = clf.score(X_scaled, y)

    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.metric("Model Training Accuracy", f"{acc * 100:.2f}%")
        st.write("First 5 training instances:")
        st.dataframe({"x1": X_scaled[:5, 0], "x2": X_scaled[:5, 1], "class": y[:5]})

    with col_d2:
        fig_bound, ax_bound = plt.subplots(figsize=(6, 4))
        cmap_light = ListedColormap(["#FFEEEE", "#EEFFEE"])
        cmap_bold = ["#CC0000", "#007700"]
        ax_bound.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.7)
        ax_bound.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor="k", s=25)
        ax_bound.set_title(f"Decision Boundary (RELU | Acc: {acc:.2f})")
        ax_bound.set_xlabel("Feature 1 (Standardized)")
        ax_bound.set_ylabel("Feature 2 (Standardized)")
        st.pyplot(fig_bound)
