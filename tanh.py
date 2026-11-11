import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

st.set_page_config(
    page_title="Tanh",
    page_icon="〰️",
    layout="wide"
)

# ---------------------------------------------------------
# Sidebar: Controls & Parameters
# ---------------------------------------------------------
st.sidebar.subheader("Curve")
test_x = st.sidebar.slider("Input value x", min_value=-6.0, max_value=6.0, value=1.0, step=0.05)

st.sidebar.subheader("Neural Network Demo")
st.sidebar.caption("Dataset")
dataset_type = st.sidebar.selectbox("Dataset", ["Moons", "Circles"], label_visibility="collapsed")
samples = st.sidebar.slider("Samples", min_value=100, max_value=1000, value=300, step=50)
noise = st.sidebar.slider("Noise", min_value=0.00, max_value=0.50, value=0.20, step=0.01)

st.sidebar.subheader("Model Parameters")
hidden_input = st.sidebar.text_input("Hidden layer sizes (e.g. 16,16)", value="16,16")
alpha = st.sidebar.slider("L2 regularisation (alpha)", min_value=0.0001, max_value=0.01, value=0.0005, step=0.0001, format="%.4f")
learning_rate = st.sidebar.slider("Learning rate", min_value=0.001, max_value=0.1, value=0.01, step=0.005)
max_iter = st.sidebar.slider("Max iterations (epochs)", min_value=100, max_value=3000, value=800, step=100)
random_seed = st.sidebar.number_input("Random seed", value=42, step=1)

# Parse hidden layer sizes safely
try:
    hidden_layer_sizes = tuple(int(i.strip()) for i in hidden_input.split(",") if i.strip().isdigit())
    if len(hidden_layer_sizes) == 0:
        hidden_layer_sizes = (16, 16)
except Exception:
    hidden_layer_sizes = (16, 16)

# ---------------------------------------------------------
# Main Page: Title, Description & Equations
# ---------------------------------------------------------
st.title("Hyperbolic Tangent (Tanh) Function")
st.write(
    "Tanh is a smooth, continuous, S-shaped activation function that maps input values into the range between -1 and 1. "
    "Because it is zero-centered, it generally enables faster convergence during training compared to standard logistic sigmoid."
)

st.latex(r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}")
st.latex(r"\tanh'(x) = 1 - \tanh^2(x)")

# Computations for user test input point
point_out = np.tanh(test_x)
point_grad = 1.0 - (point_out ** 2)

# Output card & Theoretical properties
with st.container(border=True):
    st.subheader("Output for your input")
    col1, col2, col3 = st.columns(3)
    col1.metric("x", f"{test_x:.2f}")
    col2.metric("tanh(x)", f"{point_out:.4f}")
    col3.metric("tanh'(x)", f"{point_grad:.4f}")

    st.subheader("Properties")
    st.markdown(
        """
        - Range: $(-1, 1)$
        - Zero-centered output prevents systematic directional bias in downstream weight updates
        - Stronger gradients than Sigmoid (maximum derivative of $1.0$ at $x = 0$)
        - Still vulnerable to vanishing gradients when inputs saturate ($|x| > 3$)
        """
    )

st.divider()

# ---------------------------------------------------------
# Section 1: Activation Function Curve & First Derivative Plot (Big Square)
# ---------------------------------------------------------
st.subheader("1. Tanh Function Curve & First Derivative")

plot_limit = max(6.0, abs(test_x) + 1.0)
x_vals = np.linspace(-plot_limit, plot_limit, 500)
tanh_vals = np.tanh(x_vals)
tanh_grads = 1.0 - (tanh_vals ** 2)

# Square 6x6 figure size with tight framing
fig_act, ax_act = plt.subplots(figsize=(6, 6))
ax_act.plot(x_vals, tanh_vals, label=r"Tanh: $\tanh(x)$", color="#9467bd", linewidth=2.2)
ax_act.plot(x_vals, tanh_grads, label=r"Derivative: $1 - \tanh^2(x)$", color="#d62728", linestyle="--", linewidth=1.6)
ax_act.scatter([test_x], [point_out], color="black", s=45, zorder=5, label=f"Point ({test_x:.2f}, {point_out:.2f})")

ax_act.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax_act.axhline(1, color="gray", linewidth=0.8, linestyle=":")
ax_act.axhline(-1, color="gray", linewidth=0.8, linestyle=":")
ax_act.axvline(0, color="gray", linewidth=0.8, linestyle=":")
ax_act.set_xlim(-plot_limit, plot_limit)
ax_act.set_ylim(-1.15, 1.15)

ax_act.set_xlabel("Input ($x$)", fontsize=11)
ax_act.set_ylabel("Activation / Gradient", fontsize=11)
ax_act.set_title("Tanh Function and First Derivative", fontsize=12, pad=10)
ax_act.grid(True, linestyle="--", alpha=0.5)

# Strictly constrained, fixed-size legend
ax_act.legend(
    loc="lower right",
    fontsize=9,
    markerscale=0.7,
    handlelength=1.6,
    borderpad=0.4,
    labelspacing=0.3,
    framealpha=0.9
)
fig_act.tight_layout()

col_plot = st.columns([1, 2, 1])
with col_plot[1]:
    st.pyplot(fig_act)

st.divider()

# ---------------------------------------------------------
# Section 2: Neural Network using Tanh (Decision Boundary)
# ---------------------------------------------------------
st.subheader("2. Neural Network using Tanh")
st.write(
    "A Multi-Layer Perceptron (MLP) is trained on a toy 2-D dataset. "
    "Every hidden layer uses the **Tanh** activation, so the decision boundary below shows what this activation lets the network learn."
)

if dataset_type == "Moons":
    X, y = make_moons(n_samples=samples, noise=noise, random_state=int(random_seed))
else:
    X, y = make_circles(n_samples=samples, noise=noise, factor=0.5, random_state=int(random_seed))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation="tanh",
    alpha=alpha,
    learning_rate_init=learning_rate,
    max_iter=max_iter,
    random_state=int(random_seed)
)
clf.fit(X_scaled, y)
acc = clf.score(X_scaled, y)

x_pad = 0.5
y_pad = 0.5
x_min, x_max = X_scaled[:, 0].min() - x_pad, X_scaled[:, 0].max() + x_pad
y_min, y_max = X_scaled[:, 1].min() - y_pad, X_scaled[:, 1].max() + y_pad
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig_bound, ax_bound = plt.subplots(figsize=(6, 5))
cmap_bg = ListedColormap(["#F0EEF8", "#E8F6F6"])
ax_bound.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.8)

colors = np.where(y == 0, "#6a0dad", "#008080")
ax_bound.scatter(X_scaled[:, 0], X_scaled[:, 1], c=colors, edgecolor="black", s=25, linewidth=0.7)

ax_bound.set_title("MLP with Tanh", fontsize=12)
ax_bound.set_xlabel("x1")
ax_bound.set_ylabel("x2")
fig_bound.tight_layout()

col_center = st.columns([1, 2, 1])
with col_center[1]:
    st.pyplot(fig_bound)
    st.caption(f"Model Training Accuracy: **{acc * 100:.2f}%**")
