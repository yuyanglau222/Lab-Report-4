import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Visualizer", page_icon="📈", layout="centered")

# Removed "1." from title
st.title("📈 Rectified Linear Unit (ReLU)")
st.markdown(
    """
    **Mathematical Definition:**
    $$f(z) = \max(0, z)$$
    **First Derivative:**
    $$f'(z) = \\begin{cases} 1 & \\text{if } z > 0 \\\\ 0 & \\text{if } z < 0 \\end{cases}$$
    """
)

col1, col2 = st.columns(2)
with col1:
    x_range = st.slider("X-axis Span ($\pm z$)", min_value=2.0, max_value=20.0, value=6.0, step=1.0)
with col2:
    test_z = st.number_input("Test Input Value ($z$)", value=2.5, step=0.5)

# Vector computation
z = np.linspace(-x_range, x_range, 500)
relu_out = np.maximum(0, z)
relu_grad = np.where(z > 0, 1.0, 0.0)

# Point evaluation
val_out = max(0.0, float(test_z))
val_grad = 1.0 if test_z > 0 else 0.0

col_m1, col_m2 = st.columns(2)
col_m1.metric(f"Output: f({test_z})", f"{val_out:.4f}")
col_m2.metric(f"Derivative: f'({test_z})", f"{val_grad:.1f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(z, relu_out, label=r"ReLU: $\max(0, z)$", color="#1f77b4", linewidth=2.5)
ax.plot(z, relu_grad, label=r"Derivative: $f'(z)$", color="#d62728", linestyle="--", linewidth=1.8)
ax.scatter([test_z], [val_out], color="black", s=40, zorder=5, label=f"Point ({test_z:.1f}, {val_out:.2f})")

ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
ax.set_xlabel("Input ($z$)")
ax.set_ylabel("Activation / Gradient")
ax.set_title("ReLU Function and First Derivative")
ax.grid(True, linestyle="--", alpha=0.5)

# Compact, non-intrusive legend
ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

st.pyplot(fig)

with st.expander("Analysis & Properties"):
    st.markdown(
        """
        - **Computational Complexity:** $O(1)$ operations via hardware thresholding.
        - **Strengths:** Eliminates gradient saturation for all positive inputs ($z > 0$), expediting convergence.
        - **Limitations:** Suffers from the *Dying ReLU* phenomenon when gradients remain zero for negative activations ($z \le 0$).
        """
    )
