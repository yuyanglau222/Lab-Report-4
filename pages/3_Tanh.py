import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tanh Visualizer", page_icon="〰️", layout="centered")

# Removed "3." from the title
st.title("〰️ Hyperbolic Tangent (Tanh)")
st.markdown(
    """
    **Mathematical Definition:**
    $$\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$$
    **First Derivative:**
    $$\\tanh'(z) = 1 - \\tanh^2(z)$$
    """
)

col1, col2 = st.columns(2)
with col1:
    x_range = st.slider("X-axis Span ($\pm z$)", min_value=2.0, max_value=20.0, value=6.0, step=1.0)
with col2:
    test_z = st.number_input("Test Input Value ($z$)", value=0.0, step=0.5)

# Vector computation
z = np.linspace(-x_range, x_range, 500)
tanh_out = np.tanh(z)
tanh_grad = 1.0 - (tanh_out ** 2)

# Point evaluation
val_out = np.tanh(test_z)
val_grad = 1.0 - (val_out ** 2)

col_m1, col_m2 = st.columns(2)
col_m1.metric(f"Output: tanh({test_z})", f"{val_out:.4f}")
col_m2.metric(f"Derivative: tanh'({test_z})", f"{val_grad:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(z, tanh_out, label=r"$\tanh(z)$", color="#9467bd", linewidth=2.5)
ax.plot(z, tanh_grad, label=r"Derivative: $1 - \tanh^2(z)$", color="#d62728", linestyle="--", linewidth=1.8)
ax.scatter([test_z], [val_out], color="black", s=40, zorder=5, label=f"Point ({test_z:.1f}, {val_out:.2f})")

ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax.axhline(1, color="gray", linewidth=0.8, linestyle=":")
ax.axhline(-1, color="gray", linewidth=0.8, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")

ax.set_xlabel("Input ($z$)")
ax.set_ylabel("Activation / Gradient")
ax.set_ylim(-1.15, 1.15)
ax.set_title("Tanh Function and First Derivative")
ax.grid(True, linestyle="--", alpha=0.5)

# Compact, non-intrusive legend
ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

st.pyplot(fig)

with st.expander("Analysis & Properties"):
    st.markdown(
        """
        - **Range:** $(-1, 1)$
        - **Strengths:** Zero-centered outputs prevent bias shifts in downstream weight updates, offering faster optimization than standard sigmoid.
        - **Limitations:** Still suffers from gradient saturation when $|z| > 3$, leading to vanishing gradients in deep architectures.
        """
    )
