import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Visualizer", page_icon="📉", layout="centered")

# Removed "2." from title
st.title("📉 Sigmoid (Logistic) Function")
st.markdown(
    """
    **Mathematical Definition:**
    $$\sigma(z) = \\frac{1}{1 + e^{-z}}$$
    **First Derivative:**
    $$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$
    """
)

col1, col2 = st.columns(2)
with col1:
    x_range = st.slider("X-axis Span ($\pm z$)", min_value=2.0, max_value=20.0, value=6.0, step=1.0)
with col2:
    test_z = st.number_input("Test Input Value ($z$)", value=0.0, step=0.5)

# Vector computation
z = np.linspace(-x_range, x_range, 500)
sig_out = 1.0 / (1.0 + np.exp(-z))
sig_grad = sig_out * (1.0 - sig_out)

# Point evaluation
val_out = 1.0 / (1.0 + np.exp(-test_z))
val_grad = val_out * (1.0 - val_out)

col_m1, col_m2 = st.columns(2)
col_m1.metric(f"Output: $\sigma$({test_z})", f"{val_out:.4f}")
col_m2.metric(f"Derivative: $\sigma'$({test_z})", f"{val_grad:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(z, sig_out, label=r"$\sigma(z)$", color="#2ca02c", linewidth=2.5)
ax.plot(z, sig_grad, label=r"Derivative: $\sigma'(z)$", color="#d62728", linestyle="--", linewidth=1.8)
ax.scatter([test_z], [val_out], color="black", s=40, zorder=5, label=f"Point ({test_z:.1f}, {val_out:.2f})")

ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax.axhline(1, color="gray", linewidth=0.8, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
ax.set_xlabel("Input ($z$)")
ax.set_ylabel("Activation / Gradient")
ax.set_ylim(-0.1, 1.1)
ax.set_title("Sigmoid Function and First Derivative")
ax.grid(True, linestyle="--", alpha=0.5)

# Compact, non-intrusive legend
ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

st.pyplot(fig)

with st.expander("Analysis & Properties"):
    st.markdown(
        """
        - **Range:** $(0, 1)$
        - **Strengths:** Smooth, continuous mapping; outputs translate directly into class probabilities.
        - **Limitations:** Maximum derivative is only $0.25$ at $z=0$. In deep networks, multiplying these fractional values causes the vanishing gradient problem.
        """
    )
