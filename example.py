# example.py
import numpy as np
import matplotlib.pyplot as plt
from snakes import (
    circle_init,
    multiscale_active_contour,
    edge_strength_map,
)

# Load image
try:
    from skimage import data
    img = data.coins().astype(np.float64) / 255.0
except Exception:
    img = np.zeros((256, 256), dtype=np.float64)
    Y, X = np.ogrid[:256, :256]
    img[((X-128)**2/70**2 + (Y-128)**2/50**2) <= 1.0] = 1.0

# Init snake (center one coin)
init = circle_init(cx=160, cy=100, r=46, n=220)   # slightly larger + a few more points

schedule = [
    {"sigma": 5.0, "max_iters": 220, "w_term": 0.00},
    {"sigma": 3.0, "max_iters": 240, "w_term": 0.06},
    {"sigma": 1.4, "max_iters": 260, "w_term": 0.12},
    {"sigma": 0.8, "max_iters": 320, "w_term": 0.15},
]

snake, _ = multiscale_active_contour(
    img, init, schedule,
    alpha=0.02,  # ↓ tension (less shrinkage)
    beta =8.0,   # ↑ bending (smoother)
    gamma=0.8,   # slightly smaller step
    w_edge=1.0, w_line=0.0, w_term=0.0,   # per-stage w_term comes from schedule
    closed=True, resample_every=18, conv_tol=3e-4, verbose=False
)

# Panels: original, edge-strength, overlay
edge_strength = edge_strength_map(img, sigma=1.5)

fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
axs[0].imshow(img, cmap="gray"); axs[0].set_title("Original"); axs[0].axis("off")

im = axs[1].imshow(edge_strength, cmap="magma")
axs[1].set_title(r"Edge strength $\|\nabla(G_\sigma * I)\|^2$")
axs[1].axis("off")
fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

axs[2].imshow(img, cmap="gray")
axs[2].plot(init[:,0], init[:,1], '--r', lw=1, label="init")
axs[2].plot(snake[:,0], snake[:,1], '-g', lw=2, label="snake")
axs[2].legend(); axs[2].axis("off"); axs[2].set_title("Overlay")

plt.tight_layout()
plt.savefig("snake_multiscale.png")
print("Saved snake_multiscale.png")
