import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Data
# =========================

apple_domain = np.array([
    [0.0000, 0.2591, 0.3050],
    [0.2609, 0.0000, 0.0807],
    [0.3088, 0.0743, 0.0000]
])

apple_class = np.array([
    [0.0000, 0.1307],
    [0.1329, 0.0000]
])

tomato_domain = np.array([
    [0.0000, 0.2436, 0.2774],
    [0.2418, 0.0000, 0.0209],
    [0.2722, 0.0209, 0.0000]
])

tomato_class = np.array([
    [0.0000, 0.0881],
    [0.0900, 0.0000]
])

domain_labels = ["PlantVillage", "PlantDoc", "PlantWild"]
class_labels = ["Healthy", "Diseased"]

# =========================
# Common color scale
# =========================

all_values = np.concatenate([
    apple_domain.flatten(),
    apple_class.flatten(),
    tomato_domain.flatten(),
    tomato_class.flatten()
])

vmin = 0
vmax = np.max(all_values)

# =========================
# Apple Domain
# =========================

plt.figure(figsize=(6, 5))
sns.heatmap(
    apple_domain,
    annot=True,
    fmt=".4f",
    cmap="viridis",
    xticklabels=domain_labels,
    yticklabels=domain_labels,
    vmin=vmin,
    vmax=vmax
)
plt.title("Apple Domain Distance")
plt.tight_layout()
plt.savefig("apple_domain_distance.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# Apple Class
# =========================

plt.figure(figsize=(5, 4))
sns.heatmap(
    apple_class,
    annot=True,
    fmt=".4f",
    cmap="viridis",
    xticklabels=class_labels,
    yticklabels=class_labels,
    vmin=vmin,
    vmax=vmax
)
plt.title("Apple Class Distance")
plt.tight_layout()
plt.savefig("apple_class_distance.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# Tomato Domain
# =========================

plt.figure(figsize=(6, 5))
sns.heatmap(
    tomato_domain,
    annot=True,
    fmt=".4f",
    cmap="viridis",
    xticklabels=domain_labels,
    yticklabels=domain_labels,
    vmin=vmin,
    vmax=vmax
)
plt.title("Tomato Domain Distance")
plt.tight_layout()
plt.savefig("tomato_domain_distance.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# Tomato Class
# =========================

plt.figure(figsize=(5, 4))
sns.heatmap(
    tomato_class,
    annot=True,
    fmt=".4f",
    cmap="viridis",
    xticklabels=class_labels,
    yticklabels=class_labels,
    vmin=vmin,
    vmax=vmax
)
plt.title("Tomato Class Distance")
plt.tight_layout()
plt.savefig("tomato_class_distance.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved:")
print(" - apple_domain_distance.png")
print(" - apple_class_distance.png")
print(" - tomato_domain_distance.png")
print(" - tomato_class_distance.png")