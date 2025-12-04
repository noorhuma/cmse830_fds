from src.evaluate import plot_confusion
import matplotlib.pyplot as plt
import os

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

y_true = [0, 1, 0, 1, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1]

# Generate figure
fig = plot_confusion(y_true, y_pred, title="Test Confusion Matrix")

# Save it here
output_path = os.path.abspath("outputs/test_confusion.png")
fig.savefig(output_path, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved: {output_path}")
