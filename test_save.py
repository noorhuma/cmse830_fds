import os
import matplotlib
matplotlib.use("Agg")  # ensure non-interactive backend
import matplotlib.pyplot as plt

# 1) Create a simple figure
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot([0, 1], [0, 1], label="Test line")
ax.legend()
ax.set_title("Save Test")

# 2) Ensure outputs dir (absolute)
OUT_DIR = os.path.abspath(os.path.join(os.getcwd(), "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)
print("Outputs directory:", OUT_DIR)

# 3) Save with absolute path
output_path = os.path.join(OUT_DIR, "save_test.png")
print("Attempting to save to:", output_path)
fig.savefig(output_path, bbox_inches="tight")
plt.close(fig)

# 4) Verify
exists = os.path.exists(output_path)
size = os.path.getsize(output_path) if exists else 0
print(f"Exists={exists} | Size={size} bytes")
