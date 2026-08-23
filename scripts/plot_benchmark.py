import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

n, speedup = [], []
with open("benchmark.csv") as f:
    for row in csv.DictReader(f):
        n.append(int(row["n"]))
        speedup.append(float(row["speedup"]))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(n, speedup, "o-")
ax.axhline(1.0, color="gray", linestyle="--", label="break-even")
ax.set_xscale("log", base=2)
ax.set_xlabel("grid size (N x N)")
ax.set_ylabel("speedup (CPU time / GPU time)")
ax.set_title("GPU speedup vs grid size - RTX 4050, FP64")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("docs/speedup.png", dpi=150)
print("wrote docs/speedup.png")