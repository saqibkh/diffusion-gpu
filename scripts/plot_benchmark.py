import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

n, naive, tiled = [], [], []
with open("benchmark.csv") as f:
    for row in csv.DictReader(f):
        n.append(int(row["n"]))
        naive.append(float(row["speedup"]))
        tiled.append(float(row["tiled_speedup"]))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(n, naive, "o-", label="naive kernel")
ax.plot(n, tiled, "s--", label="shared-memory tiled")
ax.axhline(1.0, color="gray", linestyle="--", label="break-even")
ax.set_xscale("log", base=2)
ax.set_xlabel("grid size (N x N)")
ax.set_ylabel("speedup vs single-threaded CPU")
ax.set_title("GPU speedup vs grid size — RTX 4050, FP64")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("docs/speedup.png", dpi=150)
print("wrote docs/speedup.png")