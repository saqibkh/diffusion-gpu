import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

n, dx, err = [], [], []
with open("validation.csv") as f:
    for row in csv.DictReader(f):
        n.append(int(row["n"]))
        dx.append(float(row["dx"]))
        err.append(float(row["l2_error"]))

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(dx, err, "o-", label="measured")
ref = [err[0] * (d / dx[0]) ** 2 for d in dx]
ax.loglog(dx, ref, "--", label="2nd-order reference")
ax.set_xlabel("grid spacing dx")
ax.set_ylabel("L2 error")
ax.set_title("Grid convergence — 2D diffusion solver")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("docs/convergence.png", dpi=150)
print("wrote docs/convergence.png")