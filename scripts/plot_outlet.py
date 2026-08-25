import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

j, c = [], []
with open("outlet_profile.csv") as f:
    for row in csv.DictReader(f):
        j.append(int(row["j"]))
        c.append(float(row["concentration"]))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(j, c, "o-", markersize=3)
ax.set_xlabel("position across channel")
ax.set_ylabel("concentration")
ax.set_title("Concentration gradient at channel outlet")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("docs/outlet_profile.png", dpi=150)
print("wrote docs/outlet_profile.png")