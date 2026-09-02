import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BANDWIDTH = 192e9      # bytes/s  -- 2000 MHz GDDR6, 96-bit bus
PEAK_FP64 = 140e9      # flops/s  -- 2560 cores, 1755 MHz, FP64 at 1/64 rate

ai = np.logspace(-2, 2, 400)
roof = np.minimum(ai * BANDWIDTH, PEAK_FP64) / 1e9

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(ai, roof, label="roofline")
ax.plot(0.44, 57.5, "o", markersize=9, label="stencil kernel (1024²)")
ax.axvline(PEAK_FP64 / BANDWIDTH, color="gray", linestyle=":",
           label=f"ridge point ({PEAK_FP64 / BANDWIDTH:.2f} flops/byte)")

ax.set_xlabel("arithmetic intensity (flops/byte)")
ax.set_ylabel("performance (GFLOPS, FP64)")
ax.set_title("Roofline — RTX 4050 Laptop GPU, FP64")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("docs/roofline.png", dpi=150)
print("wrote docs/roofline.png")