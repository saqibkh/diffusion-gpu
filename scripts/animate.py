import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

files = sorted(glob.glob("frames/frame_*.csv"))
if not files:
    raise SystemExit("No frames found. Run ./build/gradient first.")

frames = [np.loadtxt(f, delimiter=",") for f in files]
vmax = max(fr.max() for fr in frames)

fig, ax = plt.subplots(figsize=(10, 3.5))
im = ax.imshow(frames[0].T, origin="lower", aspect="auto",
               cmap="viridis", vmin=0, vmax=vmax)
ax.set_xlabel("along channel")
ax.set_ylabel("across channel")
ax.set_title("Microfluidic gradient formation")
fig.colorbar(im, label="concentration")
fig.tight_layout()

def update(k):
    im.set_data(frames[k].T)
    return [im]

anim = FuncAnimation(fig, update, frames=len(frames), interval=60, blit=True)
anim.save("docs/gradient.gif", writer=PillowWriter(fps=15))
print(f"wrote docs/gradient.gif ({len(frames)} frames)")