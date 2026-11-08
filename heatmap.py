import sys, numpy as np, pandas as pd, openslide, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sid, tag = sys.argv[1], sys.argv[2]            # e.g. test_001 feats_multi
r = pd.read_csv("manifest.csv").set_index("slide_id").loc[sid]
s = openslide.OpenSlide(r.path)
meta = np.load(f"patches/{sid}.npz"); ds = float(meta["mds"])
lvl = s.get_best_level_for_downsample(32)
thumb = np.array(s.read_region((0, 0), lvl, s.level_dimensions[lvl]).convert("RGB"))
a = np.load(f"results/attn_{tag}/{sid}.npy"); a = a / a.max()
heat, sz = np.zeros(thumb.shape[:2], np.float32), max(1, int(meta["size0"] / ds))
for (x, y), w in zip(meta["coords"], a):
    heat[int(y / ds):int(y / ds) + sz, int(x / ds):int(x / ds) + sz] = w
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(thumb); ax[0].set_title("slide")
ax[1].imshow(thumb); ax[1].imshow(heat, alpha=0.5, cmap="jet"); ax[1].set_title("attention")
ax[2].imshow(meta["tumor"], cmap="gray"); ax[2].set_title("annotation")
for x in ax: x.axis("off")
plt.tight_layout(); plt.savefig(f"results/heatmap_{sid}_{tag}.png", dpi=150)
