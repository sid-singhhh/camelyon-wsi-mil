import argparse, os
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision, openslide
import torchvision.transforms as T
from PIL import Image
from skimage.color import rgb2hed, hed2rgb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PATCH = 256

def hed_jitter(img, sigma=0.05):
    hed = rgb2hed(np.asarray(img))
    hed = hed * np.random.uniform(1 - sigma, 1 + sigma, 3) + np.random.uniform(-sigma, sigma, 3)
    out = np.clip(hed2rgb(hed), 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))

class Bag(Dataset):
    def __init__(self, path, coords, jitter):
        self.path, self.coords, self.jitter, self.slide = path, coords, jitter, None
        s = openslide.OpenSlide(path)
        self.lv = [s.get_best_level_for_downsample(d) for d in (2, 4)]   # 20x, 10x
        self.ds = [s.level_downsamples[l] for l in self.lv]
        self.tf = T.Compose([T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __len__(self): return len(self.coords)
    def __getitem__(self, i):
        if self.slide is None: self.slide = openslide.OpenSlide(self.path)  # one handle per worker
        x, y = self.coords[i]
        cx, cy = x + PATCH * self.ds[0] / 2, y + PATCH * self.ds[0] / 2     # shared center
        out = []
        for l, d in zip(self.lv, self.ds):
            half = PATCH * d / 2
            img = self.slide.read_region((int(cx - half), int(cy - half)), l, (PATCH, PATCH)).convert("RGB")
            if self.jitter: img = hed_jitter(img)
            out.append(self.tf(img))
        return torch.stack(out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="manifest.csv")
    p.add_argument("--patches", default="patches")
    p.add_argument("--out", default="feats")
    p.add_argument("--jitter", action="store_true")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    model.fc = nn.Identity()
    model = model.cuda().eval()

    df = pd.read_csv(a.manifest)
    for r in tqdm(df.itertuples(), total=len(df)):
        out = f"{a.out}/{r.slide_id}.pt"
        pf = f"{a.patches}/{r.slide_id}.npz"
        if os.path.exists(out) or not os.path.exists(pf): continue
        meta = np.load(pf)
        if len(meta["coords"]) == 0: continue
        dl = DataLoader(Bag(r.path, meta["coords"], a.jitter), batch_size=a.bs, num_workers=a.workers)
        f20, f10 = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for b in dl:
                n = b.shape[0]
                f = model(b.cuda(non_blocking=True).flatten(0, 1)).view(n, 2, -1)
                f20.append(f[:, 0].half().cpu()); f10.append(f[:, 1].half().cpu())
        torch.save({"f20": torch.cat(f20), "f10": torch.cat(f10),
                    "labels": torch.tensor(meta["labels"])}, out)
