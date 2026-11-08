import argparse, os
import numpy as np, cv2, openslide, pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm

PATCH = 256

def tissue_mask(img):
    s = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]
    s = cv2.GaussianBlur(s, (5, 5), 0)
    _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m > 0

def tumor_mask(xml_path, shape, ds):
    m = np.zeros(shape, np.uint8)
    if not isinstance(xml_path, str) or not xml_path:
        return m
    tumor, excl = [], []
    for ann in ET.parse(xml_path).getroot().iter("Annotation"):
        pts = np.array([[float(c.get("X")), float(c.get("Y"))]
                        for c in ann.iter("Coordinate")]) / ds
        if len(pts) < 3: continue
        (excl if ann.get("PartOfGroup") == "_2" else tumor).append(pts.astype(np.int32))
    if tumor: cv2.fillPoly(m, tumor, 1)
    if excl:  cv2.fillPoly(m, excl, 0)
    return m

def process(r, a):
    slide = openslide.OpenSlide(r.path)
    ml = slide.get_best_level_for_downsample(32)
    mds = slide.level_downsamples[ml]
    W, H = slide.level_dimensions[ml]
    img = np.array(slide.read_region((0, 0), ml, (W, H)).convert("RGB"))
    tis = tissue_mask(img)
    tum = tumor_mask(r.xml, tis.shape, mds)
    l1 = slide.get_best_level_for_downsample(2)          # 20x, assuming level 0 is 40x
    ds1 = slide.level_downsamples[l1]
    size0 = int(PATCH * ds1)                              # patch size in level-0 pixels
    step = int(a.stride * ds1)
    W0, H0 = slide.dimensions
    coords, labels = [], []
    for y in range(0, H0 - size0, step):
        for x in range(0, W0 - size0, step):
            x1, y1 = int(x / mds), int(y / mds)
            x2, y2 = max(x1 + 1, int((x + size0) / mds)), max(y1 + 1, int((y + size0) / mds))
            if tis[y1:y2, x1:x2].mean() < a.min_tissue: continue
            coords.append((x, y))
            labels.append(int(tum[y1:y2, x1:x2].mean() > a.tumor_frac))
    np.savez_compressed(f"{a.out}/{r.slide_id}.npz", coords=np.array(coords).reshape(-1, 2),
                        labels=np.array(labels), size0=size0, mds=mds, tumor=tum)
    return len(coords), int(sum(labels))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="manifest.csv")
    p.add_argument("--out", default="patches")
    p.add_argument("--stride", type=int, default=256)    # in 20x pixels; 512 = 4x fewer patches
    p.add_argument("--min_tissue", type=float, default=0.5)
    p.add_argument("--tumor_frac", type=float, default=0.5)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.manifest)
    for r in tqdm(df.itertuples(), total=len(df)):
        if os.path.exists(f"{a.out}/{r.slide_id}.npz"): continue
        n, t = process(r, a)
        tqdm.write(f"{r.slide_id}: {n} patches, {t} tumor")
