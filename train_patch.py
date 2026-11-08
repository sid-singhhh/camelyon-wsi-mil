import argparse, os
import pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def load(df, feats, keys):
    X, Y = [], []
    for sid in df.slide_id:
        p = f"{feats}/{sid}.pt"
        if not os.path.exists(p): continue
        d = torch.load(p)
        X.append(torch.cat([d[k] for k in keys], 1)); Y.append(d["labels"])
    return torch.cat(X), torch.cat(Y).float()

p = argparse.ArgumentParser()
p.add_argument("--manifest", default="manifest.csv")
p.add_argument("--feats", default="feats")
p.add_argument("--epochs", type=int, default=10)
a = p.parse_args()
df = pd.read_csv(a.manifest)
tr, te = df[df.split == "train"], df[df.split == "test"]

for name, keys in [("20x", ["f20"]), ("10x", ["f10"]), ("20x+10x", ["f20", "f10"])]:
    Xtr, Ytr = load(tr, a.feats, keys); Xte, Yte = load(te, a.feats, keys)
    mu, sd = Xtr.float().mean(0), Xtr.float().std(0) + 1e-6
    clf = nn.Linear(Xtr.shape[1], 1).cuda()
    opt = torch.optim.Adam(clf.parameters(), 1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=((Ytr == 0).sum() / (Ytr == 1).sum().clamp(min=1)).cuda())
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 4096):
            idx = perm[i:i + 4096]
            x = ((Xtr[idx].float() - mu) / sd).cuda(); y = Ytr[idx].cuda()
            opt.zero_grad(); lossf(clf(x).squeeze(1), y).backward(); opt.step()
    with torch.no_grad():
        pr = torch.cat([torch.sigmoid(clf(((Xte[i:i+8192].float() - mu) / sd).cuda()).squeeze(1)).cpu()
                        for i in range(0, len(Xte), 8192)]).numpy()
    yt = Yte.numpy()
    print(f"{name}: n_test={len(yt)} pos={int(yt.sum())} "
          f"acc={accuracy_score(yt, pr > .5):.4f} auc={roc_auc_score(yt, pr):.4f} f1={f1_score(yt, pr > .5):.4f}")
