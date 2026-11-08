import argparse, copy, json, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

class ABMIL(nn.Module):
    def __init__(s, d_in, d=512, h=256):
        super().__init__()
        s.fc = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(0.25))
        s.V, s.U, s.w, s.cls = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, 1), nn.Linear(d, 1)
    def forward(s, x):
        h = s.fc(x)
        a = torch.softmax(s.w(torch.tanh(s.V(h)) * torch.sigmoid(s.U(h))).squeeze(-1), 0)
        return s.cls((a.unsqueeze(-1) * h).sum(0)).squeeze(), a

def load_bags(df, feats, mode):
    bags = {}
    for r in df.itertuples():
        p = f"{feats}/{r.slide_id}.pt"
        if not os.path.exists(p): continue
        d = torch.load(p)
        x = {"single": d["f20"], "f10": d["f10"], "multi": torch.cat([d["f20"], d["f10"]], 1)}[mode]
        bags[r.slide_id] = (x, r.label)
    return bags

def evaluate(model, bags):
    model.eval(); ys, ps, att = [], [], {}
    with torch.no_grad():
        for sid, (x, y) in bags.items():
            logit, a = model(x.cuda().float())
            ys.append(y); ps.append(torch.sigmoid(logit).item()); att[sid] = a.cpu().numpy()
    return ys, ps, att

p = argparse.ArgumentParser()
p.add_argument("--manifest", default="manifest.csv")
p.add_argument("--feats", default="feats")
p.add_argument("--mode", default="single", choices=["single", "multi", "f10"])
p.add_argument("--tag", default=None)
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
tag = a.tag or f"{a.feats}_{a.mode}"
torch.manual_seed(a.seed); np.random.seed(a.seed)

df = pd.read_csv(a.manifest)
trdf, vadf = train_test_split(df[df.split == "train"], test_size=0.2, stratify=df[df.split == "train"].label, random_state=a.seed)
tedf = df[df.split == "test"]
train, val, test = [load_bags(d, a.feats, a.mode) for d in (trdf, vadf, tedf)]
d_in = next(iter(train.values()))[0].shape[1]

model = ABMIL(d_in).cuda()
opt = torch.optim.Adam(model.parameters(), 2e-4, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss()
best, best_state = -1, None
for ep in range(a.epochs):
    model.train()
    for sid in np.random.permutation(list(train)):
        x, y = train[sid]
        logit, _ = model(x.cuda().float())
        loss = lossf(logit, torch.tensor(float(y)).cuda())
        opt.zero_grad(); loss.backward(); opt.step()
    ys, ps, _ = evaluate(model, val)
    auc = roc_auc_score(ys, ps)
    if auc > best: best, best_state = auc, copy.deepcopy(model.state_dict())
    print(f"epoch {ep} val_auc {auc:.3f}")

model.load_state_dict(best_state)
ys, ps, att = evaluate(model, test)
res = dict(tag=tag, seed=a.seed, val_auc=best, test_auc=roc_auc_score(ys, ps),
           test_acc=accuracy_score(ys, np.array(ps) > 0.5), n_test=len(ys))
print(res)
os.makedirs("results", exist_ok=True); os.makedirs(f"results/attn_{tag}", exist_ok=True)
json.dump(res, open(f"results/{tag}_s{a.seed}.json", "w"))
pd.DataFrame(dict(slide_id=list(test), y=ys, p=ps)).to_csv(f"results/preds_{tag}_s{a.seed}.csv", index=False)
for sid, v in att.items(): np.save(f"results/attn_{tag}/{sid}.npy", v)
