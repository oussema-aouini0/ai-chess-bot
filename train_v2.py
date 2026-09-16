"""Train value-head v2: full 58,785-row dataset with inverse-sqrt sample
weighting to counter the near-zero label skew. Same architecture/loss/optimizer
as the notebook's train_model. Saves chess_model_v2.pt + training history CSV.
Read-only w.r.t. existing files; never touches chess_model.pt.
"""
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import chess_bot as cb

ns = cb.load_namespace()
DEVICE = ns["DEVICE"]
VS = float(ns["VALUE_SCALE"])
CSV = ns["TRAIN_CSV_PATH"]
V2 = os.path.join(REPO, "chess_model_v2.pt")
HIST = os.path.join(REPO, "chess_model_v2_history.csv")

t0 = time.time()
X, y, mf, mt = ns["load_dataset"](CSV, sample_size=None, seed=42)
print(f"encoded {len(X)} in {time.time()-t0:.0f}s")

idx = np.arange(len(y))
i_tr, i_temp = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
i_va, i_te = train_test_split(i_temp, test_size=0.5, random_state=42, shuffle=True)

Xtr, ytr, mftr, mttr = X[i_tr], y[i_tr], mf[i_tr], mt[i_tr]
Xva, yva, mfva, mtva = X[i_va], y[i_va], mf[i_va], mt[i_va]

# --- Sample weights: w = 1/sqrt(bin_count), mean-1 normalized, clipped [0.25, 3] ---
edges = np.arange(-2000, 2001, 100).tolist()
labels_all = np.concatenate([ytr, yva])
counts, _ = np.histogram(labels_all, bins=[-np.inf] + edges + [np.inf])
with np.errstate(divide="ignore"):
    w_all = np.where(counts > 0, 1.0 / np.sqrt(counts), 0.0)
w_all = w_all / w_all.mean()
w_all = np.clip(w_all, 0.6, 1.6)
w_all = w_all / w_all.mean()  # keep macro mean ~1 so value_coef=5 keeps its scale
bins_tr = np.digitize(ytr, [i for i in edges], right=False)
wtr = w_all[np.minimum(bins_tr, len(w_all) - 1)].astype(np.float32)
bins_va = np.digitize(yva, [i for i in edges], right=False)
wva = w_all[np.minimum(bins_va, len(w_all) - 1)].astype(np.float32)
print("weight stats: min={:.3f} mean={:.3f} max={:.3f}".format(wtr.min(), wtr.mean(), wtr.max()))

BS = 128
def mk(Xe, ye, mfe, mte, we, shuffle):
    d = TensorDataset(torch.tensor(Xe), torch.tensor(ye), torch.tensor(mfe),
                      torch.tensor(mte), torch.tensor(we))
    return DataLoader(d, batch_size=BS, shuffle=shuffle)

tr_loader = mk(Xtr, ytr, mftr, mttr, wtr, True)
va_loader = mk(Xva, yva, mfva, mtva, wva, False)

model = ns["ChessEvalModel"](input_channels=19).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()
ce = torch.nn.functional.cross_entropy
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                   patience=4, min_lr=1e-6)

def value_metrics(yb, pb):
    rmse = float(torch.sqrt(torch.mean((pb - yb) ** 2)))
    corr = float(np.corrcoef(yb.numpy(), pb.numpy())[0, 1])
    return rmse, corr

best_val = float("inf")
epochs_no_improve = 0
hist_rows = []
t0 = time.time()
for ep in range(1, 61):
    model.train()
    tl = tv = tfce = ttce = 0.0
    nb = 0
    for xb, yb, fb, tb, wb in tr_loader:
        xb, yb, fb, tb, wb = (xb.to(DEVICE), yb.to(DEVICE), fb.to(DEVICE),
                              tb.to(DEVICE), wb.to(DEVICE))
        opt.zero_grad()
        p = model(xb)
        wv = (wb * (p["value"] - yb / VS) ** 2).mean()
        v = 5.0 * wv
        lf = ce(p["from"], fb)
        lt = ce(p["to"], tb)
        (v + lf + lt).backward()
        opt.step()
        tl += wv.item() * len(xb); tv += v.item() * len(xb)
        tfce += lf.item() * len(xb); ttce += lt.item() * len(xb)
        nb += len(xb)
    tl /= nb; tv /= nb; tfce /= nb; ttce /= nb

    model.eval()
    vl = vv = vfce = vtce = 0.0
    nv = 0
    all_y, all_p = [], []
    with torch.no_grad():
        for xb, yb, fb, tb, wb in va_loader:
            yb_ = yb
            p = model(xb.to(DEVICE))
            wv = (wb * (p["value"] - yb / VS) ** 2).mean()
            v = 5.0 * wv
            lf = ce(p["from"], fb.to(DEVICE))
            lt = ce(p["to"], tb.to(DEVICE))
            vl += wv.item() * len(xb); vv += v.item() * len(xb)
            vfce += lf.item() * len(xb); vtce += lt.item() * len(xb)
            nv += len(xb)
            all_y.extend(yb_.numpy().tolist())
            all_p.extend((p["value"].cpu().numpy() * VS).tolist())
    vl /= nv; vv /= nv; vfce /= nv; vtce /= nv
    rmse, corr = value_metrics(torch.tensor(all_y), torch.tensor(all_p))
    slope = float(np.polyfit(all_y, all_p, 1)[0])

    sched.step(vv)
    hist_rows.append([ep, tl, vl, vv, vfce, vtce, rmse, corr, slope, sched.last_epoch])
    print(f"ep {ep:2d} train(wMSE*5/CEf/CEt)={tv:.3f}/{tfce:.3f}/{ttce:.3f} "
          f"val(wMSE/unwMSE?)={vl:.4f}/{vv:.3f} ({vfce:.2f}/{vtce:.2f}) "
          f"valRMSE={rmse:.1f} corr={corr:.3f} slope={slope:.3f} lr={opt.param_groups[0]['lr']:.1e}", flush=True)

    if vl < best_val:
        best_val = vl
        epochs_no_improve = 0
        torch.save(model.state_dict(), V2)
        print(f"  -> saved v2 (best val wMSE {best_val:.4f})", flush=True)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= 8:
            print(f"Early stop at epoch {ep} (best val {best_val:.4f})")
            break

with open(HIST, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_wmse5", "val_wmse", "val_wmse5", "val_ce_f", "val_ce_t",
                "val_rmse_cp", "val_corr", "val_slope"])
    w.writerows([[h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]] for h in hist_rows])
print(f"history -> {HIST}")
print(f"v2 weights -> {V2}  (best val wMSE {best_val:.4f})  time {time.time()-t0:.0f}s")
print("DONE")