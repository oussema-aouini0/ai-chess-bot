"""Coverage diagnostic: does train.csv contain OOD-shaped positions at all?
Read-only: total pieces/row, material imbalance/row, nearest-neighbor of the
57 OOD bucket FENs in (piece-count, |imbalance|) space."""
import pathlib
import sys

import numpy as np
import pandas as pd

_DIR = pathlib.Path(__file__).resolve().parent
REPO = _DIR.parent.parent
sys.path.insert(0, str(REPO))
CSV = REPO / "train-an-ai-to-play-chess" / "train.csv"

from ood_bucket import OOD_SYNTH, RANDOM_EXTREME

VAL = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900, "k": 0}


def pc_and_imb(board_fen):
    pc = 0
    imb = 0
    for ch in board_fen:
        if ch.isalpha():
            pc += 1
            v = VAL[ch.lower()]
            imb += v if ch.isupper() else -v
    return pc, imb


df = pd.read_csv(CSV, usecols=["board"])
parts = df["board"].str.split().str[0]
stats = parts.apply(lambda s: pd.Series(pc_and_imb(s), dtype=np.float64))
pc = stats[0].to_numpy()
imb = stats[1].to_numpy()
absimb = np.abs(imb)

print("=== 1. total pieces per row (58,785 rows) ===")
for q in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
    print(f"  p{q:>3} = {int(np.percentile(pc, q))}")
for t in [32, 28, 24, 20, 16, 12, 10, 8, 6, 4, 3, 2, 1]:
    print(f"  rows with <= {t:>2} pieces: {(pc <= t).sum()}")
print("  min/max pieces:", pc.min(), pc.max())

print("\n=== 2. |material imbalance| per row ===")
for q in [50, 75, 90, 95, 97.5, 99, 100]:
    print(f"  p{q:>4} = {int(np.percentile(absimb, q))}")
for t in [300, 600, 900, 1200, 1600, 2000, 3000]:
    print(f"  rows with |imbalance| >= {t:>5}: {(absimb >= t).sum()}  ({100*(absimb >= t).mean():.2f}%)")

print("\n=== 3. nearest training match in (pieces, |imbalance|) space ===")
import chess
import chess_bot
OOD = [t[1] for t in chess_bot.EVALUATE_TACTICAL_FENS] + OOD_SYNTH + RANDOM_EXTREME
print(f"  OOD bucket size: {len(OOD)}")

train_pc = pc
train_imb = imb
train_abs = absimb
track = []
d_pc_all = []
d_imb_all = []
minviol = []
for fen in OOD:
    b = chess.Board(fen)
    bf = str(b).replace(" ", "").replace("\n", "").replace(".", "")
    qpc, qimb = pc_and_imb(bf)
    d_pc = np.abs(train_pc - qpc)
    d_abs = np.abs(train_abs - abs(qimb))
    d_comb = d_pc + d_abs / 100.0
    i_pc = int(np.argmin(d_pc)); i_abs = int(np.argmin(d_abs)); i_c = int(np.argmin(d_comb))
    track.append((qpc, qimb, d_pc[i_pc], d_abs[i_abs], d_comb[i_c],
                  train_pc[i_c], train_imb[i_c]))
    d_pc_all.append(d_pc[i_pc]); d_imb_all.append(d_abs[i_abs])

dpc = np.array(d_pc_all)
dimb = np.array(d_imb_all)
print("  OOD q :: nearest-training (piece gap / |imb| gap in cp):")
qpc = np.array([t[0] for t in track])
for k in range(len(OOD)):
    qp, qi, dp_, da_, dc, tpc, timb = track[k]
    if qp <= 10 or abs(qi) >= 600:  # sparse or lopsided only
        print(f"    OOD pc={qp:>3} |imb|={abs(qi):>5} -> nearest train pc={tpc:>3} "
              f"imb={timb:>6}  (piece-gap={dp_:>3}, imb-gap={da_:>5})")
print("  summary of nearest gap over all 57:")
print(f"    piece-count gap: min={dpc.min()} median={int(np.median(dpc))} p90={int(np.percentile(dpc,90))} max={dpc.max()}")
print(f"    |imb| gap (cp):  min={int(dimb.min())} median={int(np.median(dimb))} p90={int(np.percentile(dimb,90))} max={int(dimb.max())}")
print("DONE")