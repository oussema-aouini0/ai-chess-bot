"""Matched-depth-3 accuracy + OOD eval-correlation comparison.
Legs: pst | old (chess_model.pt, NN leaves) | v2 (chess_model_v2.pt, NN leaves).
NN leaves now run the same capture quiescence as PST (harness-fair).
"""
import csv
import math
import pathlib
import statistics
import sys

import chess
import chess.engine

_DIR = pathlib.Path(__file__).resolve().parent
REPO = _DIR.parent.parent
sys.path.insert(0, str(REPO))
import chess_bot as cb

from ood_bucket import OOD_SYNTH, RANDOM_EXTREME

LEGS = sys.argv[1:] if len(sys.argv) > 1 else ["pst"]
MATE = 99999


def _ok(fen):
    try:
        chess.Board(fen)
        return True
    except Exception:
        return False


def _corr(a, b):
    import numpy as np
    return float(np.corrcoef(a, b)[0, 1])


def math_rmse(a, b):
    import numpy as np
    return float(np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2)))


def lin_slope(x, y):
    import numpy as np
    return float(np.polyfit(x, y, 1)[0])


def stdev(x):
    return statistics.stdev(x) if len(x) > 1 else 0.0


ns = cb.load_namespace()
engine = ns["get_stockfish_engine"](ns["STOCKFISH_PATH"])
engine.configure({"Threads": 1, "Hash": 64})


def _restart_engine():
    global engine
    try:
        engine.quit()
    except Exception:
        pass
    engine = ns["get_stockfish_engine"](ns["STOCKFISH_PATH"])
    engine.configure({"Threads": 1, "Hash": 64})


SKIP = set()


def sf_score_cp(board, depth=12):
    for attempt in range(2):
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            w = info["score"].pov(chess.WHITE)
            if w.is_mate():
                return MATE if w.mate() > 0 else -MATE
            return w.score()
        except Exception as e:
            SKIP.add(board.fen())
            print(f"  [sf skip] {board.fen()!r}: {type(e).__name__} {str(e)[:50]}", flush=True)
            _restart_engine()
    return None

pool = [t[1] for t in cb.EVALUATE_TACTICAL_FENS] + cb._test_split_fens(ns, 67)
tactical_fens = [t[1] for t in cb.EVALUATE_TACTICAL_FENS]

def _load_v2():
    import torch
    m = ns["ChessEvalModel"](input_channels=19)
    m.load_state_dict(torch.load(str(REPO / "chess_model_v2.pt"),
                                 map_location="cpu", weights_only=True))
    m.eval()
    return m


models = {"pst": None, "old": None, "v2": None}
for leg in LEGS:
    if leg == "old":
        models["old"] = cb.require_model(ns)
    if leg == "v2":
        models["v2"] = _load_v2()


def sf_score_cp(board, depth=16):
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    w = info["score"].pov(chess.WHITE)
    if w.is_mate():
        return MATE if w.mate() > 0 else -MATE
    return w.score()


for leg in LEGS:
    model = models[leg]
    nn_leaf = leg != "pst"

    match = top3 = 0
    losses, depths = [], []
    for fen in pool:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        player = "w" if board.turn == chess.WHITE else "b"
        move = ns["play_nn"](fen, model, player=player, use_minimax=True,
                             depth=3, time_limit=None, nn_leaf=nn_leaf)
        if not move:
            continue
        depths.append(ns["_LAST_SEARCH"]["depth"])
        best, top, _ = cb._sf_pool(ns, engine, board, 15)
        best_cp = cb._move_cp(ns, engine, board, best, 15)
        bot_cp = cb._move_cp(ns, engine, board, move, 15)
        losses.append(cb._mate_loss(best_cp, bot_cp))
        if move == best:
            match += 1; top3 += 1
        elif move in top:
            top3 += 1
    n = len(losses)
    blunders = sum(1 for c in losses if c > 100)
    print(f"[leg={leg}] 80-pos matched-d3: "
          f"match={100*match/n:.1f}% top3={100*top3/n:.1f}% "
          f"ACPL={statistics.mean(losses):.1f} blunder={100*blunders/n:.1f}% "
          f"(avgD={statistics.mean(depths):.2f}, n={n})", flush=True)

    ood = tactical_fens + OOD_SYNTH + RANDOM_EXTREME
    boards = [chess.Board(f) for f in ood if _ok(f)]
    # Reference 1: deterministic material+PST (White POV) -- what the net collapses against
    matref = [float(ns["evaluate_board"](b, None)) for b in boards]
    # Reference 2: SF-d12 White-POV where a clean analysis was captured in ood_refs.csv
    sfref = {}
    with open(_DIR / "ood_refs.csv") as f:
        for row in list(csv.reader(f))[1:]:
            if len(row) == 2:
                try:
                    sfref[chess.Board(row[0]).fen()] = int(float(row[1]))
                except Exception:
                    pass
    preds = [ns["evaluate_board"](b, model) for b in boards]
    r = _corr(matref, preds)
    rmse = math_rmse(matref, preds)
    slope = lin_slope(matref, preds)
    print(f"[leg={leg}] OOD bucket (n={len(boards)}) vs material/PST ref: "
          f"corr={r:.3f} RMSE={rmse:.1f} slope={slope:.2f} "
          f"(matref sd={stdev(matref):.1f})", flush=True)
    sub = [(b, p) for b, p in zip(boards, preds) if b.fen() in sfref]
    if sub:
        sbb, spp = zip(*sub)
        rf = [sfref[b.fen()] for b in sbb]
        print(f"[leg={leg}] OOD bucket vs SF-d12 ref (n={len(sub)}): "
              f"corr={_corr(rf, spp):.3f} RMSE={math_rmse(rf, spp):.1f}", flush=True)

    # tactical-13 subset (inside OOD)
    tb, tp = [], []
    for b, p in zip(boards, preds):
        if b.fen() in [chess.Board(f).fen() for f in tactical_fens]:
            tb.append(b); tp.append(p)
    if tb:
        tmat = [float(ns["evaluate_board"](b, None)) for b in tb]
        print(f"[leg={leg}] tactical-13 subset: corr(vs mat)={_corr(tmat, tp):.3f} "
              f"(n={len(tb)})", flush=True)

    if nn_leaf:
        import pandas as pd
        df = pd.read_csv(ns["TRAIN_CSV_PATH"]).sample(n=500, random_state=7)
        bf = [chess.Board(f) for f in df["board"]]
        labs = df["black_score"].to_numpy()
        predb = ns["evaluate_positions"](bf, model)
        print(f"[leg={leg}] in-dist 500 vs labels: corr={_corr(labs, predb):.3f} "
              f"RMSE={math_rmse(labs, predb):.1f} slope={lin_slope(labs, predb):.2f}", flush=True)

print("\nDONE", flush=True)