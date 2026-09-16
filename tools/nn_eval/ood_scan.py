"""Precompute SF d12 White-POV scores for the OOD bucket, one fresh engine per
FEN so no single board (or engine instability) can kill the caller. Writes
ood_refs.csv (fen, white_cp) in this directory and reports skipped FENs."""
import csv
import pathlib
import sys

_DIR = pathlib.Path(__file__).resolve().parent
REPO = _DIR.parent.parent
sys.path.insert(0, str(REPO))
import chess, chess.engine
import chess_bot as cb

from ood_bucket import OOD_SYNTH, RANDOM_EXTREME

ns = cb.load_namespace()

OOD = [t[1] for t in cb.EVALUATE_TACTICAL_FENS] + OOD_SYNTH + RANDOM_EXTREME

out = []
bad = []
for i, fen in enumerate(OOD):
    try:
        b = chess.Board(fen)
    except Exception:
        bad.append((fen, "FEN"))
        print(f"BAD-FEN {fen!r}", flush=True)
        continue
    eng = None
    try:
        eng = ns["get_stockfish_engine"](ns["STOCKFISH_PATH"])
        eng.configure({"Threads": 1, "Hash": 64})
        info = eng.analyse(b, chess.engine.Limit(depth=12))
        w = info["score"].pov(chess.WHITE)
        cp = (99999 if w.mate() > 0 else -99999) if w.is_mate() else w.score()
        out.append((fen, cp))
        print(f"[{i+1}/{len(OOD)}] ok  {cp:>7}  {fen!r}", flush=True)
    except Exception as e:
        bad.append((fen, type(e).__name__))
        print(f"[{i+1}/{len(OOD)}] SKIP {fen!r} {type(e).__name__}: {str(e)[:60]}", flush=True)
    finally:
        if eng is not None:
            try:
                eng.quit()
            except Exception:
                pass

with open(_DIR / "ood_refs.csv", "w", newline="") as f:
    cw = csv.writer(f)
    cw.writerow(["fen", "white_cp"])
    cw.writerows(out)
print(f"\nwrote {len(out)} refs; skipped {len(bad)}: {bad}")
print("DONE")