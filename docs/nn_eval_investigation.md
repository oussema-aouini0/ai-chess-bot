# NN Evaluation Investigation

Closed out 2026-09. Question, methods, evidence and verdict for the "does the
NN value head competently evaluate leaf positions?" investigation. Bottom line:
**it does not, and the reason is an architecture/function-class limitation, not
data volume or data skew.** The PST + quiescence baseline remains the engine's
default evaluation everywhere.

## Original question

The engine can evaluate leaves two ways: fast piece-square tables (PST) or a
small MLP value head read from a trained board representation. At a matched
time budget, the NN-leaf engine played much weaker than the PST engine. Was the
gap caused by *search* (PST search goes deeper / gets quiescence) or by *the
evaluation itself* (the net scores positions badly)?

## Step 0 — search depth alone does not lift strength

A completed-depth run (d10, 15 s/move — 5x the normal budget) showed the engine
plateaus: match 50.0%, top3 62.5%, ACPL 121.0, blunder 16.2%, avg depth 4.51.
More search budget is not the live lever.

## NN-leaf underperforms PST at matched settings

Matched time budget (t15):
| leaf | match | top3 | ACPL | blunder |
|---|---|---|---|---|
| PST | 50.0% | 62.5% | 121.0 | 16.2% |
| NN-leaf | 20.0% | 40.0% | 256.4 | 41.2% |

Matched depth 3, harness-fair after the quiescence fix (both leaf types now run
identical capture quiescence; see `generate_notebook.py` / `chess_main.ipynb`:
`quiescence(..., nn_model=...)`, `minimax` depth==0 always routes through it):
| leaf | match | top3 | ACPL | blunder |
|---|---|---|---|---|
| PST | 50.0% | 66.2% | 57.5 | 13.8% |
| old NN-leaf | 18.8% | 45.0% | 207.3 | 36.2% |
| v2 NN-leaf | 16.2% | 43.8% | 209.3 | 38.8% |

Before the quiescence fix the old NN figures were 22.5%/41.2%/308.1/41.2 — the
missing quiescence was a real confound (ACPL 308 -> 207) but explained only a
fraction of the gap. The neural evaluation is the weak point, not search depth.

## Root cause — in-distribution fit is fine, out-of-distribution collapse is real

- In distribution (5k training rows held out): corr r=0.90, slope 0.78, RMSE
  ~189 cp. The net learns the training distribution competently (mild
  regression-to-mean).
- Out of distribution (57-position bucket: 13 real tactical + 41 synthetic
  sparse/extreme endgames + 3 random extreme-material): correlation with
  material/PST truth ≈ 0 or *negative* (old: −0.13; tactical subset −0.38).
  The net is essentially clueless on exactly the position class that needs the
  sharpest evaluation.
- Scale chain verified (÷512 train / ×512 infer, `VALUE_SCALE`), encoding
  verified (8x8x19 planes, 12 = empty), no activation on the value head, dataset
  had no duplicate boards. No silent bug.

## Retrain (v2) did not fix the collapse

Full 58,785-row dataset (not the 90%) + sample weighting: fixed 100 cp bins over
±2000 + overflow, `w = 1/sqrt(bin_count)`, mean-1 normalized, clipped to
[0.6, 1.6]. Weighted value MSE + unweighted policy CE, Adam 1e-3,
ReduceLROnPlateau(p=4, factor=0.5, min 1e-6), early stop patience 8. Best at
ep35 of 43: val wMSE 0.1205, val corr 0.856, RMSE 205.9 cp, slope 0.72.
Artifacts: `chess_model_v2.pt`, `chess_model_v2_history.csv`, `train_v2.py`.
**The OOD correlation stayed ≈ 0** (v2: −0.04 vs material, −0.23 tactical). More
data + reweighting did not move the OOD numbers.

## Coverage check — rules out "the data didn't cover this region"

Checks against all 58,785 rows (see `tools/nn_eval/coverage_diag.py`):

- Piece counts: median 17; **17,683 rows ≤ 10 pieces (30%)**, 6,369 ≤ 4 (11%),
  min 2. Sparse endgames are heavily represented.
- Material imbalance: **1,267 rows ≥ 600 cp (2.2%)**, 656 ≥ 900 cp (1.1%),
  max 1,730 cp. Lopsided positions exist in meaningful number.
- Nearest-neighbor for each of the 57 OOD FENs in (piece count, |imbalance|)
  space: piece-count gap **0 for every position** (min=median=90th=0), imbalance
  gap 0 cp for almost all, max 170 cp. Structurally near-identical training
  positions exist for every OOD position.

Result: the OOD positions are not structurally exotic — the region is present in
the training data. The model simply cannot represent the sharp tactical
"boundaries" (winning vs drawn at the same piece count) that these positions
require.

## Verdict

The value head's **function class — a small MLP over raw board planes — is the
limitation**, not data volume or skew and not search depth. Calibration and
coverage hypotheses were tested directly and rejected; the architecture
hypothesis remains the only one consistent with all evidence.

## Noted but not pursued

- **Linear recalibration** (slope ~0.7 vs 1.0): rejected. A monotone rescale is
  ordering-invariant — preserves move choice. It cannot restore
  correlation ≈ 0; the problem is noise, not bias.
- **Synthetic data generation**: rejected. Coverage is not the problem
  (see above).
- **Architecture rewrite** (deeper net / conv features / policy-value dual
  head): parked. Large investment; separate future decision.
- **Hybrid PST + delta**: flagged as the lowest-risk path if NN eval is ever
  revisited — train a small net on PST's *residual* error rather than absolute
  eval, so the net only has to learn "what PST gets wrong", a much simpler
  function.

## Artifacts

- Findings: this file.
- Engine harness: quiescence now applies to both leaf types; completed-depth
  logging + `_LAST_SEARCH` instrumentation (`generate_notebook.py`,
  `chess_main.ipynb`, `chess_bot.py`).
- Retrained value head: `chess_model_v2.pt` (+ history CSV + `train_v2.py`).
  Kept as a research artifact; **not wired in as any default anywhere**.
- Diagnostics: `tools/nn_eval/` (OOD bucket construction, SF reference scanner,
  three-way matched-depth comparison, training-coverage check).