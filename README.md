# AI Chess Bot

A neural-network chess bot built with **PyTorch** and **python-chess**. It
learns to evaluate positions and pick moves from the Kaggle
[*Train an AI to play Chess*](https://www.kaggle.com/competitions/train-an-ai-to-play-chess)
dataset, plays against humans, and can be rated by playing Stockfish at
various Elo levels. Comes with a Jupyter notebook **and** a standalone CLI.

## Features

- **Dual-head neural model** (PyTorch): a shared trunk with BatchNorm + Dropout
  going to a **value head** (`black_score` regression) and a **policy head**
  (source/destination-square logits, trained on the `best_move` labels).
- **Board encoding** into an `8x8x19` tensor: 13 one-hot piece channels plus
  side-to-move, castling rights, and en-passant channels.
- **Move selection**:
  - *Policy*: scores each legal move with the network's from/to logits, picks
    the best (fast, purely neural).
  - *Minimax*: alpha-beta with MVV-LVA move ordering, piece-square tables
    and quiescence search (the strongest mode; **default**).
  - *NN-leaf*: minimax whose leaf positions are batched through the network.
- **Play vs the bot** in Jupyter (SVG boards) or a terminal (text boards),
  as White or Black.
- **Play / rate vs Stockfish**: Elo rating by playing at increasing Stockfish
  strengths, alternating colors.
- **Model persistence**: best weights are saved to `chess_model.pt` and loaded
  automatically.

## Measured strength (vs Stockfish 19, from the fully trained model)

| Bot mode | Approx. Elo |
|---|---|
| Pure value-NN (older, single-head) | ~1310 |
| Pure policy-NN (dual-head) | ~1360-1390 |
| Minimax + PST + quiescence (default) | ~1440-1480 |

## Prerequisites

- Python 3.10+
- Dependencies (install once):

  ```bash
  pip install -r requirements.txt
  ```

- **Stockfish**: a Stockfish 19 Windows binary is bundled at
  `stockfish/stockfish.exe`; Section 1 of the notebook points at it. On
  another OS, replace it with your own Stockfish (<https://stockfishchess.org>)
  and update `STOCKFISH_PATH`.

## Host it publicly (free)

Everyone can play the bot from the internet — **free and no credit card** on
Render (~370 MB RAM used of the 512 MB free limit, HTTPS included, no
Stockfish or training data shipped):

```bash
# 1. push this folder to a GitHub repo
# 2. render.com -> New + -> Blueprint -> pick the repo  (package is free)
# 3. open the service URL, e.g. https://chess-bot.onrender.com
```

Full copy-paste steps: **[DEPLOY.md](DEPLOY.md)**. The `render.yaml`,
`Dockerfile`, `requirements-deploy.txt` and `.dockerignore` are already here.

> Note: Hugging Face Spaces used to host Docker apps for free, but as of 2026
> Docker/Gradio Spaces require a paid (PRO) plan — only static sites are free,
> which can't run a backend. Render free is the plain no-cost path.

## CLI quick start

```bash
# Train on all ~58k rows (best weights saved to chess_model.pt)
python chess_bot.py train

# Play a game vs the bot (you are White; default = minimax depth 2)
python chess_bot.py play --human w

# Best move / top-10 policy evaluations for a position
python chess_bot.py move "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
python chess_bot.py eval "<fen>" --top

# Rate the bot vs Stockfish (Elo estimate)
python chess_bot.py rate --games 3

# One game bot vs Stockfish (bot plays Black)
python chess_bot.py stockfish --elo 1320

# Variants: --nn = pure neural policy, --nnleaf = NN-leaf minimax, --depth N
python chess_bot.py play --nn          # pure neural bot
python chess_bot.py move --nnleaf "<fen>"
```

## Interactive web board (drag & drop)

A local web app with a click/drag chessboard — no typing moves:

```bash
C:/Users/Admin/anaconda3/python.exe chess_web.py
# open http://127.0.0.1:5000
```

Drag pieces to move (or click a piece to see highlighted legal moves, then
click the destination). Pawns auto-promote to queen. Options: your color, bot
mode (`Search` / `Neural` / `NN-leaf`) and search depth. The bot uses the same
engine and model as the CLI/notebooks.

## Jupyter

Two notebooks are included:

- `chess_main.ipynb` — the full pipeline (encoding, training, search, play,
  rating). **Section 1** holds the paths; **Sections 2-4** encode, build and
  train the model; **Sections 5-6** sanity-check and play.
- `chess_bot.ipynb` — a trimmed notebook frontend for the CLI. Cells reuse the
  exact `chess_bot.py` functions (train / best move / policy rankings / play /
  Stockfish / rate) with an interactive SVG board. Just run **Setup**, tune the
  knobs, then run an action cell.

## Project layout

| File | Purpose |
|---|---|
| `chess_main.ipynb` | Full pipeline: encoding, training, search, play, rating |
| `chess_bot.py` | Standalone CLI mirroring the notebook |
| `chess_bot.ipynb` | Notebook frontend for the CLI (SVG play board) |
| `chess_web.py` | Interactive Flask web board (drag & drop pieces) |
| `render.yaml` | Render Blueprint (free web service, Docker, health check) |
| `Dockerfile` / `.dockerignore` | Container image (CPU torch + gunicorn) |
| `requirements-deploy.txt` | Runtime deps for the container |
| `DEPLOY.md` | Step-by-step public hosting guide |
| `train-an-ai-to-play-chess/train.csv` | Kaggle training data |
| `chess_model.pt` | Trained model weights (created by `train`) |
| `requirements.txt` | Python dependencies |
| `generate_notebook.py` | Rebuilds `chess_main.ipynb` (dev only) |
| `generate_bot_notebook.py` | Rebuilds `chess_bot.ipynb` (dev only) |
| `stockfish/stockfish.exe` | Bundled Stockfish 19 engine (Windows) |

## Troubleshooting

- **FileNotFoundError** for the training CSV: update `TRAIN_CSV_PATH` in
  Section 1. OneDrive "on-demand" files must be downloaded to disk first.
- **Stockfish not found**: you're not on Windows, or the exe was moved/replaced.
  Set `STOCKFISH_PATH` in Section 1. The notebook also searches the PATH.
- **UCI_Elo error**: Stockfish's minimum strength limit is 1320; the notebook
  clamps Elo to at least 1320 automatically.
- **Slow play**: depth 2 minimax is fast; depth 3 takes several seconds per
  move. Lower `--depth` or use `--nn` for speed.
- **Slow training**: training is CPU-friendly (~1.1M params), but on the full
  58k-row dataset expect ~5-10 minutes. Use `--sample N` for experiments.
```