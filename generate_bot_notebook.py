"""Generate chess_bot.ipynb - a notebook frontend for chess_bot.py.

The notebook reuses chess_bot.py's exact functions (single source of truth)
through load_namespace() + the cmd_* handlers.
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = "chess_bot.ipynb"

# --------------------------------------------------------------------------- #
# Markdown cells
# --------------------------------------------------------------------------- #
TITLE = """# chess_bot — notebook edition

A thin Jupyter frontend for **`chess_bot.py`**. Every cell calls the same
functions as the CLI, so there is only one implementation to maintain.

Open the notebook, run the **Setup** cell, optionally tune the knobs, then run
any action cell:

| Cell | What it does | CLI equivalent |
|---|---|---|
| Train | Trains the dual-head model on the CSV and saves `chess_model.pt` | `python chess_bot.py train` |
| Best move | Best move for a FEN | `python chess_bot.py move "<fen>"` |
| Policy rankings | Top moves by the net's policy head | `python chess_bot.py eval "<fen>" --top` |
| Play vs the bot | Interactive game with SVG boards | `python chess_bot.py play` |
| Play vs Stockfish | One game, bot vs Stockfish | `python chess_bot.py stockfish` |
| Rate the bot | Elo estimate vs Stockfish | `python chess_bot.py rate` |
"""

SETUP_MD = """## 1. Setup

Run the cell below once. It loads the engine namespace from `chess_bot.py`.
You do **not** need to run the full `chess_main.ipynb` first.
"""

KNOBS_MD = """## 2. Tunables

Edit these values (then re-run *only* this cell and the action cells that use
them). `BOT_MODE` selects the engine:

- `"search"` — minimax + piece-square tables + quiescence (strongest, default)
- `"nn"` — pure neural policy (fast, a bit weaker)
- `"nnleaf"` — minimax with neural leaf evaluation (slow)

`DEPTH` is only used by `"search"` / `"nnleaf"`.
"""

TRAIN_MD = """## 3. Train the model

Trains on all ~58k rows by default (`SAMPLE = None`). Reports test RMSE on
`EVAL_N` fresh positions and saves the best weights to `chess_model.pt`.
On this machine expect a few minutes.
"""

MOVE_MD = """## 4. Best move & policy rankings

`MODE_FEN` below is the position to analyse. Run the **Best move** cell to see
one move (`BOT_MODE` applies) and the **Policy rankings** cell to see the top
ten candidate moves scored by the network's policy head.
"""

PLAY_MD = """## 5. Play vs the bot

Runs an interactive game against `make_bot(BOT_MODE, DEPTH)`. You are
`HUMAN_PLAYER` ('w' = White, 'b' = Black). Boards render as SVG in Jupyter.

> You can play the same game from a terminal instead:
> `python chess_bot.py play --human w` (add `--nn` / `--nnleaf` for other modes).
"""

SF_MD = """## 6. Play vs Stockfish

One game: the bot plays **Black** against Stockfish at Elo `SF_ELO`
(`TIME_LIMIT` seconds per Stockfish move). Switch roles by calling
`play_against_stockfish(bot, bot_is_white=True, ...)`.
"""

RATE_MD = """## 7. Rate the bot

Plays `GAMES` games against each of several Stockfish Elo levels, alternating
colors, and estimates the bot's Elo. Takes a few minutes.
"""

ADDITIONAL_MD = """## Notes

- A `chess_model.pt` trained on the full dataset is included, so Play / Rate /
  Move work immediately even if you skip the training cell.
- Training writes `chess_model.pt` in the project root, which the terminal CLI
  (`chess_bot.py`) also reads — the notebook and CLI share the same model.
"""

# --------------------------------------------------------------------------- #
# Code cells
# --------------------------------------------------------------------------- #
SETUP = '''import os
import sys
import argparse

NB_DIR = os.path.abspath(os.getcwd())


def _find_project_root(start):
    d = start
    for _ in range(6):
        if os.path.exists(os.path.join(d, "chess_bot.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return start
        d = parent
    return start


PROJECT_ROOT = _find_project_root(NB_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import chess_bot

ns = chess_bot.load_namespace()

print("Engine loaded from:", os.path.join(PROJECT_ROOT, "chess_bot.py"))
print("Working directory :", os.getcwd())
print("Training CSV      :", ns["TRAIN_CSV_PATH"],
      "| exists:", os.path.exists(ns["TRAIN_CSV_PATH"]))
print("Model file        :", ns["MODEL_SAVE_PATH"],
      "| exists:", os.path.exists(ns["MODEL_SAVE_PATH"]))'''

KNOBS = '''# ---- Tunables: edit before running the action cells below ----
SAMPLE        = None       # None = all ~58k rows, int = number of rows
EPOCHS        = 60
BATCH         = 128
EVAL_N        = 2000       # fresh FENs used to report RMSE after training
GAMES         = 3          # games per Stockfish Elo level (rating cell)
TIME_LIMIT    = 0.1        # seconds Stockfish gets per move
SF_ELO        = 1320       # Stockfish strength for the single-game cell
HUMAN_PLAYER  = "w"        # side you play in the interactive cell ('w' or 'b')
BOT_MODE      = "search"   # 'search' (default) | 'nn' | 'nnleaf'
DEPTH         = 2          # minimax depth (used by 'search' / 'nnleaf')
MOVE_FEN      = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
OUT_MODEL     = ""         # "" = chess_model.pt in the project root


def trained_model():
    """Load the trained weights into a fresh model (errors if none yet)."""
    return ns["load_model_weights"](ns["MODEL_SAVE_PATH"])


def make_bot(mode=BOT_MODE, depth=DEPTH):
    """Return a bot callable (fen, player) -> best move UCI string."""
    model = trained_model()
    if mode == "nn":
        return lambda fen, player="b": ns["play_nn"](fen, model, player=player)
    if mode == "nnleaf":
        return lambda fen, player="b": ns["play_nn"](
            fen, model, use_minimax=True, depth=depth, nn_leaf=True, player=player)
    return lambda fen, player="b": ns["play_nn"](
        fen, model, use_minimax=True, depth=depth, player=player)'''

TRAIN = '''# Trains on the CSV (SAMPLE rows), saves best weights to chess_model.pt.
chess_bot.cmd_train(
    ns,
    argparse.Namespace(
        sample=(SAMPLE or 0),
        epochs=EPOCHS,
        batch=BATCH,
        out=OUT_MODEL or "",
        eval=EVAL_N,
    ),
)'''

MOVE = '''board = ns["chess"].Board(MOVE_FEN)
player = "w" if board.turn == ns["chess"].WHITE else "b"
move = make_bot(BOT_MODE, DEPTH)(MOVE_FEN, player)
print("Position      :", MOVE_FEN)
print("Player to move:", player.upper())
print(f"Best move ({BOT_MODE}, depth {DEPTH}):", move)'''

POLICY = '''model = trained_model()
board = ns["chess"].Board(MOVE_FEN)
player = "w" if board.turn == ns["chess"].WHITE else "b"
print(f"Top moves by policy for {MOVE_FEN} (side to move: {player.upper()}):")
ns["play_nn"](MOVE_FEN, model, player=player, show_move_evaluations=True)'''

PLAY = '''bot = make_bot(BOT_MODE, DEPTH)
opponent = "Black" if HUMAN_PLAYER == "w" else "White"
print(f"You are {HUMAN_PLAYER.upper()}, the bot is {opponent} "
      f"({BOT_MODE}, depth {DEPTH}). Enter UCI moves, or 'quit' to end.")
ns["play_game"](bot, human_player=HUMAN_PLAYER)'''

SF = '''bot = make_bot(BOT_MODE, DEPTH)
result = ns["play_against_stockfish"](
    bot, bot_is_white=False, stockfish_elo=SF_ELO, time_limit=TIME_LIMIT)
print("Result (from White's perspective):", result)'''

RATE = '''bot = make_bot(BOT_MODE, DEPTH)
print(f"Rating the '{BOT_MODE}' bot vs Stockfish "
      f"(depth {DEPTH}, {GAMES} game(s) per level, "
      f"{TIME_LIMIT}s/move). This takes a few minutes...")
rating, summary = ns["rate_bot"](bot, num_games=GAMES, time_limit=TIME_LIMIT)
print(f"Final bot rating: {rating:.0f}")'''

# --------------------------------------------------------------------------- #
# Assemble
# --------------------------------------------------------------------------- #
nb = new_notebook(
    cells=[
        new_markdown_cell(TITLE),
        new_markdown_cell(SETUP_MD),
        new_code_cell(SETUP),
        new_markdown_cell(KNOBS_MD),
        new_code_cell(KNOBS),
        new_markdown_cell(TRAIN_MD),
        new_code_cell(TRAIN),
        new_markdown_cell(MOVE_MD),
        new_code_cell(MOVE),
        new_code_cell(POLICY),
        new_markdown_cell(PLAY_MD),
        new_code_cell(PLAY),
        new_markdown_cell(SF_MD),
        new_code_cell(SF),
        new_markdown_cell(RATE_MD),
        new_code_cell(RATE),
        new_markdown_cell(ADDITIONAL_MD),
    ]
)
nb.metadata["kernelspec"] = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3",
}

with open(OUT, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Wrote {OUT} with {len(nb.cells)} cells")