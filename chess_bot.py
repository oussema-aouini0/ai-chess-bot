"""Standalone CLI for the AI Chess Bot.

Reuses the exact engine code from chess_main.ipynb by loading the notebook's
code cells at runtime, so the CLI and notebook can never drift apart.

Usage:
    python chess_bot.py train [--sample N] [--epochs N] [--batch N]
    python chess_bot.py play [--human w|b] [--minimax] [--depth N] [--time S]
    python chess_bot.py rate [--games N] [--time S] [--bot-time S]
    python chess_bot.py stockfish [--elo N] [--as-white] [--time S] [--bot-time S]
    python chess_bot.py move <fen> [--depth N] [--time S]
    python chess_bot.py eval <fen> [--top N]
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

NOTEBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chess_main.ipynb")


def upto(src, marker):
    """Return src up to (not including) the line containing `marker`."""
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if marker in line:
            return "\n".join(lines[:i])
    return src


def drop_lines(src, needles):
    """Return src with any line starting with one of `needles` removed."""
    out = []
    for line in src.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(n) for n in needles):
            continue
        out.append(line)
    return "\n".join(out)


def load_namespace():
    """Load the notebook's engine definitions into a shared namespace."""
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    cells = [''.join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    def exec_cell(i, src):
        exec(compile(src, f"<notebook cell {i}>", "exec"), ns)

    ns = {}
    exec_cell(1, cells[1])                       # config / imports
    exec_cell(2, cells[2])                       # board encoding
    exec_cell(3, upto(cells[3], "# Encode the dataset"))   # load_dataset + split utils
    with contextlib.redirect_stdout(io.StringIO()):
        exec_cell(4, cells[4])                   # ChessEvalModel (silence its banner)
    exec_cell(5, drop_lines(cells[5], (
        "train_loader = make_loader(", "val_loader = make_loader(",
        "history = train_model(", "batch_size=128, shuffle=True)",
        "batch_size=128, shuffle=False)")))  # loaders + train_model
    exec_cell(7, cells[7])                       # save / load model
    exec_cell(8, cells[8])                       # PSTs + minimax
    exec_cell(9, cells[9])                       # evaluate_positions + play_nn
    exec_cell(11, cells[11])                     # show_board + play_game
    exec_cell(12, cells[12])                     # Stockfish engine wrapper
    exec_cell(13, upto(cells[13], "# ---------"))  # calculate_elo + rate_bot
    return ns


def require_model(ns):
    if not os.path.exists(ns["MODEL_SAVE_PATH"]):
        sys.exit("No model found. Train one first:\n    python chess_bot.py train")
    return ns["load_model_weights"](ns["MODEL_SAVE_PATH"])


def make_bot(ns, model, use_minimax=True, depth=2, nn_leaf=False, time_limit=None):
    def bot(fen, player="b"):
        return ns["play_nn"](fen, model, player=player,
                             use_minimax=use_minimax, depth=depth,
                             nn_leaf=nn_leaf, time_limit=time_limit)
    return bot


def cmd_train(ns, args):
    t0 = time.time()
    sample = args.sample if args.sample else None
    X, y, mf, mt = ns["load_dataset"](ns["TRAIN_CSV_PATH"], sample_size=sample)
    idx = ns["np"].arange(len(y))
    idx_train, idx_temp = ns["train_test_split"](
        idx, test_size=0.2, random_state=42, shuffle=True)
    idx_val, idx_test = ns["train_test_split"](
        idx_temp, test_size=0.5, random_state=42, shuffle=True)
    X_tr, y_tr, mf_tr, mt_tr = X[idx_train], y[idx_train], mf[idx_train], mt[idx_train]
    X_va, y_va, mf_va, mt_va = X[idx_val], y[idx_val], mf[idx_val], mt[idx_val]
    X_te = X[idx_test]
    print(f"Encoded in {time.time() - t0:.1f}s | "
          f"Train: {X_tr.shape[0]} | Val: {X_va.shape[0]} | "
          f"Test: {X_te.shape[0]}", flush=True)

    model = ns["ChessEvalModel"](input_channels=19).to(ns["DEVICE"])
    train_loader = ns["make_loader"](
        X_tr, y_tr, mf_tr, mt_tr, batch_size=args.batch, shuffle=True)
    val_loader = ns["make_loader"](
        X_va, y_va, mf_va, mt_va, batch_size=args.batch, shuffle=False)
    save_path = args.out or ns["MODEL_SAVE_PATH"]
    old_save = ns["MODEL_SAVE_PATH"]
    ns["MODEL_SAVE_PATH"] = save_path
    ns["train_model"](model, train_loader, val_loader, epochs=args.epochs)
    ns["save_model"](model, save_path)
    ns["MODEL_SAVE_PATH"] = old_save

    if args.eval:
        df = ns["pd"].read_csv(ns["TRAIN_CSV_PATH"]).sample(
            n=args.eval, random_state=1)
        boards = [ns["chess"].Board(fen=f) for f in df["board"]]
        ys = df["black_score"].values.astype(float)
        preds = ns["evaluate_positions"](boards, model)
        rmse = float(((preds - ys) ** 2).mean()) ** 0.5
        mae = float(abs(preds - ys).mean())
        print(f"Test RMSE: {rmse:.1f} | MAE: {mae:.1f} (n={args.eval})")


def cmd_play(ns, args):
    model = require_model(ns)
    if args.nn:
        ai = make_bot(ns, model, use_minimax=False)
    elif args.nnleaf:
        ai = make_bot(ns, model, use_minimax=True, depth=args.depth, nn_leaf=True,
                      time_limit=args.time)
    else:
        ai = make_bot(ns, model, use_minimax=True, depth=args.depth,
                      time_limit=args.time)
    ns["play_game"](ai, human_player=args.human)


def cmd_rate(ns, args):
    model = require_model(ns)
    if args.nn:
        bot = make_bot(ns, model, use_minimax=False)
    elif args.nnleaf:
        bot = make_bot(ns, model, use_minimax=True, depth=args.depth, nn_leaf=True,
                       time_limit=args.bot_time)
    else:
        bot = make_bot(ns, model, use_minimax=True, depth=args.depth,
                       time_limit=args.bot_time)
    rating, summary = ns["rate_bot"](
        bot, num_games=args.games, time_limit=args.time)
    print(f"Final bot rating: {rating:.0f}")


def cmd_stockfish(ns, args):
    model = require_model(ns)
    if args.nn:
        bot = make_bot(ns, model, use_minimax=False)
    elif args.nnleaf:
        bot = make_bot(ns, model, use_minimax=True, depth=args.depth, nn_leaf=True,
                       time_limit=args.bot_time)
    else:
        bot = make_bot(ns, model, use_minimax=True, depth=args.depth,
                       time_limit=args.bot_time)
    result = ns["play_against_stockfish"](
        bot, bot_is_white=args.as_white,
        stockfish_elo=args.elo, time_limit=args.time)
    print(f"Result: {result} (from White's perspective)")


def cmd_move(ns, args):
    model = require_model(ns)
    board = ns["chess"].Board(fen=args.fen)
    player = "w" if board.turn == ns["chess"].WHITE else "b"
    if args.nn:
        move = ns["play_nn"](args.fen, model, player=player)
    elif args.nnleaf:
        move = ns["play_nn"](args.fen, model, player=player,
                             use_minimax=True, depth=args.depth, nn_leaf=True,
                             time_limit=args.time)
    else:
        move = ns["play_nn"](args.fen, model, player=player,
                             use_minimax=True, depth=args.depth,
                             time_limit=args.time)
    print(move)


def cmd_eval(ns, args):
    model = require_model(ns)
    board = ns["chess"].Board(fen=args.fen)
    player = "w" if board.turn == ns["chess"].WHITE else "b"
    if args.top:
        ns["play_nn"](args.fen, model, player=player,
                      show_move_evaluations=True)
    else:
        print(ns["play_nn"](args.fen, model, player=player))


def main():
    ns = load_namespace()
    p = argparse.ArgumentParser(description="AI Chess Bot CLI")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train the neural model")
    t.add_argument("--sample", type=int, default=0,
                   help="rows to use (0 = all)")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--out", type=str, default="",
                   help="save model to this path instead of chess_model.pt")
    t.add_argument("--eval", type=int, default=2000,
                   help="FENs to measure test RMSE against (0 = skip)")
    t.set_defaults(func=cmd_train)

    p_ = sub.add_parser("play", help="play a game vs the bot")
    p_.add_argument("--human", choices=["w", "b"], default="w",
                    help="which color you play")
    p_.add_argument("--nn", action="store_true",
                    help="use pure neural evaluation (no search)")
    p_.add_argument("--nnleaf", action="store_true",
                    help="minimax with neural leaf eval (slow)")
    p_.add_argument("--depth", type=int, default=6,
                    help="max search depth ceiling")
    p_.add_argument("--time", type=float, default=3.0,
                    help="seconds per bot move (iterative deepening)")
    p_.set_defaults(func=cmd_play)

    r = sub.add_parser("rate", help="estimate Elo by playing Stockfish")
    r.add_argument("--games", type=int, default=3,
                   help="games per Stockfish Elo level")
    r.add_argument("--time", type=float, default=0.1,
                   help="seconds per Stockfish move")
    r.add_argument("--bot-time", type=float, default=3.0,
                   help="seconds per bot move (iterative deepening)")
    r.add_argument("--nn", action="store_true",
                   help="use pure neural evaluation (no search)")
    r.add_argument("--nnleaf", action="store_true",
                   help="minimax with neural leaf eval (slow)")
    r.add_argument("--depth", type=int, default=6,
                   help="max search depth ceiling")
    r.set_defaults(func=cmd_rate)

    s = sub.add_parser("stockfish", help="one game bot vs Stockfish")
    s.add_argument("--elo", type=int, default=1320)
    s.add_argument("--as-white", action="store_true",
                   help="bot plays White")
    s.add_argument("--time", type=float, default=0.1)
    s.add_argument("--bot-time", type=float, default=3.0,
                   help="seconds per bot move (iterative deepening)")
    s.add_argument("--nn", action="store_true",
                   help="use pure neural evaluation (no search)")
    s.add_argument("--nnleaf", action="store_true",
                   help="minimax with neural leaf eval (slow)")
    s.add_argument("--depth", type=int, default=6,
                   help="max search depth ceiling")
    s.set_defaults(func=cmd_stockfish)

    m = sub.add_parser("move", help="best move for a FEN")
    m.add_argument("fen")
    m.add_argument("--nn", action="store_true",
                   help="use pure neural evaluation (no search)")
    m.add_argument("--nnleaf", action="store_true",
                   help="minimax with neural leaf eval (slow)")
    m.add_argument("--depth", type=int, default=6,
                   help="max search depth ceiling")
    m.add_argument("--time", type=float, default=3.0,
                   help="seconds for search (iterative deepening)")
    m.set_defaults(func=cmd_move)

    e = sub.add_parser("eval", help="evaluate a FEN")
    e.add_argument("fen")
    e.add_argument("--top", action="store_true",
                   help="show top move evaluations")
    e.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(ns, args)


if __name__ == "__main__":
    main()