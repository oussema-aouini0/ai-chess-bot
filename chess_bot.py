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
    python chess_bot.py evaluate [--positions N] [--sf-depth N] [--blunder N]
                                 [--depth N] [--bot-time S] [--runs N] [--games N]
                                 [--nn-leaf]
"""

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import time

import chess
import chess.engine

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


# Independent tactical positions (legality vetted against Stockfish).
EVALUATE_TACTICAL_FENS = [
    ("mate1_rook_backrank", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 2"),
    ("mate1_rook_queen", "7k/6pp/8/8/8/8/6PP/5Q1K w - - 0 1"),
    ("mate2_smothered", "6rk/6pp/8/8/8/8/8/5QNK w - - 0 1"),
    ("backrank_def", "6k1/5ppp/8/8/8/8/6PP/5R1K w - - 0 2"),
    ("greek_gift", "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1"),
    ("pin_reveal", "r4rk1/pp1n1ppp/2p5/3b4/8/2N2N2/PP2PPPP/4RRK1 w - - 0 1"),
    ("dbl_check_knight", "r1bq1rk1/ppp2ppp/2n5/3pp3/8/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 1"),
    ("promotion_race", "k7/8/P7/8/8/8/8/1K6 w - - 0 1"),
    ("queenside_attack", "6k1/6pp/8/8/8/8/6PP/5Q1K w - - 0 1"),
    ("skewer_guard", "2q2rk1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1"),
    ("knight_fork_op", "r1bqkb1r/ppp2ppp/2n2n2/3pp3/8/2N2N2/PPPPPPPP/R1BQKB1R w KQkq - 0 1"),
    ("attacking_f7", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"),
    ("bishop_pin_rook", "k7/8/8/8/8/8/4R3/2K5 w - - 0 1"),
]


def _test_split_fens(ns, n, sample_size=20000, seed=42):
    """Replicate the notebook's train/val/test split and return `n` FENs from
    the held-out test slice (same rows the model never trains on)."""
    df = ns["pd"].read_csv(ns["TRAIN_CSV_PATH"])
    df = df.sample(n=min(sample_size, len(df)), random_state=seed)
    idx = ns["np"].arange(len(df))
    _, idx_temp = ns["train_test_split"](
        idx, test_size=0.2, random_state=42, shuffle=True)
    _, idx_test = ns["train_test_split"](
        idx_temp, test_size=0.5, random_state=42, shuffle=True)
    test = df.iloc[idx_test]
    test = test.sample(n=min(n, len(test)), random_state=1)
    return test["board"].tolist()


def _cp_from_score(score):
    """Convert a python-chess PovScore (white POV) to a plain centipawn int,
    mapping mates to +/-100000 so losses stay comparable."""
    w = score.white()
    if w.is_mate():
        return (100000 - w.mate()) if w.mate() > 0 else -(100000 + w.mate())
    return w.score()


def _sf_pool(ns, engine, board, depth=15, multipv=3):
    """Stockfish evaluation of `board`: top-3 moves, best move, and score."""
    res = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    top = [r["pv"][0].uci() for r in res]
    best = top[0]
    best_cp = _cp_from_score(res[0]["score"])
    return best, top, best_cp


def _bot_move(ns, model, fen, player, depth, time_limit, mode="search", nn_leaf=False):
    move = ns["play_nn"](fen, model, player=player, use_minimax=True,
                         depth=depth, time_limit=time_limit, nn_leaf=nn_leaf)
    stats = ns.get("_LAST_SEARCH")
    depth_done = stats.get("depth") if stats else None
    nodes = stats.get("nodes") if stats else None
    return move, depth_done, nodes


def _move_cp(ns, engine, board, move_uci, depth=15):
    """Score of a position AFTER the side to move plays move_uci, from the
    mover's perspective (so ACPL won't flip sign by color)."""
    mover = board.turn
    b = board.copy()
    b.push(ns["chess"].Move.from_uci(move_uci))
    w = _cp_from_score(engine.analyse(b, chess.engine.Limit(depth=depth))["score"])
    return w if mover == ns["chess"].WHITE else -w


def _mate_loss(best_cp, bot_cp, bound=90000):
    """Centipawn loss per position, avoiding the mate-vs-win artifact.

    Mates are mapped to +/-100000, so a position where SF finds a long
    forced mate (mate-in-9 -> +99991) but the bot plays a different,
    still-winning move (+440) would look like a ~99k 'blunder'.  That is
    not an accuracy failure: both moves win the game.  When SF's best is a
    mate we only penalize the bot if it failed to keep a clear win (<50cp).
    Mates on both sides (both convert, or both get mated) score 0."""
    b = abs(best_cp) >= bound
    p = abs(bot_cp) >= bound
    if b and p:
        return 0.0
    if b:
        if best_cp > 0:
            return max(0.0, best_cp - bot_cp) if bot_cp < 50 else 0.0
        return max(0.0, bot_cp - best_cp) if bot_cp > -50 else 0.0
    return max(0.0, best_cp - bot_cp)


def _accuracy_config(ns, engine, model, fens, args, depth, time_limit, label, nn_leaf=False):
    """Score one bot config across the position pool; returns metrics dict."""
    match = top3 = 0
    losses = []
    depths = []
    nodes = []
    for fen in fens:
        board = ns["chess"].Board(fen)
        if board.is_game_over():
            continue
        player = "w" if board.turn == ns["chess"].WHITE else "b"
        bot_uci, depth_done, node_count = _bot_move(
            ns, model, fen, player, depth, time_limit, nn_leaf=nn_leaf)
        if not bot_uci:
            continue
        if depth_done is not None:
            depths.append(depth_done)
        if node_count is not None:
            nodes.append(node_count)
        best, top, _ = _sf_pool(ns, engine, board, args.sf_depth)
        best_cp = _move_cp(ns, engine, board, best, args.sf_depth)
        bot_cp = _move_cp(ns, engine, board, bot_uci, args.sf_depth)
        losses.append(_mate_loss(best_cp, bot_cp))
        if bot_uci == best:
            match += 1
            top3 += 1
        elif bot_uci in top:
            top3 += 1
    n = len(losses)
    blunders = sum(1 for c in losses if c > args.blunder)
    return {
        "label": label,
        "n": n,
        "match": 100.0 * match / n if n else 0.0,
        "top3": 100.0 * top3 / n if n else 0.0,
        "acpl": statistics.mean(losses) if n else 0.0,
        "blunder": 100.0 * blunders / n if n else 0.0,
        "blunder_threshold": args.blunder,
        "avg_d": statistics.mean(depths) if depths else 0.0,
        "min_d": min(depths) if depths else 0,
        "max_d": max(depths) if depths else 0,
        "depth_hist": {d: depths.count(d) for d in sorted(set(depths))},
        "avg_nodes": statistics.mean(nodes) if nodes else 0.0,
    }


def _elo_run(ns, model, args, run_idx):
    bot = make_bot(ns, model, use_minimax=True, depth=args.depth,
                   time_limit=args.bot_time, nn_leaf=args.nn_leaf)
    rating, summary = ns["rate_bot"](bot, num_games=args.games,
                                     time_limit=args.time)
    return rating, summary


def cmd_evaluate(ns, args):
    """Full evaluation: move accuracy + Elo, both vs Stockfish."""
    t0 = time.time()
    model = require_model(ns)

    tactical = [fen for _, fen in EVALUATE_TACTICAL_FENS]
    n_test = max(0, args.positions - len(tactical))
    test_fens = _test_split_fens(ns, n_test)
    # keep the pool balanced & documented: tactical first, then test-split
    fens = tactical + test_fens
    print(f"Position pool: {len(fens)}  "
          f"({len(tactical)} tactical + {len(test_fens)} test-split)", flush=True)

    engine = ns["get_stockfish_engine"](ns["STOCKFISH_PATH"])
    engine.configure({"Threads": 1, "Hash": 64})

    leaf = "NN" if args.nn_leaf else "PST"
    bots = [
        {"label": f"old fixed-depth-2 ({args.depth_old})",
         "depth": args.depth_old, "time": None, "leaf": False},
        {"label": f"new d{args.depth}/t{args.bot_time:g}s {leaf}",
         "depth": args.depth, "time": args.bot_time, "leaf": args.nn_leaf},
        {"label": f"new d{args.depth_deep}/t{args.time_deep:g}s {leaf}",
         "depth": args.depth_deep, "time": args.time_deep, "leaf": args.nn_leaf},
    ]

    print("\n=== Accuracy vs Stockfish " +
          f"(SF depth {args.sf_depth}, blunder>{args.blunder}cp) ===", flush=True)
    header = (f"{'config':42s} {'n':>4} {'match%':>7} {'top3%':>7} {'ACPL':>7} "
              f"{'blunder%':>9} {'avgD':>5} {'dMin':>4} {'dMax':>4} {'nodes':>9}")
    print(header)
    print("-" * len(header))
    for bot in bots:
        m = _accuracy_config(ns, engine, model, fens, args,
                             bot["depth"], bot["time"], bot["label"],
                             nn_leaf=bot["leaf"])
        hist = " ".join(f"d{d}:{c}" for d, c in m["depth_hist"].items())
        print(f"{m['label']:42s} {m['n']:4d} {m['match']:6.1f}% {m['top3']:6.1f}% "
              f"{m['acpl']:7.1f} {m['blunder']:8.1f}% {m['avg_d']:5.2f} "
              f"{m['min_d']:4d} {m['max_d']:4d} {m['avg_nodes']:9.0f}", flush=True)
        print(f"    depth dist: {hist}", flush=True)

    engine.quit()

    ratings = []
    if args.runs > 0:
        print("\n=== Elo vs Stockfish (5 levels, {0} games/level x {1} runs) ==="
              .format(args.games, args.runs), flush=True)
        for i in range(args.runs):
            rating, _summary = _elo_run(ns, model, args, i)
            ratings.append(rating)
            print(f"run {i + 1}: final rating {rating:.0f}", flush=True)
        if len(ratings) > 1:
            mean = statistics.mean(ratings)
            sd = statistics.stdev(ratings) if len(ratings) > 1 else 0.0
            spread = f"mean {mean:.0f} +/- {sd:.0f}"
        else:
            spread = f"rating {ratings[0]:.0f}"
        print(f"\nElo: {spread}  ({'NN' if args.nn_leaf else 'PST'}-leaf bot: depth {args.depth}, "
              f"{args.bot_time:g}s/move, SF {args.time:g}s/move)", flush=True)

    print(f"\nTotal evaluate time: {time.time() - t0:.0f}s", flush=True)


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

    ev = sub.add_parser(
        "evaluate",
        help="move-accuracy metrics + multi-run Elo, both vs Stockfish")
    ev.add_argument("--positions", type=int, default=80,
                    help="total position count (tactical + test-split)")
    ev.add_argument("--sf-depth", type=int, default=15,
                    help="Stockfish analysis depth for accuracy")
    ev.add_argument("--blunder", type=int, default=100,
                    help="centipawn loss threshold that counts as a blunder")
    ev.add_argument("--depth", type=int, default=6,
                    help="bot depth ceiling for the rated/default config")
    ev.add_argument("--bot-time", type=float, default=3.0,
                    help="seconds per bot move (default config)")
    ev.add_argument("--depth-old", type=int, default=2,
                    help="bot depth for the old fixed-depth baseline")
    ev.add_argument("--depth-deep", type=int, default=8,
                    help="bot depth ceiling for the deeper config")
    ev.add_argument("--time-deep", type=float, default=5.0,
                    help="seconds per bot move (deeper config)")
    ev.add_argument("--runs", type=int, default=3,
                    help="independent rate_bot runs for the Elo estimate")
    ev.add_argument("--games", type=int, default=2,
                    help="games per Stockfish Elo level per run")
    ev.add_argument("--time", type=float, default=0.1,
                    help="seconds per Stockfish move during Elo games")
    ev.add_argument("--nn-leaf", action="store_true",
                    help="use the neural net as leaf eval in search "
                         "(TT disabled, slow)")
    ev.set_defaults(func=cmd_evaluate)

    args = p.parse_args()
    args.func(ns, args)


if __name__ == "__main__":
    main()