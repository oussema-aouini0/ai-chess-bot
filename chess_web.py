"""Interactive chess Web app (Flask) for the AI Chess Bot.

Reuses the engine from chess_bot.py (single source of truth): the bot mode,
depth, time budget and model are shared with the CLI and the notebooks.

Features:
- Visual polish: light/dark themes, board skins, check highlight, captured
  pieces, material delta, animations, optional sounds.
- Gameplay: difficulty presets, per-side clocks (with increment), undo,
  resume via localStorage, coach (hint + eval bar), PGN export.
- No-auth arcade leaderboard backed by an optional Postgres (DATABASE_URL).
  Without it (or without psycopg2) the app runs fully; the leaderboard just
  reports itself unavailable.

Run:
    waitress-serve --listen=127.0.0.1:5000 --threads=6 chess_web:app
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import sys
import threading
import time as _time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Limit torch/OpenMP to a few threads early (matters on shared 0.1-CPU hosts).
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("TORCH_THREADS", "2"))
os.environ.setdefault("NUMEXPR_MAX_THREADS", "2")

import chess
import chess.svg
from flask import Flask, jsonify, request, render_template_string

import chess_bot

# psycopg2 is optional at runtime: the leaderboard degrades gracefully if the
# driver or DATABASE_URL is missing.
try:
    import psycopg2
    _PSYCOPG2 = True
except ImportError:
    psycopg2 = None
    _PSYCOPG2 = False

START_FEN = chess.STARTING_FEN
NS = chess_bot.load_namespace()
try:
    NS["torch"].set_num_threads(max(1, int(os.environ.get("TORCH_THREADS", "2"))))
except Exception:
    pass
MODEL = None

# Production config (override anything via env)
PORT = int(os.environ.get("PORT", "5000"))
MAX_CONCURRENT = max(1, int(os.environ.get("MAX_CONCURRENT", "4")))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "4"))
MODES = {"search", "nn", "nnleaf"}
AI_SEM = threading.Semaphore(MAX_CONCURRENT)


# --------------------------------------------------------------------------- #
# Leaderboard (no-auth, arcade style)                                         #
# --------------------------------------------------------------------------- #
DB_URL = os.environ.get("DATABASE_URL", "").strip()
_db_conn = None
_db_lock = threading.Lock()
_lb_cache = {"t": 0.0, "data": None}

LEADERBOARD_BASE = {"win": 100, "draw": 30, "loss": 0}
LEADERBOARD_MULT = {
    "search": {1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5},
    "nn": 1.0,
    "nnleaf": {1: 1.5, 2: 1.8, 3: 2.2},
}
STREAK_BONUS = 10
STREAK_CAP = 100
NAME_MAX = 24


def _score_pts(result, mode, depth, streak):
    base = LEADERBOARD_BASE.get(result, 0)
    m = LEADERBOARD_MULT.get(mode, {})
    mult = m.get(depth, 1.0) if isinstance(m, dict) else float(m)
    return int(round(base * mult)) + min(streak * STREAK_BONUS, STREAK_CAP)


def _db():
    """Return a working connection or None (unavailable: no URL / no driver)."""
    global _db_conn
    if not DB_URL or not _PSYCOPG2:
        return None
    with _db_lock:
        try:
            if _db_conn is None or _db_conn.closed:
                _db_conn = psycopg2.connect(DB_URL, connect_timeout=5)
                _db_conn.autocommit = True
                with _db_conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS leaderboard (
                            id BIGSERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            score INTEGER NOT NULL,
                            result TEXT NOT NULL,
                            mode TEXT NOT NULL,
                            depth INTEGER NOT NULL,
                            streak INTEGER NOT NULL DEFAULT 0,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        )""")
            return _db_conn
        except Exception:
            try:
                if _db_conn is not None:
                    _db_conn.close()
            except Exception:
                pass
            _db_conn = None
            return None


def _clean_name(raw):
    name = "".join(ch for ch in str(raw or "").strip() if ch.isalnum() or ch in " _-")
    return (name or "Player")[:NAME_MAX]


def _leaderboard_data(force=False):
    now = _time.time()
    cached = _lb_cache.get("data")
    if cached is not None and not force and now - _lb_cache["t"] < 15:
        return cached
    conn = _db()
    if conn is None:
        return {"available": False}
    try:
        with _db_lock:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT name, score, result, mode, depth, streak, "
                    "to_char(created_at, 'YYYY-MM-DD HH24:MI') "
                    "FROM leaderboard ORDER BY score DESC, id ASC LIMIT 10")
                top = [dict(zip(("name", "score", "result", "mode", "depth",
                                 "streak", "when"), row)) for row in cur.fetchall()]
                cur.execute(
                    "SELECT name, score, result, "
                    "to_char(created_at, 'YYYY-MM-DD HH24:MI') "
                    "FROM leaderboard ORDER BY id DESC LIMIT 10")
                recent = [dict(zip(("name", "score", "result", "when"), row))
                          for row in cur.fetchall()]
        data = {"available": True, "top": top, "recent": recent}
        _lb_cache.update({"t": now, "data": data})
        return data
    except Exception:
        return {"available": False}


def _clamp_int(value, lo, hi, default):
    try:
        return max(lo, min(int(value), hi))
    except (TypeError, ValueError):
        return default


def _get_model():
    global MODEL
    if MODEL is None:
        MODEL = chess_bot.require_model(NS)
    return MODEL


def _bot_factory(mode, depth):
    model = _get_model()
    if mode == "nn":
        return chess_bot.make_bot(NS, model, use_minimax=False)
    if mode == "nnleaf":
        return chess_bot.make_bot(NS, model, use_minimax=True,
                                  depth=depth, nn_leaf=True)
    return chess_bot.make_bot(NS, model, use_minimax=True, depth=depth)


def _ai_move(board, mode, depth):
    fen = board.fen()
    player = "w" if board.turn == chess.WHITE else "b"
    bot = _bot_factory(mode, depth)
    with AI_SEM:  # cap simultaneous CPU searches across players
        return bot(fen, player)


def _clamp(params):
    """Sanitize mode/depth/side from request args/body (dict-like)."""
    mode = params.get("mode", "search")
    if mode not in MODES:
        mode = "search"
    depth = _clamp_int(params.get("depth", 2), 1, MAX_DEPTH, 2)
    side = params.get("side", "w")
    if side not in ("w", "b"):
        side = "w"
    return mode, depth, side


def _move_pair(before_fen, uci):
    """Find the legal move matching from/to.

    Honors an explicit promotion piece (e.g. 'e7e8n'); falls back to queen
    for legacy 4-char input without a promotion suffix."""
    if not uci or len(uci) not in (4, 5) or not uci.isalnum():
        return None, None
    try:
        b = chess.Board(fen=before_fen)
        m = chess.Move.from_uci(uci)
    except ValueError:
        return None, None
    if len(uci) == 5:
        return (m, b) if m in b.legal_moves else (None, None)
    matches = [x for x in b.legal_moves
               if x.from_square == m.from_square and x.to_square == m.to_square]
    if not matches:
        return None, None
    promos = [x for x in matches if x.promotion]
    if promos:
        queen = next((x for x in promos if x.promotion == chess.QUEEN), promos[0])
        mv = queen
    else:
        mv = matches[0]
    return mv, b


def _status(board):
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        return f"Checkmate — {winner} wins"
    if board.is_stalemate():
        return "Stalemate — draw"
    if board.is_insufficient_material():
        return "Insufficient material — draw"
    if board.is_seventyfive_moves():
        return "75-move rule — draw"
    if board.is_fivefold_repetition():
        return "Fivefold repetition — draw"
    return "Game over — draw"


def _turn_status(board):
    return "Check! Your move" if board.is_check() else "Your turn"


def _draw_board(board):
    """Return (over, result, status). Treats threefold repetition and the
    fifty-move rule as immediate draws (online-play style), in addition to
    python-chess's automatic game-over conditions."""
    if board.is_repetition(3):
        return True, "1/2-1/2", "Draw by threefold repetition"
    if board.is_fifty_moves():
        return True, "1/2-1/2", "Draw by fifty-move rule"
    if board.is_game_over():
        return True, board.result(), _status(board)
    return False, None, None


app = Flask(__name__)


def piece_svg(code):
    color = chess.WHITE if code[0] == "w" else chess.BLACK
    ptype = getattr(chess, {"P": "PAWN", "N": "KNIGHT", "B": "BISHOP",
                            "R": "ROOK", "Q": "QUEEN", "K": "KING"}[code[1]])
    return chess.svg.piece(chess.Piece(ptype, color))


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/pieces/<code>.svg")
def piece(code):
    if len(code) != 2 or code[0] not in "wb" or code[1] not in "PNBRQK":
        return ("bad code", 404)
    return piece_svg(code), 200, {"Content-Type": "image/svg+xml"}


@app.route("/api/legal")
def legal():
    try:
        board = chess.Board(fen=request.args.get("fen", START_FEN))
    except ValueError:
        return jsonify({"error": "bad fen"}), 400
    sq = request.args.get("sq")
    if not sq or len(sq) != 2:
        return jsonify({"squares": []})
    target = chess.parse_square(sq)
    squares = sorted({
        chess.square_name(m.to_square)
        for m in board.legal_moves if m.from_square == target
    })
    return jsonify({"squares": squares})


@app.route("/healthz")
def health():
    return ("ok", 200)


@app.route("/api/eval")
def eval_pos():
    """PST eval, White POV (cp). Powers the eval bar / coach display."""
    try:
        board = chess.Board(fen=request.args.get("fen", START_FEN))
    except ValueError:
        return jsonify({"error": "bad fen"}), 400
    cp = float(NS["evaluate_board"](board, None))
    return jsonify({"cp": cp})


@app.route("/api/hint")
def hint():
    """Best move (PST+quiescence) plus the current eval. Capped depth for the
    free tier's CPU."""
    try:
        board = chess.Board(fen=request.args.get("fen", START_FEN))
    except ValueError:
        return jsonify({"error": "bad fen"}), 400
    depth = _clamp_int(request.args.get("depth", 2), 1, 3, 2)
    with AI_SEM:
        player = "w" if board.turn == chess.WHITE else "b"
        move = NS["play_nn"](board.fen(), None, player=player,
                             use_minimax=True, depth=depth,
                             time_limit=None, nn_leaf=False)
    cp = float(NS["evaluate_board"](board, None))
    if move:
        m = chess.Move.from_uci(move)
        san = board.san(m) if m in board.legal_moves else move
        return jsonify({"move": move, "san": san, "cp": cp, "depth": depth})
    return jsonify({"move": None, "san": None, "cp": cp})


@app.route("/api/new_game")
def new_game():
    mode, depth, side = _clamp(request.args)
    data = {"fen": START_FEN, "bot_first": side != "w",
            "bot_move": None, "bot_san": None, "status": "Your turn"}
    if side != "w":
        b = chess.Board(START_FEN)
        bot_uci = _ai_move(b, mode, depth)
        m = chess.Move.from_uci(bot_uci)
        san = b.san(m)
        b.push(m)
        data["fen"] = b.fen()
        data["bot_move"] = bot_uci
        data["bot_san"] = san
        data["status"] = "Your turn"
    return jsonify(data)


@app.route("/api/move", methods=["POST"])
def make_move():
    body = request.get_json(force=True)
    fen = body.get("fen", START_FEN)
    uci = body.get("uci", "")
    mode, depth, side = _clamp(body)

    try:
        board = chess.Board(fen=fen)
    except ValueError:
        return jsonify({"ok": False, "reason": "Invalid position"}), 400

    mv, _ = _move_pair(fen, uci)
    if mv is None:
        return jsonify({"ok": False, "reason": "Illegal move"}), 200

    human_color = side
    bot_is_white = human_color != "w"

    # Optional soft clock enforcement: the client reports its remaining ms and
    # the server adjudicates only a flag-fall (value <= 0). The heavy lifting
    # stays client-side so the server stays stateless.
    human_ms_left = body.get("human_ms_left")
    if human_ms_left is not None:
        try:
            human_ms_left = int(human_ms_left)
        except (TypeError, ValueError):
            human_ms_left = None
        if human_ms_left is not None and human_ms_left <= 0 and not board.is_game_over():
            result = "1-0" if bot_is_white else "0-1"
            winner = "White" if bot_is_white else "Black"
            return jsonify({"ok": True, "game_over": True, "result": result,
                            "status": f"Game over — timeout, {winner} wins",
                            "fen": fen, "timeout": True})

    san_human = board.san(mv)
    board.push(mv)
    fen_after_human = board.fen()

    resp = {
        "ok": True,
        "fen_after_human": fen_after_human,
        "san_human": san_human,
        "bot_move": None,
        "bot_san": None,
        "game_over": False,
        "result": None,
        "status": None,
    }

    over, result, reason = _draw_board(board)
    if over:
        resp["game_over"] = True
        resp["result"] = result
        resp["status"] = reason
        resp["fen"] = fen_after_human
        return jsonify(resp)

    if board.turn != (chess.WHITE if bot_is_white else chess.BLACK):
        resp["status"] = _turn_status(board)
        resp["fen"] = fen_after_human
        return jsonify(resp)

    # bot replies
    bot_uci = _ai_move(board, mode, depth)
    bot_mv = chess.Move.from_uci(bot_uci)
    san_bot = board.san(bot_mv)
    board.push(bot_mv)
    resp["bot_move"] = bot_uci
    resp["bot_san"] = san_bot
    resp["fen"] = board.fen()
    over, result, reason = _draw_board(board)
    if over:
        resp["game_over"] = True
        resp["result"] = result
        resp["status"] = reason
    else:
        resp["status"] = _turn_status(board)
    return jsonify(resp)


@app.route("/api/leaderboard")
def leaderboard():
    return jsonify(_leaderboard_data())


@app.route("/api/score", methods=["POST"])
def add_score():
    body = request.get_json(force=True)
    conn = _db()
    if conn is None:
        return jsonify({"ok": False, "available": False}), 503
    name = _clean_name(body.get("name"))
    result = str(body.get("result", "draw")).strip()
    if result not in ("win", "draw", "loss"):
        result = "draw"
    mode = str(body.get("mode", "search")).strip()
    if mode not in MODES:
        mode = "search"
    depth = _clamp_int(body.get("depth", 2), 1, MAX_DEPTH, 2)
    streak = _clamp_int(body.get("streak", 0), 0, 99, 0)
    pts = _score_pts(result, mode, depth, streak)
    try:
        with _db_lock:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO leaderboard (name, score, result, mode, depth, streak) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (name, pts, result, mode, depth, streak))
                cur.execute("SELECT COUNT(*) FROM leaderboard WHERE score > %s", (pts,))
                rank = int(cur.fetchone()[0]) + 1
        _leaderboard_data(force=True)
        return jsonify({"ok": True, "score": pts, "rank": rank})
    except Exception:
        return jsonify({"ok": False}), 500


INDEX_HTML = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title> Watcha Chess Bot </title>
<link rel="icon" href="data:,">
<style>
  :root, [data-theme="dark"] {
    --bg:#1e1c1b; --panel:#36312e; --panel2:#433d39; --txt:#efe9e2;
    --sub:#b9b0a6; --sel:#ffd75e; --legal:#7dff6b; --check:#ff5f56;
    --accent:#e8b64c; --shadow:rgba(0,0,0,.5); --border:#555;
    --b-light:#f0d9b5; --b-dark:#b58863; --b-border:#161412;
    --win:#7dff6b; --lose:#ff7a6e; --draw:#ffd75e;
  }
  [data-theme="light"] {
    --bg:#ece7df; --panel:#f5f1ea; --panel2:#e7e0d4; --txt:#281f18;
    --sub:#6d635a; --sel:#e6b422; --legal:#2e9c3b; --check:#d93025;
    --accent:#b5771c; --shadow:rgba(0,0,0,.18); --border:#c9c1b6;
    --b-border:#7a6c5c;
  }
  [data-skin="brown"] { --b-light:#f0d9b5; --b-dark:#b58863; }
  [data-skin="green"] { --b-light:#ebecd0; --b-dark:#739552; }
  [data-skin="blue"]  { --b-light:#dee3e6; --b-dark:#8ca2ad; }

  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--txt); transition:background .2s,color .2s; }
  .wrap { max-width:1100px; margin:0 auto; padding:14px 16px; display:flex;
          gap:18px; flex-wrap:wrap; align-items:flex-start; }
  header { width:100%; display:flex; align-items:center; gap:10px; }
  header h1 { font-size:20px; margin:0; letter-spacing:.4px; }
  header .spacer { flex:1; }
  .iconbtn { width:34px; height:34px; border-radius:8px; border:1px solid var(--border);
             background:var(--panel2); color:var(--txt); cursor:pointer; font-size:15px; }
  .iconbtn:hover { filter:brightness(1.12); }
  .iconbtn.on { outline:2px solid var(--accent); }
  header select { font-size:12px; padding:4px 6px; }

  .board-box { flex:1 1 520px; max-width:660px; display:flex; gap:10px; align-items:stretch; }
  #board-wrap { flex:1; position:relative; }
  #board { display:grid; grid-template-columns:repeat(8,1fr);
           grid-template-rows:repeat(8,1fr); width:100%; aspect-ratio:1/1;
           border:4px solid var(--b-border); border-radius:6px;
           box-shadow:0 10px 28px var(--shadow); user-select:none;
           touch-action:none; background:var(--b-dark); }
  .cell { position:relative; display:flex; align-items:center; justify-content:center; }
  .cell.light { background:var(--b-light); }
  .cell.dark  { background:var(--b-dark); }
  .cell.has-piece { cursor:grab; }
  .cell.legal::after { content:""; position:absolute; width:30%; height:30%;
        border-radius:50%; background:rgba(0,0,0,.22); pointer-events:none; }
  .cell.legal.has-piece::after { width:100%; height:100%;
        border:3px solid rgba(0,0,0,.32); border-radius:50%; background:transparent; }
  .cell.selected { box-shadow:inset 0 0 0 4px var(--sel); }
  .cell.last-from, .cell.last-to { box-shadow:inset 0 0 0 4px var(--sel); }
  .cell.in-check img { filter:drop-shadow(0 0 6px var(--check)); }
  .cell img { width:92%; height:92%; pointer-events:none; object-fit:contain; }
  .coord { position:absolute; font-size:9px; opacity:.55; pointer-events:none; }
  .coord.file { bottom:1px; right:2px; }
  .coord.rank { top:1px; left:2px; }
  #hint-layer { position:absolute; inset:0; pointer-events:none; z-index:3; }
  #thinking { position:absolute; inset:0; display:none; align-items:center;
          justify-content:center; background:rgba(0,0,0,.5); z-index:4;
          border-radius:4px; font-size:16px; font-weight:600; letter-spacing:.5px;
          text-align:center; padding:8px; }
  #evalwrap { flex:0 0 18px; border:2px solid var(--b-border); border-radius:6px;
          overflow:hidden; position:relative; background:#444; }
  #evalbar { position:absolute; bottom:0; left:0; right:0; background:var(--b-light);
          height:50%; transition:height .3s; }
  #evalnum { position:absolute; bottom:2px; left:0; right:0; text-align:center;
          font-size:8px; color:#fff; text-shadow:0 1px 2px #000; z-index:2;
          word-break:break-all; }

  .panel { flex:0 1 350px; background:var(--panel); border-radius:10px;
           padding:14px 16px; box-shadow:0 6px 20px var(--shadow); }
  .panel h2 { font-size:14px; margin:0 0 8px; color:var(--sub); }
  .row { display:flex; align-items:center; gap:8px; margin:7px 0; flex-wrap:wrap; }
  label { font-size:13px; opacity:.9; white-space:nowrap; }
  select, button { font:inherit; padding:5px 8px; border-radius:7px;
          border:1px solid var(--border); background:var(--panel2); color:var(--txt); }
  button { cursor:pointer; }
  button:hover { filter:brightness(1.12); }
  .seg { display:inline-flex; border:1px solid var(--border); border-radius:7px; overflow:hidden; }
  .seg button { border:none; border-radius:0; background:var(--panel2); padding:4px 9px; }
  .seg button.active { background:var(--accent); color:#1e1c1b; font-weight:600; }
  .clocks { display:flex; gap:8px; margin:4px 0; }
  .clock { flex:1; text-align:center; padding:4px 6px; border-radius:7px;
           background:var(--panel2); border:1px solid var(--border); font-variant-numeric:tabular-nums; }
  .clock.active { outline:2px solid var(--accent); }
  .clock .who { font-size:10px; opacity:.7; display:block; }
  .clock .tm { font-size:14px; font-weight:700; letter-spacing:.5px; }
  .clock.low .tm { color:var(--lose); }
  #status { min-height:22px; padding:6px 0; font-weight:600; color:var(--accent); }
  .captured { display:flex; align-items:center; font-size:14px; letter-spacing:1px;
          min-height:22px; line-height:1.5; font-family:ui-monospace,Consolas,monospace; }
  .captured .spacer { flex:1; }
  .mat { font-size:11px; opacity:.8; margin:0 6px; }
  .tabs { display:flex; gap:6px; margin:8px 0 6px; }
  .tabs button { padding:4px 12px; }
  .tabs button.active { background:var(--accent); color:#1e1c1b; font-weight:600; }
  .tabbody { display:none; }
  .tabbody.active { display:block; }
  #moves { background:var(--panel2); border:1px solid var(--border); border-radius:6px;
           padding:8px; height:250px; overflow:auto; font-family:ui-monospace,
           Consolas,monospace; font-size:13px; line-height:1.65; }
  #moves div { display:grid; grid-template-columns:32px 1fr 1fr; gap:2px; }
  #moves .last { color:var(--accent); font-weight:600; }
  #moves .end { grid-column:1 / -1; color:var(--sub); font-style:italic; }
  table.lb { width:100%; border-collapse:collapse; font-size:12.5px; }
  table.lb th, table.lb td { padding:4px 6px; text-align:left; border-bottom:1px solid var(--border); }
  table.lb th { color:var(--sub); font-weight:600; font-size:11px; }
  table.lb .rank { color:var(--accent); font-weight:700; }
  .res-win { color:var(--win); } .res-loss { color:var(--lose); } .res-draw { color:var(--draw); }
  .hint { font-size:11.5px; opacity:.75; margin-top:8px; line-height:1.5; }
  .toast { position:fixed; left:50%; bottom:22px; transform:translateX(-50%) translateY(80px);
           background:var(--accent); color:#1e1c1b; font-weight:600; padding:8px 16px;
           border-radius:8px; box-shadow:0 6px 20px var(--shadow); opacity:0;
           transition:opacity .25s, transform .25s; z-index:700; }
  .toast.show { opacity:1; transform:translateX(-50%) translateY(0); }

  #modal { position:fixed; inset:0; background:rgba(0,0,0,.6); z-index:600;
           display:none; align-items:center; justify-content:center; padding:16px; }
  #modal.open { display:flex; }
  #modal-box { background:var(--panel); border-radius:12px; padding:20px 22px;
               max-width:340px; width:100%; box-shadow:0 14px 40px var(--shadow);
               text-align:center; }
  #modal-box h3 { margin:0 0 6px; font-size:20px; }
  #modal-box .result { font-size:14px; color:var(--sub); margin-bottom:14px; word-break:break-word; }
  #modal-box .actions { display:flex; gap:8px; justify-content:center; flex-wrap:wrap; }
  #modal-box input { width:100%; margin:8px 0; padding:7px; text-align:center;
          border:1px solid var(--border); border-radius:7px; background:var(--panel2); color:var(--txt); }
  #modal-box .saveq { font-size:12px; opacity:.8; margin-top:4px; }

  #resume-bar { position:fixed; left:50%; top:12px; transform:translateX(-50%);
                background:var(--accent); color:#1e1c1b; font-weight:600;
                padding:8px 14px; border-radius:8px; display:none; gap:10px;
                box-shadow:0 6px 20px var(--shadow); z-index:500; align-items:center; }
  #resume-bar button { background:#1e1c1b; color:var(--txt); border:none; padding:3px 10px; border-radius:6px; }

  #promo { position:fixed; inset:0; background:rgba(0,0,0,.55); z-index:650;
           display:none; align-items:center; justify-content:center; }
  #promo.open { display:flex; }
  #promo-box { display:flex; gap:8px; background:var(--panel); padding:12px;
               border-radius:10px; box-shadow:0 8px 24px var(--shadow); }
  #promo-box button { width:64px; height:64px; padding:0; background:var(--panel2);
           border:1px solid var(--border); border-radius:8px; cursor:pointer;
           display:flex; align-items:center; justify-content:center; }
  #promo-box img { width:56px; height:56px; pointer-events:none; }

  @media (max-width:860px) {
    .board-box { flex-basis:100%; }
    .panel { flex-basis:100%; }
  }
</style>
</head>
<body data-skin="brown">
<div class="wrap">
  <header>
    <h1>♞ Watcha </h1>
    <div class="spacer"></div>
    <button class="iconbtn" id="btn-eval" title="Eval bar on/off">📊</button>
    <button class="iconbtn" id="btn-sound" title="Sound on/off">🔇</button>
    <select id="skin" title="Board colors">
      <option value="brown">Brown</option>
      <option value="green">Green</option>
      <option value="blue">Blue</option>
    </select>
    <button class="iconbtn" id="btn-theme" title="Theme">🌙</button>
  </header>

  <div class="board-box">
    <div id="evalwrap"><div id="evalnum">~</div><div id="evalbar"></div></div>
    <div id="board-wrap">
      <div id="board"></div>
      <svg id="hint-layer"></svg>
      <div id="thinking">AI is thinking…</div>
    </div>
  </div>

  <div class="panel">
    <div class="row">
      <label>You play</label>
      <select id="side"><option value="w">White</option><option value="b">Black</option></select>
      <label>Mode</label>
      <select id="mode">
        <option value="search">Search (minimax)</option>
        <option value="nn">Neural (policy)</option>
        <option value="nnleaf">NN-leaf (slow)</option>
      </select>
    </div>
    <div class="row">
      <label>Difficulty</label>
      <div class="seg" id="diff">
        <button data-d="1">Easy</button>
        <button data-d="2" class="active">Normal</button>
        <button data-d="3">Hard</button>
      </div>
      <label>Time</label>
      <select id="time">
        <option value="0">Off</option>
        <option value="5 0">5+0</option>
        <option value="10 5">10+5</option>
        <option value="15 10">15+10</option>
      </select>
    </div>
    <div class="row">
      <button id="btn-new">New game</button>
      <button id="btn-undo" title="Take back your last move">Undo</button>
      <button id="btn-hint">Hint</button>
      <button id="btn-pgn" title="Copy PGN">PGN</button>
    </div>

    <div class="clocks">
      <div class="clock" id="clk-w"><span class="who">White</span><span class="tm">--:--</span></div>
      <div class="clock" id="clk-b"><span class="who">Black</span><span class="tm">--:--</span></div>
    </div>
    <div id="status">Start a game to play</div>
    <div class="captured"><span id="cap-w"></span><span class="mat" id="mat-w"></span>
      <span class="spacer"></span><span class="mat" id="mat-b"></span><span id="cap-b"></span>
    </div>

    <div class="tabs">
      <button id="tab-moves" class="active">Moves</button>
      <button id="tab-lb">Leaderboard</button>
    </div>
    <div class="tabbody active" id="tab-moves-body">
      <div id="moves"></div>
    </div>
    <div class="tabbody" id="tab-lb-body">
      <h2>Top scores</h2>
      <table class="lb" id="lb-top"><thead><tr><th>#</th><th>Name</th><th>Score</th><th>Result</th><th>When</th></tr></thead><tbody></tbody></table>
      <h2>Recent</h2>
      <table class="lb" id="lb-recent"><thead><tr><th>Name</th><th>Score</th><th>Result</th><th>When</th></tr></thead><tbody></tbody></table>
      <div class="hint" id="lb-note"></div>
    </div>
    <div class="hint">Drag pieces to move — or tap a piece then a destination.
      Promotions ask on the last rank. Hint (H) shows the engine's best move.</div>
  </div>
</div>

<div id="resume-bar"><span>Resume your last game?</span><button onclick="resumeGame()">Resume</button><button onclick="dismissResume()">Dismiss</button></div>

<div id="modal"><div id="modal-box">
  <h3 id="m-title">Game over</h3>
  <div class="result" id="m-result"></div>
  <div class="actions">
    <button onclick="newGame()">Rematch</button>
    <button onclick="copyPGN()">Copy PGN</button>
    <button onclick="closeModal()">Close</button>
  </div>
  <div class="saveq" id="m-save"></div>
</div></div>

<div id="promo"><div id="promo-box"></div></div>
<div class="toast" id="toast"></div>

<script>
const files = ['a','b','c','d','e','f','g','h'];
const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const GLYPH = {K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙',k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};
const PIECE_VAL = {P:100,N:320,B:330,R:500,Q:900,K:0};

let state = {
  fen: START_FEN, myColor: 'w', mode: 'search', depth: 2,
  timeMin: 0, incSec: 0, moves: [], sanSeq: [], checkpoints: [],
  last: null, over: false, result: null, status: 'Start a game to play',
  thinking: false, selected: null, targets: [],
  humanMs: 0, botMs: 0, clockTurn: 'w',
  hint: null,
};
const settings = {
  theme: localStorage.getItem('cs_theme') || 'dark',
  skin: localStorage.getItem('cs_skin') || 'brown',
  sound: (localStorage.getItem('cs_sound') || '1') === '1',
  eval: (localStorage.getItem('cs_eval') || '0') === '1',
};
let streak = (parseInt(localStorage.getItem('cs_streak') || '0', 10)) || 0;
const evalCache = new Map();

// ---------------------------------------------------------------- helpers ---
const $ = id => document.getElementById(id);
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}
function esc(s) {
  return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
const myBotColor = () => state.myColor === 'w' ? 'b' : 'w';

function applySettings() {
  document.documentElement.setAttribute('data-theme', settings.theme);
  document.body.setAttribute('data-skin', settings.skin);
  $('skin').value = settings.skin;
  $('btn-theme').textContent = settings.theme === 'dark' ? '🌙' : '☀️';
  $('btn-sound').textContent = settings.sound ? '🔊' : '🔇';
  $('btn-sound').classList.toggle('on', settings.sound);
  $('btn-eval').classList.toggle('on', settings.eval);
}
function saveSettings() {
  localStorage.setItem('cs_theme', settings.theme);
  localStorage.setItem('cs_skin', settings.skin);
  localStorage.setItem('cs_sound', settings.sound ? '1' : '0');
  localStorage.setItem('cs_eval', settings.eval ? '1' : '0');
}

// ----------------------------------------------------------------- sounds ---
let actx = null;
function beep(freq, dur, type, when) {
  if (!settings.sound) return;
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    const t = actx.currentTime + (when || 0);
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = type || 'sine'; o.frequency.value = freq;
    g.gain.setValueAtTime(.12, t); g.gain.exponentialRampToValueAtTime(.0001, t + dur);
    o.connect(g).connect(actx.destination);
    o.start(t); o.stop(t + dur);
  } catch (e) {}
}
const sndMove = () => beep(430, .06);
const sndCapture = () => { beep(260, .09, 'square'); beep(180, .12, 'square', .02); };
const sndCheck = () => { beep(700, .09); beep(520, .12, 'sine', .1); };
const sndWin = () => [523, 659, 784, 1047].forEach((f, i) => beep(f, .16, 'sine', i * .12));
const sndLose = () => [392, 311, 262, 196].forEach((f, i) => beep(f, .18, 'sine', i * .13));
const sndDraw = () => { beep(440, .14); beep(440, .14, 'sine', .16); };

// ---------------------------------------------------------------- board -----
function cellSquares() {
  const layout = [];
  for (let r = 0; r < 8; r++) {
    const rank = state.myColor === 'w' ? 8 - r : 1 + r;
    for (let c = 0; c < 8; c++) {
      const file = state.myColor === 'w' ? files[c] : files[7 - c];
      layout.push(file + rank);
    }
  }
  return layout;
}
function fenBoard() { return state.fen.split(' ')[0]; }
function fenTurn() { return state.fen.split(' ')[1]; }
function fenIndex(sq) { const f = sq.charCodeAt(0) - 97; const r = 8 - (+sq[1]); return r * 8 + f; }
function squareName(sq) { return files[sq % 8] + (8 - Math.floor(sq / 8)); }
function pieceAt(sq) {
  const idx = fenIndex(sq); const place = fenBoard(); let n = 0;
  for (const ch of place) {
    if (ch === '/') continue;
    if (/\d/.test(ch)) { n += +ch; continue; }
    if (n === idx) return ch;
    n++;
  }
  return null;
}

function parseFen() {
  const place = fenBoard(); const map = {}; let idx = 0;
  for (const ch of place) {
    if (ch === '/') continue;
    if (/\d/.test(ch)) { idx += +ch; continue; }
    map[idx] = ch; idx++;
  }
  const kings = {}; const mats = { w: 0, b: 0 }; const pieceMats = { w: {}, b: {} };
  for (let sq = 0; sq < 64; sq++) {
    const ch = map[sq]; if (!ch) continue;
    const color = ch === ch.toUpperCase() ? 'w' : 'b';
    const pt = ch.toUpperCase();
    if (pt === 'K') kings[color] = sq;
    const v = PIECE_VAL[pt] || 0;
    mats[color] += v;
    if (pt !== 'K') pieceMats[color][pt] = (pieceMats[color][pt] || 0) + 1;
  }
  const captured = { w: '', b: '' };
  const order = ['Q', 'R', 'B', 'N', 'P'];
  for (const color of ['w', 'b']) {
    for (const pt of order) {
      const have = pieceMats[color][pt] || 0;
      const need = { Q: 1, R: 2, B: 2, N: 2, P: 8 }[pt];
      const n = need - have; if (n <= 0) continue;
      const victim = color === 'w' ? pt : pt.toLowerCase();
      captured[color === 'w' ? 'b' : 'w'] += GLYPH[victim].repeat(n);
    }
  }
  const turn = fenTurn() === 'w' ? 'w' : 'b';
  const inCheck = isKingAttacked(map, kings[turn], turn === 'w' ? 'b' : 'w');
  return { map, kings, mats, captured, delta: mats.w - mats.b, turn, inCheck, pieceMats };
}

function isKingAttacked(map, kingSq, attackerColor) {
  // Brute-force attack detection for the given king square (sliding lines,
  // knight, king, pawn attack squares) from the attacker's perspective.
  for (const [sq, ch] of Object.entries(map)) {
    const s = +sq; if (!ch) continue;
    const c = ch === ch.toUpperCase() ? 'w' : 'b';
    if (c !== attackerColor) continue;
    const pt = ch.toUpperCase();
    const df = s % 8 - kingSq % 8, dr = Math.floor(s / 8) - Math.floor(kingSq / 8);
    if (pt === 'P') {
      // a pawn attacks diagonally toward its forward rank
      if (Math.abs(df) === 1 && dr === (attackerColor === 'w' ? 1 : -1)) return true;
      continue;
    }
    if (pt === 'N') {
      if ((Math.abs(df) === 2 && Math.abs(dr) === 1) || (Math.abs(df) === 1 && Math.abs(dr) === 2)) return true;
      continue;
    }
    if (pt === 'K') {
      if (Math.abs(df) <= 1 && Math.abs(dr) <= 1) return true;
      continue;
    }
    const slideR = [0, -1, 0, 1], slideC = [-1, 0, 1, 0];
    const diagR = [-1, -1, 1, 1], diagC = [-1, 1, -1, 1];
    let lines;
    if (pt === 'R') lines = [slideR, slideC];
    else if (pt === 'B') lines = [diagR, diagC];
    else lines = [slideR, slideC, diagR, diagC];
    for (let L = 0; L < lines[0].length; L++) {
      let f = s % 8, r = Math.floor(s / 8);
      while (true) {
        f += lines[1][L]; r += lines[0][L];
        if (f < 0 || f > 7 || r < 0 || r > 7) break;
        const cell = r * 8 + f;
        if (cell === kingSq) return true;
        if (map[cell]) break;
      }
    }
  }
  return false;
}

function render() {
  const layout = cellSquares();
  const box = $('board'); box.innerHTML = '';
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const sq = layout[r * 8 + c];
      const cell = document.createElement('div');
      cell.className = 'cell ' + ((r + c) % 2 === 0 ? 'light' : 'dark');
      cell.id = 'sq-' + sq; cell.dataset.square = sq;
      const pc = pieceAt(sq);
      if (pc) {
        const code = (pc === pc.toUpperCase() ? 'w' : 'b') + pc.toUpperCase();
        const img = document.createElement('img');
        img.src = `/pieces/${code}.svg`; img.alt = code;
        cell.appendChild(img); cell.classList.add('has-piece');
      }
      if (r === 7) cell.appendChild(coord('file', sq[0]));
      if (c === 0) cell.appendChild(coord('rank', sq[1]));
      cell.addEventListener('click', () => clickCell(sq));
      box.appendChild(cell);
    }
  }
  clearHighlights();
  highlightLast();
  const P = parseFen();
  if (P.inCheck && P.kings[P.turn] !== undefined) {
    const c = $('sq-' + squareName(P.kings[P.turn]));
    if (c) c.classList.add('in-check');
  }
  renderCaptured(P);
  renderEval();
  drawHint();
}
function coord(cls, text) {
  const s = el('span', 'coord ' + cls);
  s.textContent = text;
  return s;
}
function renderCaptured(P) {
  $('cap-w').textContent = P.captured.w;
  $('cap-b').textContent = P.captured.b;
  const d = P.delta;
  $('mat-w').textContent = d > 0 ? '+' + Math.round(d) : '';
  $('mat-b').textContent = d < 0 ? '+' + Math.round(-d) : '';
}
function clearHighlights() {
  document.querySelectorAll('.cell.selected,.cell.legal').forEach(c =>
    c.classList.remove('selected', 'legal'));
}
function highlightLast() {
  document.querySelectorAll('.last-from,.last-to').forEach(c =>
    c.classList.remove('last-from', 'last-to'));
  if (!state.last) return;
  const f = $('sq-' + state.last.from), t = $('sq-' + state.last.to);
  if (f) f.classList.add('last-from');
  if (t) t.classList.add('last-to');
}

// ----------------------------------------------------------------- eval ----
function evalKey(fen) { return fen.replace(/ [0-9]+ [0-9]+$/, '') + ' ' + fen.split(' ')[1]; }
function renderEval() {
  const bar = $('evalbar'), num = $('evalnum');
  if (!settings.eval || state.over) { bar.style.height = '50%'; num.textContent = '~'; return; }
  const key = evalKey(state.fen);
  let cp = evalCache.get(key);
  if (cp === undefined) {
    const fen = state.fen;
    fetch('/api/eval?fen=' + encodeURIComponent(fen)).then(r => r.json()).then(d => {
      if (d && typeof d.cp === 'number' && fen === state.fen) { evalCache.set(key, d.cp); renderEval(); }
    }).catch(() => {});
    return;
  }
  num.textContent = cp === 0 ? '0' : (cp > 0 ? '+' + Math.round(cp) : Math.round(cp));
  const side = fenTurn() === 'w' ? 1 : -1;
  const signed = Math.max(-1, Math.min(1, (cp * side) / 1500));
  bar.style.height = Math.round(50 + signed * 50) + '%';
}

// ---------------------------------------------------------- hint / arrows ---
function drawHint() {
  const layer = $('hint-layer');
  layer.innerHTML = '';
  if (!state.hint || state.over) return;
  const [from, to] = state.hint;
  const layout = cellSquares();
  const fromIdx = layout.indexOf(from), toIdx = layout.indexOf(to);
  if (fromIdx < 0 || toIdx < 0) return;
  const x1 = (fromIdx % 8 + .5) / 8, y1 = (Math.floor(fromIdx / 8) + .5) / 8;
  const x2 = (toIdx % 8 + .5) / 8, y2 = (Math.floor(toIdx / 8) + .5) / 8;
  const len = Math.hypot(x2 - x1, y2 - y1) || 1;
  const ux = (x2 - x1) / len, uy = (y2 - y1) / len;
  layer.setAttribute('viewBox', '0 0 8 8');
  layer.setAttribute('preserveAspectRatio', 'none');
  const ah = .14, aw = .10;
  const hx = x2 - ux * ah, hy = y2 - uy * ah;
  const poly = [[x1, y1], [hx - uy * aw, hy + ux * aw], [x2, y2], [hx + uy * aw, hy - ux * aw]];
  const pts = poly.map(p => p[0].toFixed(3) + ',' + p[1].toFixed(3)).join(' ');
  const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  const polyEl = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
  polyEl.setAttribute('points', pts);
  polyEl.setAttribute('fill', 'rgba(125,255,107,.5)');
  polyEl.setAttribute('stroke', '#7dff6b'); polyEl.setAttribute('stroke-width', '.03');
  g.appendChild(polyEl); layer.appendChild(g);
}
function requestHint() {
  if (state.over || state.thinking || !state.fen) return;
  setThinking(true);
  fetch(`/api/hint?fen=${encodeURIComponent(state.fen)}&depth=${state.depth}`)
    .then(r => r.json())
    .then(d => {
      if (d && d.move) {
        state.hint = [d.move.slice(0, 2), d.move.slice(2, 4)];
        if (typeof d.cp === 'number') evalCache.set(evalKey(state.fen), d.cp);
        drawHint();
        renderEval();
      }
    })
    .catch(() => {})
    .finally(() => setThinking(false));
}

// ------------------------------------------------------------ interactions --
let drag = null;
function cellPx() { const c = document.querySelector('.cell'); return c ? c.getBoundingClientRect().width : 64; }

$('board').addEventListener('mousedown', e => {
  if (state.thinking || state.over || e.button !== 0) return;
  const cell = e.target.closest('.cell');
  if (!cell) return;
  const sq = cell.dataset.square, pc = pieceAt(sq);
  if (!pc || !isUserPiece(pc)) return;
  e.preventDefault();
  const img = cell.querySelector('img'); if (!img) return;
  const clone = img.cloneNode(); const size = cellPx();
  clone.style.cssText = 'position:fixed;z-index:500;pointer-events:none;' +
    `width:${size}px;height:${size}px;left:${e.clientX - size / 2}px;top:${e.clientY - size / 2}px;`;
  document.body.appendChild(clone);
  drag = { from: sq, clone, size };
});
window.addEventListener('mousemove', e => {
  if (!drag) return;
  drag.clone.style.left = (e.clientX - drag.size / 2) + 'px';
  drag.clone.style.top = (e.clientY - drag.size / 2) + 'px';
});
window.addEventListener('mouseup', e => {
  if (!drag) return;
  const { from, clone } = drag; drag = null; clone.remove();
  const cell = e.target.closest ? e.target.closest('.cell') : null;
  const to = cell && cell.dataset.square;
  if (to && to !== from) sendMove(from, to);
});

function isUserPiece(pc) {
  const color = pc === pc.toUpperCase() ? 'w' : 'b';
  return color === state.myColor && fenTurn() === state.myColor;
}
function clearSel() { state.selected = null; state.targets = []; clearHighlights(); }

async function clickCell(sq) {
  if (state.thinking || state.over) return;
  if (state.selected) {
    if (sq === state.selected) { clearSel(); return; }
    if (state.targets.includes(sq)) { sendMove(state.selected, sq); return; }
  }
  const pc = pieceAt(sq);
  if (!pc || !isUserPiece(pc)) { clearSel(); return; }
  state.selected = sq;
  const res = await fetch(`/api/legal?fen=${encodeURIComponent(state.fen)}&sq=${sq}`);
  const data = await res.json();
  state.targets = data.squares || [];
  clearHighlights();
  const sel = $('sq-' + sq); if (sel) sel.classList.add('selected');
  state.targets.forEach(t => { const c = $('sq-' + t); if (c) c.classList.add('legal'); });
}

// ---------------------------------------------------------------- clocks ---
function fmtMs(ms) {
  if (ms >= 3600000) {
    const h = Math.floor(ms / 3600000), m = Math.floor(ms % 3600000 / 60000);
    return h + ':' + String(m).padStart(2, '0');
  }
  const m = Math.floor(ms / 60000), s = Math.floor(ms % 60000 / 1000);
  return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
}
function renderClocks() {
  const wc = $('clk-w'), bc = $('clk-b');
  if (!state.timeMin) {
    wc.querySelector('.tm').textContent = '--:--';
    bc.querySelector('.tm').textContent = '--:--';
    wc.classList.remove('active', 'low'); bc.classList.remove('active', 'low');
    return;
  }
  const wMs = state.myColor === 'w' ? state.humanMs : state.botMs;
  const bMs = state.myColor === 'b' ? state.humanMs : state.botMs;
  wc.querySelector('.tm').textContent = fmtMs(wMs);
  bc.querySelector('.tm').textContent = fmtMs(bMs);
  const live = !state.over && state.clockTurn;
  wc.classList.toggle('active', live && state.clockTurn === 'w');
  bc.classList.toggle('active', live && state.clockTurn === 'b');
  wc.classList.toggle('low', wMs <= 60000);
  bc.classList.toggle('low', bMs <= 60000);
}
setInterval(() => {
  if (!state.timeMin || state.over) return;
  if (state.clockTurn === 'b') state.botMs = Math.max(0, state.botMs - 250);
  else state.humanMs = Math.max(0, state.humanMs - 250);
  if (state.humanMs <= 0) onHumanTime();
  else if (state.botMs <= 0) onBotTime();
  renderClocks();
}, 250);
function onHumanTime() {
  if (!state.timeMin || state.over) return;
  finishGame(state.myColor === 'w' ? '0-1' : '1-0', 'You ran out of time');
}
function onBotTime() {
  if (!state.timeMin || state.over) return;
  finishGame(state.myColor === 'w' ? '1-0' : '0-1', 'Bot ran out of time');
}

// --------------------------------------------------------------- game flow --
function setThinking(t) { state.thinking = t; const d = $('thinking'); if (d) d.style.display = t ? 'flex' : 'none'; }
function setStatus(t) { state.status = t; const s = $('status'); if (s) s.textContent = t; }
function glyphFor(san) {
  if (!san) return '';
  const m = san.match(/^([NBRQK])/);
  return m ? GLYPH[m[1]] + san.slice(1) : san;
}
function addMove(no, white, black) {
  const box = $('moves');
  const row = el('div');
  row.append(el('span', '', no + '.'), el('span', '', glyphFor(white)), el('span', '', glyphFor(black)));
  box.querySelectorAll('.last').forEach(x => x.classList.remove('last'));
  row.classList.add('last');
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
}
function addEndMove(result) {
  const box = $('moves');
  box.appendChild(el('div', 'end', result || 'Done'));
  box.scrollTop = box.scrollHeight;
}
function clearMovesList() { $('moves').innerHTML = ''; }
function rebuildMoves() {
  const seq = state.sanSeq || [];
  state.moves = [];
  for (let i = 0; i < seq.length; i += 2) {
    state.moves.push({ n: i / 2 + 1, white: seq[i], black: seq[i + 1] || '' });
  }
  clearMovesList();
  state.moves.forEach(m => addMove(m.n, m.white, m.black));
}

function persistGame() {
  localStorage.setItem('cs_game', JSON.stringify({
    fen: state.fen, myColor: state.myColor, mode: state.mode, depth: state.depth,
    timeMin: state.timeMin, incSec: state.incSec,
    sanSeq: state.sanSeq, checkpoints: state.checkpoints, last: state.last,
    over: state.over, result: state.result,
    humanMs: state.humanMs, botMs: state.botMs, clockTurn: state.clockTurn,
    status: state.status,
  }));
}
function closeGame() { try { localStorage.removeItem('cs_game'); } catch (e) {} }

function newGame() {
  state.hint = null; drawHint();
  state.myColor = $('side').value;
  state.mode = $('mode').value;
  const active = document.querySelector('#diff button.active');
  state.depth = active ? +active.dataset.d : 2;
  const [tm, inc] = $('time').value.split(' ').map(x => +x || 0);
  state.timeMin = tm; state.incSec = inc;
  if (state.timeMin) state.humanMs = state.botMs = state.timeMin * 60000;
  else state.humanMs = state.botMs = 0;
  state.moves = []; state.sanSeq = []; state.checkpoints = [];
  state.last = null; state.over = false; state.result = null;
  state.selected = null; state.targets = [];
  state.fen = START_FEN;
  state.clockTurn = state.myColor;
  clearMovesList(); closeModal(); saveSettings();
  render(); renderClocks();
  const humanFirst = state.myColor === 'w';
  setStatus(humanFirst ? 'New game — your move' : 'Bot to move first…');
  if (!humanFirst) {
    setThinking(true);
    state.clockTurn = myBotColor();
    renderClocks();
    fetch(`/api/new_game?side=${state.myColor}&mode=${state.mode}&depth=${state.depth}`)
      .then(r => r.json())
      .then(d => {
        state.last = d.bot_move ? { from: d.bot_move.slice(0, 2), to: d.bot_move.slice(2, 4) } : null;
        state.fen = d.fen;
        state.sanSeq.push(d.bot_san || d.bot_move);
        state.checkpoints.push({ fen: START_FEN, what: 'bot-first', sanH: null, sanB: d.bot_san });
        state.clockTurn = state.myColor;
        rebuildMoves();
        persistGame();
        render(); renderClocks(); setStatus('Your turn');
        setThinking(false);
      })
      .catch(() => setThinking(false));
  } else {
    persistGame();
  }
}

async function sendMove(from, to) {
  if (state.thinking || state.over) return;
  const pc = pieceAt(from);
  let uci = from + to;
  if (pc && pc.toLowerCase() === 'p') {
    const toRank = to[1];
    if ((pc === 'P' && toRank === '8') || (pc === 'p' && toRank === '1')) {
      const color = pc === pc.toUpperCase() ? 'w' : 'b';
      const promo = await pickPromotion(color);
      if (!promo) return;
      uci = from + to + promo;
    }
  }
  const beforeFen = state.fen;
  const wasCapture = !!parseFen().map[fenIndex(to)];
  const body = {
    fen: state.fen, uci, mode: state.mode, depth: state.depth, side: state.myColor,
  };
  if (state.timeMin) body.human_ms_left = state.humanMs;
  state.clockTurn = myBotColor();
  renderClocks();
  setThinking(true);
  try {
    const res = await fetch('/api/move', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!data.ok) {
      clearSel();
      const st = $('status');
      st.textContent = data.reason || 'Illegal move';
      st.classList.remove('shake'); void st.offsetWidth; st.classList.add('shake');
      setThinking(false);
      return;
    }
    if (data.timeout) { finishGame(data.result, data.status || 'Timeout'); return; }
    state.last = data.bot_move ? { from: data.bot_move.slice(0, 2), to: data.bot_move.slice(2, 4) }
                               : { from, to };
    state.fen = data.fen;
    state.sanSeq.push(data.san_human);
    if (data.bot_san) state.sanSeq.push(data.bot_san);
    state.checkpoints.push({ fen: beforeFen, what: 'human', sanH: data.san_human, sanB: data.bot_san });
    if (state.timeMin) {
      const cap = state.timeMin * 60000 + state.incSec * 1000;
      state.humanMs = Math.min(state.humanMs + state.incSec * 1000, cap);
      if (data.bot_san) state.botMs = Math.min(state.botMs + state.incSec * 1000, cap);
    }
    state.clockTurn = state.myColor;
    wasCapture ? sndCapture() : sndMove();
    rebuildMoves();
    if (data.game_over) {
      finishGame(data.result, data.status || 'Game over');
    } else {
      setStatus(data.status || 'Your turn');
      if (parseFen().inCheck) sndCheck();
    }
    evalCache.clear();
    render(); renderClocks();
  } catch (e) {
    setStatus('Error: ' + (e && e.message));
  } finally {
    if (!state.over) setThinking(false);
  }
}

function finishGame(result, status) {
  state.over = true; state.result = result; state.hint = null;
  if (status) setStatus(status);
  addEndMove(result);
  result === '1/2-1/2' ? sndDraw() : (playerWon() ? sndWin() : sndLose());
  streak = playerWon() ? streak + 1 : 0;
  localStorage.setItem('cs_streak', String(streak));
  drawHint();
  render(); renderClocks();
  persistGame();
  openModal();
}
function playerWon() {
  return (state.result === '1-0' && state.myColor === 'w') ||
         (state.result === '0-1' && state.myColor === 'b');
}

function undo() {
  const cps = state.checkpoints;
  if (!cps.length) return;
  if (cps[cps.length - 1].what === 'bot-first') return;
  const cp = cps.pop();
  state.fen = cp.fen || START_FEN;
  state.over = false; state.result = null; state.last = null; state.hint = null;
  state.sanSeq = state.sanSeq.slice(0, Math.max(0, state.sanSeq.length - (cp.sanB ? 2 : 1)));
  state.clockTurn = state.myColor;
  setStatus('Your turn');
  rebuildMoves();
  closeModal();
  persistGame();
  render(); renderClocks();
}

// ------------------------------------------------------------------ PGN ---
function buildPGN() {
  const meWhite = state.myColor === 'w';
  const botName = 'AI (' + state.mode + ' d' + state.depth + ')';
  const d = new Date();
  let pgn = '[Event "Casual"]\n[Site "AI Chess Bot"]\n';
  pgn += `[Date "${d.getFullYear()}.${String(d.getMonth() + 1).padStart(2, '0')}.${String(d.getDate()).padStart(2, '0')}"]\n`;
  pgn += '[Round "-"]\n';
  pgn += `[White "${meWhite ? 'You' : botName}"]\n`;
  pgn += `[Black "${meWhite ? botName : 'You'}"]\n`;
  pgn += `[Result "${state.result || '*'}"]\n\n`;
  const seq = state.sanSeq || [];
  let line = '';
  for (let i = 0; i < seq.length; i++) {
    const no = Math.floor(i / 2) + 1;
    line += (i % 2 === 0 ? `${no}. ${seq[i]} ` : `${seq[i]} `);
    if (line.length > 70) { pgn += line + '\n'; line = ''; }
  }
  pgn += line + (state.result || '*');
  return pgn;
}
async function copyPGN() {
  const txt = buildPGN();
  try {
    await navigator.clipboard.writeText(txt);
    toast('PGN copied');
  } catch (e) {
    const ta = document.createElement('textarea');
    ta.value = txt; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); toast('PGN copied'); } catch (e2) { toast('PGN in console'); }
    ta.remove();
  }
}

// ------------------------------------------------------------ leaderboard --
function refreshLeaderboard() {
  const note = $('lb-note');
  note.textContent = 'Loading…';
  fetch('/api/leaderboard').then(r => r.json()).then(d => {
    if (!d || !d.available) {
      note.textContent = 'Leaderboard unavailable (no DATABASE_URL configured on the server).';
      $('lb-top').querySelector('tbody').innerHTML = '<tr><td colspan="5">—</td></tr>';
      $('lb-recent').querySelector('tbody').innerHTML = '<tr><td colspan="4">—</td></tr>';
      return;
    }
    note.textContent = 'Arcade high scores — no accounts needed. Beat the bot, save your name.';
    const top = $('lb-top').querySelector('tbody');
    top.innerHTML = d.top.length ? '' : '<tr><td colspan="5">No scores yet</td></tr>';
    d.top.forEach((r, i) => {
      top.appendChild(el('tr', '', '<td class="rank">' + (i + 1) + '</td><td>' + esc(r.name) +
        '</td><td><b>' + r.score + '</b></td><td class="res-' + clsResult(r.result) + '">' + r.result +
        '</td><td>' + r.when + '</td>'));
    });
    const rec = $('lb-recent').querySelector('tbody');
    rec.innerHTML = d.recent.length ? '' : '<tr><td colspan="4">No games yet</td></tr>';
    d.recent.forEach(r => {
      rec.appendChild(el('tr', '', '<td>' + esc(r.name) + '</td><td><b>' + r.score + '</b></td>' +
        '<td class="res-' + clsResult(r.result) + '">' + r.result + '</td><td>' + r.when + '</td>'));
    });
  }).catch(() => { note.textContent = 'Leaderboard unavailable right now.'; });
}
function clsResult(result) {
  return result === 'win' ? 'win' : (result === 'loss' ? 'loss' : 'draw');
}
function switchTab(showLB) {
  $('tab-moves').classList.toggle('active', !showLB);
  $('tab-lb').classList.toggle('active', !!showLB);
  $('tab-moves-body').classList.toggle('active', !showLB);
  $('tab-lb-body').classList.toggle('active', !!showLB);
}

// ------------------------------------------------------------------ modal ---
const playerResult = () => playerWon() ? 'win' : (state.result === '1/2-1/2' ? 'draw' : 'loss');

function openModal() {
  const outcome = playerWon() ? 'You win!' : (state.result === '1/2-1/2' ? 'Draw' : 'Bot wins');
  $('m-title').textContent = outcome;
  $('m-result').textContent = `${state.result} — ${state.status}`;
  $('m-save').innerHTML = '';
  fetch('/api/leaderboard').then(r => r.json()).then(d => {
    if (d && d.available) {
      const savedName = localStorage.getItem('cs_name') || '';
      $('m-save').innerHTML =
        '<input id="lb-name" maxlength="24" placeholder="Your name" value="' + esc(savedName) + '">' +
        '<div class="saveq">Save your score (' + _scorePreview() + ')?</div>' +
        '<button onclick="saveScore()">Save score</button>';
    }
  }).catch(() => {});
  $('modal').classList.add('open');
}
function _scorePreview() {
  const base = { win: 100, draw: 30, loss: 0 }[playerResult()] || 0;
  const table = {
    search: { 1: 1, 2: 1.5, 3: 2, 4: 2.5 },
    nn: 1,
    nnleaf: { 1: 1.5, 2: 1.8, 3: 2.2 },
  };
  const m = table[state.mode];
  const mult = (m && typeof m === 'object') ? (m[state.depth] || 1) : (m || 1);
  const pts = Math.round(base * mult) + Math.min(streak * 10, 100);
  return '~' + pts + ' pts';
}
function closeModal() { $('modal').classList.remove('open'); }
async function saveScore() {
  const name = ($('lb-name') && $('lb-name').value.trim()) || 'Player';
  localStorage.setItem('cs_name', name);
  const res = await fetch('/api/score', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, result: playerResult(), mode: state.mode, depth: state.depth, streak }),
  });
  const d = await res.json();
  if (d && d.ok) {
    toast('Saved — rank #' + d.rank + ' (' + d.score + ' pts)');
    $('m-save').innerHTML = '';
    switchTab(true);
    refreshLeaderboard();
  } else {
    toast('Leaderboard unavailable right now');
  }
}

// ------------------------------------------------------------------ toast ---
let toastTimer = null;
function toast(msg) {
  const t = $('toast');
  t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2600);
}

// ------------------------------------------------------------------ promo ---
function pickPromotion(color) {
  return new Promise(resolve => {
    const overlay = $('promo'); const box = $('promo-box');
    box.innerHTML = '';
    ['q', 'r', 'b', 'n'].forEach(p => {
      const btn = document.createElement('button');
      const img = document.createElement('img');
      img.src = `/pieces/${color}${p.toUpperCase()}.svg`; img.alt = p;
      btn.appendChild(img);
      btn.addEventListener('click', () => { overlay.classList.remove('open'); resolve(p); });
      box.appendChild(btn);
    });
    overlay.classList.add('open');
  });
}

// ------------------------------------------------------------- resume/undo --
function dismissResume() { $('resume-bar').style.display = 'none'; }
function maybeOfferResume() {
  try {
    const s = JSON.parse(localStorage.getItem('cs_game') || 'null');
    if (s && s.fen && s.sanSeq && s.sanSeq.length) {
      $('resume-bar').style.display = 'flex';
      return true;
    }
  } catch (e) {}
  return false;
}
function resumeGame() {
  try {
    const s = JSON.parse(localStorage.getItem('cs_game') || 'null');
    if (!s || !s.fen) return;
    Object.assign(state, s);
    state.hint = null; state.thinking = false; state.selected = null; state.targets = [];
    dismissResume();
    syncControlsFromState();
    rebuildMoves();
    if (state.over) addEndMove(state.result);
    render(); renderClocks();
    if (state.over) openModal(); else setStatus(state.status || 'Your turn');
  } catch (e) { console.error(e); }
}
function syncControlsFromState() {
  $('side').value = state.myColor;
  $('mode').value = state.mode;
  document.querySelectorAll('#diff button').forEach(b =>
    b.classList.toggle('active', +b.dataset.d === state.depth));
  const tv = state.timeMin ? `${state.timeMin} ${state.incSec}` : '0';
  $('time').value = tv;
}

// -------------------------------------------------------------------- ui ----
function bindUI() {
  $('btn-theme').addEventListener('click', () => {
    settings.theme = settings.theme === 'dark' ? 'light' : 'dark';
    applySettings(); saveSettings();
  });
  $('btn-sound').addEventListener('click', () => {
    settings.sound = !settings.sound; applySettings(); saveSettings();
    if (settings.sound) sndMove();
  });
  $('btn-eval').addEventListener('click', () => {
    settings.eval = !settings.eval; applySettings(); saveSettings(); renderEval();
  });
  $('skin').addEventListener('change', () => { settings.skin = $('skin').value; applySettings(); saveSettings(); });
  $('btn-new').addEventListener('click', () => newGame());
  $('btn-undo').addEventListener('click', () => undo());
  $('btn-hint').addEventListener('click', () => requestHint());
  $('btn-pgn').addEventListener('click', () => copyPGN());
  document.querySelectorAll('#diff button').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#diff button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      state.depth = +b.dataset.d;
    });
  });
  $('tab-moves').addEventListener('click', () => switchTab(false));
  $('tab-lb').addEventListener('click', () => { switchTab(true); refreshLeaderboard(); });
  window.addEventListener('keydown', e => {
    if (e.key.toLowerCase() === 'h' && !state.over && !state.thinking) requestHint();
  });
}

applySettings();
bindUI();
render();
if (!maybeOfferResume()) newGame();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"\nInteractive chessboard:  http://127.0.0.1:{port}\n")
    print(f"MAX_CONCURRENT={MAX_CONCURRENT} | MAX_DEPTH={MAX_DEPTH} | "
          "production servers should use 'gunicorn chess_web:app' instead")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)