"""Interactive chess Web app (Flask) for the AI Chess Bot.

Reuses the engine from chess_bot.py (single source of truth): the bot mode,
depth and model are shared with the CLI and the notebooks.

Run:
    C:/Users/Admin/anaconda3/python.exe chess_web.py
Then open http://127.0.0.1:5000 in your browser. Drag pieces to move;
click a piece to see its legal moves, then click a destination.
"""

import os
import sys
import threading

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Limit torch/OpenMP to a few threads early (matters on shared 0.1-CPU hosts).
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("TORCH_THREADS", "2"))

import chess
import chess.svg
from flask import Flask, jsonify, request, render_template_string

import chess_bot

START_FEN = chess.STARTING_FEN
NS = chess_bot.load_namespace()
try:
    NS["torch"].set_num_threads(max(1, int(os.environ.get("TORCH_THREADS", "2"))))
except Exception:
    pass
MODEL = None

# Production config (Hugging Face Spaces sets PORT; override anything via env)
PORT = int(os.environ.get("PORT", "5000"))
MAX_CONCURRENT = max(1, int(os.environ.get("MAX_CONCURRENT", "4")))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "4"))
MODES = {"search", "nn", "nnleaf"}
AI_SEM = threading.Semaphore(MAX_CONCURRENT)


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
    try:
        depth = max(1, min(int(params.get("depth", 2)), MAX_DEPTH))
    except (TypeError, ValueError):
        depth = 2
    side = params.get("side", "w")
    if side not in ("w", "b"):
        side = "w"
    return mode, depth, side


def _move_pair(before_fen, uci):
    """Find the legal move matching from/to (auto-queen promotions)."""
    if not uci or len(uci) not in (4, 5) or not uci.isalnum():
        return None, None
    try:
        b = chess.Board(fen=before_fen)
        m = chess.Move.from_uci(uci)
    except ValueError:
        return None, None
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

    human_color = side
    bot_is_white = human_color != "w"

    if board.is_game_over():
        resp["game_over"] = True
        resp["result"] = board.result()
        resp["status"] = _status(board)
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
    resp["status"] = _turn_status(board) if not board.is_game_over() else _status(board)
    if board.is_game_over():
        resp["game_over"] = True
        resp["result"] = board.result()
    return jsonify(resp)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI Chess Bot</title>
<link rel="icon" href="data:,">
<style>
  :root { --light:#f0d9b5; --dark:#b58863; --sel:#ffff66; --legal:#7dff6b; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         background:#312e2b; color:#eee; display:flex; justify-content:center; }
  .wrap { max-width:1000px; width:100%; padding:16px; display:flex;
          flex-wrap:wrap; gap:20px; align-items:flex-start; }
  h1 { font-size:22px; margin:0 0 12px; }
  .board-box { flex:1 1 560px; max-width:640px; position:relative; }
  #board { display:grid; grid-template-columns:repeat(8,1fr);
           grid-template-rows:repeat(8,1fr); width:100%; aspect-ratio:1/1;
           border:4px solid #1f1d1c; border-radius:4px;
           box-shadow:0 8px 24px rgba(0,0,0,.5); user-select:none;
           touch-action:none; }
  .cell { position:relative; display:flex; align-items:center; justify-content:center; }
  .cell.light { background:var(--light); }
  .cell.dark  { background:var(--dark); }
  .cell.has-piece { cursor:grab; }
  .cell.legal::after { content:""; position:absolute; width:30%; height:30%;
        border-radius:50%; background:rgba(0,0,0,.18); pointer-events:none; }
  .cell.legal.has-piece::after { width:100%; height:100%;
        border:3px solid rgba(0,0,0,.3); border-radius:50%; background:transparent; }
  .cell.selected { box-shadow:inset 0 0 0 4px var(--sel); }
  .cell.last-from, .cell.last-to { box-shadow:inset 0 0 0 4px #f7f769; }
  .cell img { width:92%; height:92%; pointer-events:none; object-fit:contain; }
  .coord { position:absolute; font-size:10px; opacity:.55; }
  .coord.file { bottom:1px; right:2px; }
  .coord.rank { top:1px; left:2px; }
  #thinking { position:absolute; inset:0; display:none; align-items:center;
          justify-content:center; background:rgba(0,0,0,.45); z-index:5;
          border-radius:4px; font-size:20px; font-weight:600; letter-spacing:.5px; }
  .panel { flex:0 1 320px; background:#3a3633; border-radius:8px;
           padding:14px 16px; box-shadow:0 4px 16px rgba(0,0,0,.4); }
  .panel h2 { font-size:15px; margin:0 0 10px; color:#cbb; }
  .row { display:flex; align-items:center; gap:8px; margin:8px 0; flex-wrap:wrap; }
  label { font-size:13px; opacity:.85; white-space:nowrap; }
  select, button { font:inherit; padding:5px 8px; border-radius:6px;
          border:1px solid #555; background:#24211f; color:#eee; }
  button { cursor:pointer; }
  button:hover { background:#35312e; }
  #status { min-height:22px; padding:8px 0; font-weight:600; color:#ffe27a; }
  #moves { background:#24211f; border:1px solid #444; border-radius:6px;
           padding:8px; height:260px; overflow:auto; font-family:ui-monospace,
           Consolas,monospace; font-size:13px; line-height:1.6; }
  #moves div { display:grid; grid-template-columns:28px 1fr 1fr; }
  .hint { font-size:12px; opacity:.7; margin-top:8px; line-height:1.5; }
  .shake { animation:shake .3s; }
  @keyframes shake { 25%{transform:translateX(-4px)} 75%{transform:translateX(4px)} }
</style>
</head>
<body>
<div class="wrap">
  <div class="board-box">
    <div id="thinking">AI is thinking...</div>
    <div id="board"></div>
  </div>

  <div class="panel">
    <h1>AI Chess Bot</h1>
    <div class="row">
      <label>You play</label>
      <select id="side"><option value="w">White</option><option value="b">Black</option></select>
      <label>Bot</label>
      <select id="mode">
        <option value="search">Search (minimax)</option>
        <option value="nn">Neural (policy)</option>
        <option value="nnleaf">NN-leaf (slow)</option>
      </select>
      <select id="depth">
        <option value="1">depth 1</option>
        <option value="2" selected>depth 2</option>
        <option value="3">depth 3</option>
      </select>
      <button onclick="newGame()">New game</button>
    </div>
    <div id="status">Your turn</div>
    <div id="moves"></div>
    <div class="hint">Drag pieces to move. Or click a piece, then click a
      highlighted square to move there. Pawns auto-promote to queen.
      Stockfish is used for rating in the CLI; the bot here is the neural
      search engine from chess_main.ipynb / chess_bot.py.</div>
  </div>
</div>

<script>
const files = ['a','b','c','d','e','f','g','h'];
let state = {
  fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  myColor: 'w', mode: 'search', depth: 2,
  selected: null, targets: [], thinking: false, over: false, moveNo: 1,
};

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
function pieceAt(sq) {
  const idx = fenIndex(sq);
  const place = fenBoard();
  let n = 0;
  for (const ch of place) {
    if (ch === '/') continue;
    if (/\d/.test(ch)) { n += +ch; continue; }
    if (n === idx) return ch;
    n++;
  }
  return null;
}
function fenIndex(sq) {
  const f = sq.charCodeAt(0) - 97;
  const r = 8 - (+sq[1]);
  return r * 8 + f;
}
function sqFromPos(r, c) {
  // visual row/col -> square, respecting board orientation
  const layout = cellSquares();
  return layout[r * 8 + c];
}

function render() {
  const board = document.getElementById('board');
  board.innerHTML = '';
  const place = fenBoard();
  const layout = cellSquares();
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const sq = layout[r * 8 + c];
      const cell = document.createElement('div');
      cell.className = 'cell ' + ((r + c) % 2 === 0 ? 'light' : 'dark');
      cell.id = 'sq-' + sq;
      cell.dataset.square = sq;

      const pc = pieceAt(sq);
      if (pc) {
        const code = (pc === pc.toUpperCase() ? 'w' : 'b') + pc.toUpperCase();
        const img = document.createElement('img');
        img.src = `/pieces/${code}.svg`;
        img.alt = code;
        cell.appendChild(img);
        cell.classList.add('has-piece');
      }
      if (r === 7) cell.appendChild(coord('file', sq[0]));
      if (c === 0) cell.appendChild(coord('rank', sq[1]));

      cell.addEventListener('click', () => clickCell(sq));
      board.appendChild(cell);
    }
  }
  clearSel();
  highlightLast();
}

let drag = null;

function cellPx() {
  const c = document.querySelector('.cell');
  return c ? c.getBoundingClientRect().width : 64;
}

// --- Pointer-based drag & drop (works like lichess / chess.com) ---
board.addEventListener('mousedown', e => {
  if (state.thinking || state.over || e.button !== 0) return;
  const cell = e.target.closest('.cell');
  if (!cell) return;
  const sq = cell.dataset.square;
  const pc = pieceAt(sq);
  if (!pc || !isUserPiece(pc)) return;   // let click-select handle it
  e.preventDefault();
  const img = cell.querySelector('img');
  if (!img) return;
  const clone = img.cloneNode();
  const size = cellPx();
  clone.style.cssText = 'position:fixed;z-index:500;pointer-events:none;' +
    `width:${size}px;height:${size}px;left:${e.clientX - size / 2}px;` +
    `top:${e.clientY - size / 2}px;`;
  document.body.appendChild(clone);
  drag = { from: sq, clone, size };
});

window.addEventListener('mousemove', e => {
  if (!drag) return;
  const s = drag.size;
  drag.clone.style.left = (e.clientX - s / 2) + 'px';
  drag.clone.style.top = (e.clientY - s / 2) + 'px';
});

window.addEventListener('mouseup', e => {
  if (!drag) return;
  const { from, clone } = drag;
  drag = null;
  clone.remove();
  const el = document.elementFromPoint(e.clientX, e.clientY);
  const cell = el && el.closest('.cell');
  const to = cell && cell.dataset.square;
  if (to && to !== from) sendMove(from, to);
});

function coord(cls, text) {
  const s = document.createElement('span');
  s.className = 'coord ' + cls;
  s.textContent = text;
  return s;
}

function isUserPiece(pc) {
  const color = pc === pc.toUpperCase() ? 'w' : 'b';
  return color === state.myColor && fenTurn() === state.myColor;
}

function clearHighlights() {
  document.querySelectorAll('.cell.selected,.cell.legal').forEach(c =>
    c.classList.remove('selected', 'legal'));
}

function clearSel() {
  state.selected = null; state.targets = [];
  clearHighlights();
}

function highlightLast() {
  document.querySelectorAll('.last-from,.last-to').forEach(c =>
    c.classList.remove('last-from', 'last-to'));
  if (!state.last) return;
  const f = document.getElementById('sq-' + state.last.from);
  const t = document.getElementById('sq-' + state.last.to);
  if (f) f.classList.add('last-from');
  if (t) t.classList.add('last-to');
}

async function newGame() {
  state.myColor = document.getElementById('side').value;
  state.mode = document.getElementById('mode').value;
  state.depth = +document.getElementById('depth').value;
  state.thinking = true; state.over = false; state.last = null;
  state.moveNo = 1;
  document.getElementById('moves').innerHTML = '';
  document.getElementById('status').textContent = 'Starting game...';
  showThinking();
  try {
    const res = await fetch(`/api/new_game?side=${state.myColor}&mode=${state.mode}&depth=${state.depth}`);
    const data = await res.json();
    state.fen = data.fen;
    if (data.bot_move) {
      state.last = { from: data.bot_move.slice(0, 2), to: data.bot_move.slice(2, 4) };
      addMove(1, '', data.bot_san || data.bot_move);
    }
    document.getElementById('status').textContent = data.status || 'Your turn';
  } catch (e) {
    document.getElementById('status').textContent = 'Error: ' + e.message;
  }
  hideThinking();
  state.thinking = false;
  render();
}

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
  const sel = document.getElementById('sq-' + sq);
  if (sel) sel.classList.add('selected');
  state.targets.forEach(t => {
    const cell = document.getElementById('sq-' + t);
    if (cell) cell.classList.add('legal');
  });
}

async function sendMove(from, to) {
  if (state.thinking || state.over) return;
  const pc = pieceAt(from);
  let promotion = null;
  if (pc && pc.toLowerCase() === 'p') {
    const toRank = to[1];
    if ((pc === 'P' && toRank === '8') || (pc === 'p' && toRank === '1')) promotion = 'q';
  }
  const body = {
    fen: state.fen, uci: from + to, mode: state.mode, depth: state.depth,
    side: state.myColor, promotion,
  };
  setThinking(true);
  try {
    const res = await fetch('/api/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!data.ok) {
      clearSel();
      const status = document.getElementById('status');
      status.textContent = data.reason || 'Illegal move';
      status.classList.add('shake');
      setTimeout(() => status.classList.remove('shake'), 300);
      return;
    }
    state.fen = data.fen;
    state.last = { from, to };
    state.over = data.game_over;
    addMove(state.moveNo, data.san_human, data.bot_san || '');
    if (data.bot_move) {
      state.last = { from: data.bot_move.slice(0, 2), to: data.bot_move.slice(2, 4) };
      if (data.bot_san) state.moveNo++;
    } else if (state.over) {
      state.moveNo++;
    }
    document.getElementById('status').textContent =
      data.status || (state.over ? 'Game over' : 'Your turn');
  } catch (e) {
    document.getElementById('status').textContent = 'Error: ' + e.message;
  } finally {
    setThinking(false);
  }
  render();
}

function addMove(no, white, black) {
  const box = document.getElementById('moves');
  const row = document.createElement('div');
  const a = document.createElement('span'); a.textContent = no + '.';
  const b = document.createElement('span'); b.textContent = white;
  const c = document.createElement('span'); c.textContent = black;
  row.append(a, b, c);
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
}

function showThinking() { document.getElementById('thinking').style.display = 'flex'; }
function hideThinking() { document.getElementById('thinking').style.display = 'none'; }
function setThinking(t) {
  state.thinking = t;
  if (t) showThinking(); else hideThinking();
}

render();
newGame();
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