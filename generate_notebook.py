"""Generates the improved chess_main.ipynb notebook."""
import json

cells = []


def md(source: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(source)})


def code(source: str):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": _lines(source),
        "execution_count": None,
        "outputs": [],
    })


def _lines(source: str):
    lines = source.split("\n")
    return [line + "\n" if i < len(lines) - 1 else line for i, line in enumerate(lines)]


# --------------------------------------------------------------------------- #
# Title
# --------------------------------------------------------------------------- #
md("""# AI Chess Bot - Improved

A neural-network chess engine built with **PyTorch** and **python-chess**.
The bot evaluates board positions from FEN strings, picks moves, plays against
humans, and can be rated by playing Stockfish at various Elo levels.

## How to use
1. **Section 2** - set your paths (training CSV + Stockfish binary)
2. **Sections 3-4** - encode the data and optionally train the model
3. **Section 5** - play against the bot
4. **Section 6** - play or rate against Stockfish

> If you do not want to retrain, skip to **Section 4.4** to load a saved model.
""")

# --------------------------------------------------------------------------- #
# Section 1 - Setup & Configuration
# --------------------------------------------------------------------------- #
md("""## 1. Setup & Configuration

Run this cell to install dependencies (first run only).
""")

code("""# Install dependencies (uncomment on a fresh environment)
# !pip install -r requirements.txt
""")

code("""import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import chess
import chess.engine

print(f"PyTorch {torch.__version__} | python-chess {chess.__version__}")

# Set device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -----------------------------------------------------------------------------
# CONFIGURATION - update these paths for your machine
# -----------------------------------------------------------------------------
PROJECT_DIR = os.getcwd()  # Jupyter starts in the notebook's directory

# Path to the Kaggle "Train an AI to play Chess" training data
TRAIN_CSV_PATH = os.path.join(PROJECT_DIR, "train-an-ai-to-play-chess", "train.csv")

# Path to the Stockfish executable
# (auto-detected fallback: common names / PATH lookups).
STOCKFISH_PATH = os.path.join(PROJECT_DIR, "stockfish", "stockfish.exe")

MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "chess_model.pt")

# Black scores are in ~centipawn-scale; rescale so value loss is well-behaved.
VALUE_SCALE = 512.0

print(f"Train CSV: {TRAIN_CSV_PATH}")
print(f"Model will be saved to: {MODEL_SAVE_PATH}")
""")

# --------------------------------------------------------------------------- #
# Section 2 - Board Encoding
# --------------------------------------------------------------------------- #
md("""## 2. Board Encoding

Each board is encoded as an **8x8x19** tensor:

| Channels | Meaning |
|---|---|
| 0-12 | one-hot piece encoding (r n b q k p R N B Q K P .) |
| 13 | side to move (1 = white, 0 = black) |
| 14 | white kingside castling right |
| 15 | white queenside castling right |
| 16 | black kingside castling right |
| 17 | black queenside castling right |
| 18 | en passant target square available |

Adding turn / castling / en-passant information (beyond plain piece layout)
gives the model critical state that the original 13-channel encoding ignored.
""")

code("""PIECE_CHARS = "rnbqkpRNBQKP."


def encode_piece(piece: str) -> np.ndarray:
    \"\"\"One-hot encode a single piece character into a 13-dim vector.\"\"\"
    arr = np.zeros(len(PIECE_CHARS))
    if piece in PIECE_CHARS:
        arr[PIECE_CHARS.index(piece)] = 1.0
    return arr


def encode_board(board: chess.Board) -> np.ndarray:
    \"\"\"Encode a board into an 8x8x19 tensor (float32).\"\"\"
    rows = []
    for row in str(board).split('\\n'):
        rows.append([encode_piece(p) for p in row.replace(' ', '')])
    tensor = np.array(rows, dtype=np.float32)  # 8x8x13

    # Auxiliary channels (spread across all 64 squares for simplicity)
    aux = np.zeros((8, 8, 6), dtype=np.float32)
    aux[:, :, 0] = 1.0 if board.turn == chess.WHITE else 0.0
    aux[:, :, 1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    aux[:, :, 2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    aux[:, :, 3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    aux[:, :, 4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    aux[:, :, 5] = 1.0 if board.ep_square is not None else 0.0

    return np.concatenate([tensor, aux], axis=-1)  # 8x8x19


def encode_fen(fen: str) -> np.ndarray:
    \"\"\"Encode a FEN string into an 8x8x19 tensor.\"\"\"
    return encode_board(chess.Board(fen=fen))
""")

# --------------------------------------------------------------------------- #
# Section 3 - Data Loading & Preprocessing
# --------------------------------------------------------------------------- #
md("""## 3. Data Loading & Preprocessing

Loads the training CSV, encodes every board, and creates a *shuffled*
train/validation/test split (the original code just took the last 1000 rows,
which leaked ordering bias into the validation set).
""")

code("""import time

from sklearn.model_selection import train_test_split


def load_dataset(csv_path: str, sample_size: int = 20000, seed: int = 42):
    \"\"\"Load, shuffle, and encode the dataset.

    Args:
        csv_path: path to train.csv (columns: board, black_score, best_move, id)
        sample_size: number of rows to use (None = use everything)
        seed: random seed for reproducibility

    Returns:
        X, y, move_from, move_to where X is (N, 8, 8, 19), y is black_score,
        and move_from / move_to are the source/destination squares (0..63) of
        best_move, used as policy targets.
    \"\"\"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Training CSV not found at: {csv_path}\\n"
            "Please update TRAIN_CSV_PATH in Section 1."
        )

    df = pd.read_csv(csv_path)

    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=seed)

    print(f"Encoding {len(df)} positions...")
    start = time.time()

    X = np.stack([encode_fen(fen) for fen in df['board']]).astype(np.float32)
    y = df['black_score'].astype(np.float32).values

    move_from = np.zeros(len(df), dtype=np.int64)
    move_to = np.zeros(len(df), dtype=np.int64)
    for i, move in enumerate(df['best_move']):
        parsed = chess.Move.from_uci(str(move).strip())
        move_from[i] = parsed.from_square
        move_to[i] = parsed.to_square

    print(f"Encoded {len(X)} positions in {time.time() - start:.1f}s")
    print(f"X shape: {X.shape}, y range: [{y.min():.1f}, {y.max():.1f}]")
    return X, y, move_from, move_to


# Encode the dataset (use sample_size=None to use all ~58k rows)
X, y, move_from, move_to = load_dataset(TRAIN_CSV_PATH, sample_size=20000)

# Proper shuffled split: 80% train / 10% val / 10% test
idx = np.arange(len(y))
idx_train, idx_temp = train_test_split(
    idx, test_size=0.2, random_state=42, shuffle=True
)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.5, random_state=42, shuffle=True
)
X_train, y_train, mf_train, mt_train = (X[idx_train], y[idx_train],
                                        move_from[idx_train], move_to[idx_train])
X_val, y_val, mf_val, mt_val = (X[idx_val], y[idx_val],
                                move_from[idx_val], move_to[idx_val])
X_test, y_test, mf_test, mt_test = (X[idx_test], y[idx_test],
                                    move_from[idx_test], move_to[idx_test])

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
""")

# --------------------------------------------------------------------------- #
# Section 4 - Model
# --------------------------------------------------------------------------- #
md("""## 4. Model

A deeper network than the original single-hidden-layer model, with **BatchNorm**
and **Dropout** for faster, more stable training.

```
Flatten(8x8x19)
  → Dense(512, ReLU) → BatchNorm → Dropout(0.3)
  → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
  → Dense(128, ReLU)
  → Dense(1)          <- predicted black_score
```

The output is the estimated **score from Black's perspective** (matching the
training labels), so White wants to *maximize* the score and Black wants to
*minimize* it.
""")

code("""class ChessEvalModel(nn.Module):
    \"\"\"Dual-head model: value (score) + policy (best move from/to squares).

    The value head outputs black_pov_score / VALUE_SCALE. The policy heads
    output logits over source (64) and destination (64) squares; at move time
    legal moves are scored as p_from[orig] + p_to[dest].
    \"\"\"
    def __init__(self, input_channels: int = 19):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * input_channels, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)
        self.from_head = nn.Linear(128, 64)
        self.to_head = nn.Linear(128, 64)

    def forward(self, x):
        features = self.net(x)
        return {
            'value': self.value_head(features).squeeze(-1),
            'from': self.from_head(features),
            'to': self.to_head(features),
        }


# -----------------------------------------------------------------------------
# Build model + optimizer
# -----------------------------------------------------------------------------
model = ChessEvalModel(input_channels=19).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

policy_from = nn.LogSoftmax(dim=1)
policy_to = nn.LogSoftmax(dim=1)
""")

# --------------------------------------------------------------------------- #
# Section 4.2 - Training
# --------------------------------------------------------------------------- #
md("""### 4.2 Training

Trains with a learning-rate scheduler and early stopping on the validation loss.
""")

code("""from torch.utils.data import DataLoader, TensorDataset


def make_loader(X, y, mf, mt, batch_size=128, shuffle=True):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(mf, dtype=torch.long),
        torch.tensor(mt, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


train_loader = make_loader(X_train, y_train, mf_train, mt_train,
                           batch_size=128, shuffle=True)
val_loader = make_loader(X_val, y_val, mf_val, mt_val,
                         batch_size=128, shuffle=False)


def train_model(model, train_loader, val_loader, epochs=60,
                lr=1e-3, patience_early_stop=8, patience_lr=4, verbose=True,
                value_coef=5.0, policy_labels=('from', 'to')):
    \"\"\"Train the dual value/policy model.

    Loss = value_coef * MSE(scaled value) + CE(policy from) + CE(policy to).
    Value targets are scaled by VALUE_SCALE for stable training.
    \"\"\"
    optimizer = optim.Adam(model.parameters(), lr=lr)
    value_loss_fn = nn.MSELoss()
    schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience_lr, min_lr=1e-6
    )

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, train_batches = 0.0, 0
        for xb, yb, fb, tb in train_loader:
            xb, yb, fb, tb = (xb.to(DEVICE), yb.to(DEVICE),
                              fb.to(DEVICE), tb.to(DEVICE))
            optimizer.zero_grad()
            pred = model(xb)
            loss = (value_coef * value_loss_fn(pred['value'], yb / VALUE_SCALE)
                    + nn.functional.cross_entropy(pred['from'], fb)
                    + nn.functional.cross_entropy(pred['to'], tb))
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1
        train_loss = train_loss_sum / train_batches

        # Validation
        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for xb, yb, fb, tb in val_loader:
                xb, yb, fb, tb = (xb.to(DEVICE), yb.to(DEVICE),
                                  fb.to(DEVICE), tb.to(DEVICE))
                pred = model(xb)
                loss = (value_coef * value_loss_fn(pred['value'], yb / VALUE_SCALE)
                        + nn.functional.cross_entropy(pred['from'], fb)
                        + nn.functional.cross_entropy(pred['to'], tb))
                val_loss_sum += loss.item()
                val_batches += 1
        val_loss = val_loss_sum / val_batches

        schedule.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} "
                  f"| train loss: {train_loss:8.4f} "
                  f"| val loss: {val_loss:8.4f} | lr: {current_lr:.2e}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_early_stop:
                print(f"Early stopping at epoch {epoch} (best val loss {best_val:.4f})")
                break

    return history


history = train_model(model, train_loader, val_loader, epochs=60)
""")

# --------------------------------------------------------------------------- #
# Section 4.3 - Training curves
# --------------------------------------------------------------------------- #
md("""### 4.3 Training curves""")

code("""plt.style.use('ggplot')
plt.figure(figsize=(8, 4))
plt.plot(history['train_loss'], label='train loss')
plt.plot(history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('MSE (black_score)')
plt.title('Loss during training')
plt.legend()
plt.grid(True)
plt.show()
""")

# --------------------------------------------------------------------------- #
# Section 4.4 - Save / load model
# --------------------------------------------------------------------------- #
md("""### 4.4 Save / load model

Training automatically saves the best weights to `MODEL_SAVE_PATH`.
On later runs you can skip training entirely and load them directly.
""")

code("""def save_model(model, path=MODEL_SAVE_PATH):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model_weights(path=MODEL_SAVE_PATH):
    model = ChessEvalModel(input_channels=19).to(DEVICE)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.eval()
        print(f"Model loaded from {path}")
    else:
        raise FileNotFoundError(f"No saved model found at {path}")
    return model


# Uncomment to load a previously trained model instead of retraining:
# model = load_model_weights()
""")

# --------------------------------------------------------------------------- #
# Section 5 - Search & Move Selection
# --------------------------------------------------------------------------- #
md("""## 5. Search & Move Selection

### 5.1 Classical minimax (material + piece-square tables)

A minimax search with alpha-beta pruning, move ordering (captures first,
MVV-LVA), **piece-square tables** for positional awareness, and a
**quiescence search** (search captures at leaf nodes) to avoid the horizon effect.
""")

code("""# -----------------------------------------------------------------------------
# Piece-square tables (standard chess PSTs, White POV)
# -----------------------------------------------------------------------------
PIECE_VALUES = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 0}

PAWN_PST = np.array([
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [ 50,  50,  50,  50,  50,  50,  50,  50],
    [ 10,  10,  20,  30,  30,  20,  10,  10],
    [  5,   5,  10,  25,  25,  10,   5,   5],
    [  0,   0,   0,  20,  20,   0,   0,   0],
    [  5,  -5, -10,   0,   0, -10,  -5,   5],
    [  5,  10,  10, -20, -20,  10,  10,   5],
    [  0,   0,   0,   0,   0,   0,   0,   0],
])

KNIGHT_PST = np.array([
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20,   0,   0,   0,   0, -20, -40],
    [-30,   0,  10,  15,  15,  10,   0, -30],
    [-30,   5,  15,  20,  20,  15,   5, -30],
    [-30,   0,  15,  20,  20,  15,   0, -30],
    [-30,   5,  10,  15,  15,  10,   5, -30],
    [-40, -20,   0,   5,   5,   0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
])

BISHOP_PST = np.array([
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,  10,  10,  10,  10,   0, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
])

ROOK_PST = np.array([
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  5,  10,  10,  10,  10,  10,  10,   5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [  0,   0,   0,   5,   5,   0,   0,   0],
])

QUEEN_PST = np.array([
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,   5,   5,   5,   0, -10],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [  0,   0,   5,   5,   5,   5,   0,  -5],
    [-10,   5,   5,   5,   5,   5,   0, -10],
    [-10,   0,   5,   0,   0,   0,   0, -10],
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
])

KING_PST = np.array([
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [ 20,  20,   0,   0,   0,   0,  20,  20],
    [ 20,  30,  10,   0,   0,  10,  30,  20],
])

PST = {
    chess.PAWN: PAWN_PST, chess.KNIGHT: KNIGHT_PST, chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST, chess.QUEEN: QUEEN_PST, chess.KING: KING_PST,
}


def evaluate_board(board, nn_model=None):
    \"\"\"Material + PST evaluation, White POV (cp).
    If nn_model is given, evaluates with the neural net instead.\"\"\"
    if board.is_checkmate():
        return -99999 if board.turn == chess.WHITE else 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if nn_model is not None:
        return -float(evaluate_positions([board], nn_model)[0])

    score = 0
    for ptype, value in PIECE_VALUES.items():
        for sq in board.pieces(ptype, chess.WHITE):
            rank, file = sq // 8, sq % 8
            score += value + PST[ptype][7 - rank][file]
        for sq in board.pieces(ptype, chess.BLACK):
            rank, file = sq // 8, sq % 8
            score -= value + PST[ptype][rank][file]
    return score


import random
import time

MATE = 99999
MATE_ENTERING = MATE - 256               # scores of this magnitude (or more) are mates
_MAX_QDEPTH = 8                          # quiescence depth cap (capture chains)
TT_SIZE = 1 << 18                        # bounded transposition table (memory-safe)
FLAG_EXACT, FLAG_UPPER, FLAG_LOWER = 1, 0, 2
_TT = {}


class _TimeUp(Exception):
    \"\"\"Internal: abort current depth when the time budget is exhausted.\"\"\"
    pass


# Deterministic Zobrist keys (64-bit) for the transposition table.
_rng = random.Random(0x5EED)
_ZOBRIST = [[_rng.getrandbits(64) for _ in range(64)] for _ in range(12)]
_ZOB_TURN = _rng.getrandbits(64)
_ZOB_CASTLE = [_rng.getrandbits(64) for _ in range(16)]
_ZOB_EP = [_rng.getrandbits(64) for _ in range(64)]


def zobrist_key(board):
    key = _ZOB_TURN if board.turn == chess.WHITE else 0
    cr = board.castling_rights
    castle_idx = ((cr >> 56) & 1) * 8 + ((cr >> 63) & 1) * 4 \
        + (cr & 1) * 2 + ((cr >> 7) & 1)
    key ^= _ZOB_CASTLE[castle_idx]
    ep = board.ep_square
    if ep is not None:
        key ^= _ZOB_EP[ep]
    for sq, piece in board.piece_map().items():
        color = 1 if piece.color == chess.WHITE else 0
        key ^= _ZOBRIST[(piece.piece_type - 1) * 2 + color][sq]
    return key


def order_moves(board, tt_move=None, killers=None, history=None):
    \"\"\"Order moves to improve alpha-beta pruning: TT best first, then killer
    moves, then MVV-LVA captures, then the history heuristic for quiet moves.\"\"\"
    if killers is None:
        killers = ()
    hist = history if history is not None else {}

    def move_score(move):
        if move == tt_move:
            return float('inf')
        if move in killers:
            return 1e8
        s = 0
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square)
            attacker = board.piece_type_at(move.from_square)
            if victim:
                s += 10000 + 10 * PIECE_VALUES.get(victim, 0) - PIECE_VALUES.get(attacker, 0)
        else:
            s += hist.get((move.from_square, move.to_square), 0)
        if board.gives_check(move):
            s += 2000
        if move.promotion:
            s += 1500
        return s

    return sorted(board.legal_moves, key=move_score, reverse=True)


def _time_up(ctx):
    deadline = ctx["deadline"]
    if deadline is None:
        return False
    ctx["nodes"] += 1
    if ctx["nodes"] & 255 == 0 and time.monotonic() > deadline:
        ctx["hit"] = True
        return True
    return False


def _leaf_eval(board, nn_model=None, ply=0):
    \"\"\"PST evaluation with ply-aware mate scores so the TT cannot mix mate
    scores from different plies (mate-distance correctness).\"\"\"
    v = evaluate_board(board, nn_model)
    if nn_model is None:
        if v >= MATE_ENTERING:
            return MATE - ply
        if v <= -MATE_ENTERING:
            return -MATE + ply
    return v


def _adj_store(score, ply):
    if score >= MATE_ENTERING:
        return score + ply
    if score <= -MATE_ENTERING:
        return score - ply
    return score


def _adj_read(score, ply):
    if score >= MATE_ENTERING:
        return score - ply
    if score <= -MATE_ENTERING:
        return score + ply
    return score


def _store_tt(key, depth, flag, score, move):
    _TT[key] = (depth, flag, score, move)
    if len(_TT) > TT_SIZE:
        _TT.clear()


def quiescence(board, alpha, beta, is_maximizing, ctx=None, ply=0, qdepth=_MAX_QDEPTH):
    \"\"\"Search only captures at leaf nodes to reduce the horizon effect.
    Depth is capped (`qdepth`) and it never recurses into quiet moves, so a
    capture chain stays bounded and cannot stall iterative deepening.\"\"\"
    if ctx is None:
        ctx = {"deadline": None, "nodes": 0, "hit": False}
    if _time_up(ctx):
        raise _TimeUp()
    if qdepth <= 0 or board.is_game_over():
        return _leaf_eval(board, ply=ply)

    stand_pat = _leaf_eval(board, ply=ply)
    if is_maximizing:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)

    for move in order_moves(board):
        if not board.is_capture(move):
            continue
        board.push(move)
        try:
            score = quiescence(board, alpha, beta, not is_maximizing, ctx,
                               ply + 1, qdepth - 1)
        finally:
            board.pop()
        if is_maximizing:
            alpha = max(alpha, score)
            if alpha >= beta:
                return beta
        else:
            beta = min(beta, score)
            if alpha >= beta:
                return alpha

    return alpha if is_maximizing else beta


def minimax(board, depth, alpha, beta, is_maximizing, nn_model=None,
            ctx=None, ply=0, use_tt=True):
    \"\"\"Alpha-beta minimax with a transposition table, killer moves and a
    history heuristic. With nn_model, leaf positions are scored by the neural
    net in one batched forward pass (no quiescence, and the TT is disabled
    so PST and NN scores never mix). Without it, quiescence search is used
    with PST evaluation.\"\"\"
    if _time_up(ctx):
        raise _TimeUp()
    if board.is_game_over():
        return _leaf_eval(board, nn_model, ply)

    use_tt = use_tt and nn_model is None
    key = zobrist_key(board) if use_tt else None
    tt_move = None
    if key is not None:
        entry = _TT.get(key)
        if entry is not None:
            tt_move = entry[3]
            if entry[0] >= depth:
                cached = _adj_read(entry[2], ply)
                if entry[1] == FLAG_EXACT:
                    return cached
                if entry[1] == FLAG_LOWER and cached >= beta:
                    return cached
                if entry[1] == FLAG_UPPER and cached <= alpha:
                    return cached

    if depth == 0:
        if nn_model is None:
            return quiescence(board, alpha, beta, is_maximizing, ctx, ply)
        child_boards = []
        for move in order_moves(board):
            if _time_up(ctx):
                raise _TimeUp()
            board.push(move)
            child_boards.append(board.copy())
            board.pop()
        scores = -evaluate_positions(child_boards, nn_model)  # White POV
        if is_maximizing:
            return float(max(scores))
        return float(min(scores))

    killers = (ctx or {}).get("killers", [])
    hist = (ctx or {}).get("history", {})
    moves = order_moves(board, tt_move=tt_move,
                        killers=killers[ply] if ply < len(killers) else None,
                        history=hist)

    if is_maximizing:
        best = -float('inf')
        best_move = None
        for move in moves:
            if _time_up(ctx):
                raise _TimeUp()
            board.push(move)
            try:
                value = minimax(board, depth - 1, alpha, beta, False,
                                nn_model, ctx, ply + 1, use_tt)
            finally:
                board.pop()
            if best_move is None or value > best:
                best, best_move = value, move
            alpha = max(alpha, value)
            if beta <= alpha:
                cutoff = True
                break
        else:
            cutoff = False
    else:
        best = float('inf')
        best_move = None
        for move in moves:
            if _time_up(ctx):
                raise _TimeUp()
            board.push(move)
            try:
                value = minimax(board, depth - 1, alpha, beta, True,
                                nn_model, ctx, ply + 1, use_tt)
            finally:
                board.pop()
            if best_move is None or value < best:
                best, best_move = value, move
            beta = min(beta, value)
            if beta <= alpha:
                cutoff = True
                break
        else:
            cutoff = False

    if cutoff:
        best_move = best_move or (moves[0] if moves else None)

    if cutoff and best_move is not None and not board.is_capture(best_move) \
            and best_move.promotion is None:
        slot = killers[ply] if ply < len(killers) else None
        if slot is not None and best_move not in slot:
            slot.insert(0, best_move)
            del slot[2:]
        if hist is not None:
            hist[(best_move.from_square, best_move.to_square)] = \
                hist.get((best_move.from_square, best_move.to_square), 0) + depth * depth

    if key is not None and depth >= 1:
        if cutoff:
            flag = FLAG_LOWER if is_maximizing else FLAG_UPPER
        else:
            flag = FLAG_EXACT
        _store_tt(key, depth, flag, _adj_store(best, ply), best_move)

    return best


def _root_ordered(board, prev_move, use_tt):
    \"\"\"Root move order: previous-iteration best move first, then the TT's
    stored best move, then the ordered legal moves.\"\"\"
    ordered = []
    if prev_move is not None and prev_move in board.legal_moves:
        ordered.append(prev_move)
    if use_tt:
        entry = _TT.get(zobrist_key(board))
        if entry is not None and entry[3] is not None \
                and entry[3] not in ordered and entry[3] in board.legal_moves:
            ordered.append(entry[3])
    for move in order_moves(board):
        if move not in ordered:
            ordered.append(move)
    return ordered


def find_best_move(board, depth=3, nn_model=None, time_limit=None):
    \"\"\"Best move for the side to move via iterative-deepening minimax +
    alpha-beta + a bounded transposition table (plus killers/history ordering).

    Iterative deepening runs depth 1, 2, 3, ... up to the `depth` ceiling or
    until `time_limit` seconds elapse (whichever comes first), reusing the
    previous iteration's best move at the root so deeper searches start from
    the strongest line. Returns the best move from the last fully completed
    depth. If no depth completes (tiny budget), returns the first root move.\"\"\"
    if board.is_game_over():
        return None
    legal = list(board.legal_moves)
    if not legal:
        return None
    if depth < 1:
        depth = 1

    ctx = {"deadline": None, "killers": [[] for _ in range(depth + _MAX_QDEPTH + 4)],
           "history": {}, "nodes": 0, "hit": False}
    if time_limit is not None:
        ctx["deadline"] = time.monotonic() + max(0.0, float(time_limit))

    white = board.turn == chess.WHITE
    use_tt = nn_model is None
    best_move = legal[0]
    prev_move = None
    try:
        for d in range(1, depth + 1):
            root_moves = _root_ordered(board, prev_move, use_tt)
            if white:
                best = -MATE
                found = None
                for move in root_moves:
                    if _time_up(ctx):
                        raise _TimeUp()
                    board.push(move)
                    try:
                        value = minimax(board, d - 1, -MATE, MATE, False,
                                        nn_model, ctx, 1, use_tt)
                    finally:
                        board.pop()
                    if found is None or value > best:
                        best, found = value, move
            else:
                best = MATE
                found = None
                for move in root_moves:
                    if _time_up(ctx):
                        raise _TimeUp()
                    board.push(move)
                    try:
                        value = minimax(board, d - 1, -MATE, MATE, True,
                                        nn_model, ctx, 1, use_tt)
                    finally:
                        board.pop()
                    if found is None or value < best:
                        best, found = value, move
            prev_move = found
            if found is not None:
                best_move = found
    except _TimeUp:
        pass

    return best_move""")


# --------------------------------------------------------------------------- #
# Section 5.2 - Neural move selection
# --------------------------------------------------------------------------- #
md("""### 5.2 Neural move selection (default)

`play_nn` evaluates **all legal moves in a single batched call** to the model
(the original called `predict` once per move, which was very slow).
""")

code("""@torch.no_grad()
def evaluate_positions(boards, model):
    \"\"\"Batch-evaluate a list of boards; returns np array of scores (Black POV).\"\"\"
    if not boards:
        return np.array([])
    tensors = np.stack([encode_board(b) for b in boards]).astype(np.float32)
    tensors = torch.tensor(tensors, device=DEVICE)
    model.eval()
    out = model(tensors)
    return out['value'].cpu().numpy() * VALUE_SCALE


def play_nn(fen, model, show_move_evaluations=False, player='b', use_minimax=False, depth=3, nn_leaf=False, time_limit=None):
    \"\"\"Pick the best move for the given FEN.

    Args:
        fen: FEN string of the current position
        model: trained evaluation model
        player: 'b' (Black, minimize score) or 'w' (White, maximize score)
        use_minimax: if True, use minimax search instead of pure neural eval
        depth: search depth when use_minimax=True (max depth ceiling)
        nn_leaf: when True, minimax leaf positions are batched through the net
            (slower); when False, a fast PST + quiescence search is used.
        time_limit: optional wall-clock budget (seconds) for iterative
            deepening. None keeps the old fixed-depth behavior.
    \"\"\"
    board = chess.Board(fen)

    if use_minimax:
        best = find_best_move(board, depth, nn_model=model if nn_leaf else None,
                              time_limit=time_limit)
        if show_move_evaluations:
            print(f'Best move using Minimax: {best}')
        return str(best)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    # Policy-based move selection: score each legal move with the net's
    # source/destination logits, then optionally break policy ties by value.
    tensors = np.stack([encode_board(board)]).astype(np.float32)
    tensors = torch.tensor(tensors, device=DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(tensors)
    p_from = out['from'][0].cpu().numpy()
    p_to = out['to'][0].cpu().numpy()

    move_scores = []
    for move in legal_moves:
        move_scores.append(
            p_from[move.from_square] + p_to[move.to_square])

    if show_move_evaluations:
        order = np.argsort(move_scores)[::-1]
        for rank in order[:10]:
            print(f'{legal_moves[rank]}: {move_scores[rank]:.2f}')

    best_idx = int(np.argmax(move_scores))
    return str(legal_moves[best_idx])
""")

# --------------------------------------------------------------------------- #
# Section 5.3 - Sanity check
# --------------------------------------------------------------------------- #
md("""### 5.3 Sanity check""")

code("""# Neural move selection from the starting position
board = chess.Board()
start = time.time()
nn_move = play_nn(board.fen(), model, show_move_evaluations=True, player='w')
print(f"NN picked: {nn_move} in {time.time() - start:.1f}s")

# Minimax search with neural leaf evaluation (depth 2 for speed)
start = time.time()
mm_move = play_nn(board.fen(), model, use_minimax=True, depth=2)
print(f"NN-leaf Minimax picked: {mm_move} in {time.time() - start:.1f}s")
""")

# --------------------------------------------------------------------------- #
# Section 6 - Playing the bot
# --------------------------------------------------------------------------- #
md("""## 6. Playing the Bot

### 6.1 Interactive play (Jupyter / CLI)

Play a game against the neural bot. Type `quit` to end early.
""")

code("""try:
    from IPython.display import SVG, display
except Exception:
    SVG = lambda s: s
    def display(*args, **kwargs):
        for _a in args:
            print(_a)

# If running in plain Python (non-Jupyter), print text boards instead of SVG.
try:
    from IPython import get_ipython
    IN_NOTEBOOK = get_ipython() is not None
except Exception:
    IN_NOTEBOOK = False


def show_board(board):
    if IN_NOTEBOOK:
        display(SVG(board._repr_svg_()))
    else:
        print(board)
        print()


def play_game(ai_function, human_player='w'):
    \"\"\"Play a game. human_player: 'w' = human is White, 'b' = human is Black.\"\"\"
    board = chess.Board()
    print(f"Playing as {'White' if human_player == 'w' else 'Black'}. Type 'quit' to end.")

    while board.outcome() is None:
        show_board(board)

        if (board.turn == chess.WHITE) == (human_player == 'w'):
            user_move = input('Your move: ').strip()
            if user_move.lower() == 'quit':
                break
            legal = [m for m in board.legal_moves]
            legal_uci = {m.uci(): m for m in legal}
            legal_san = {board.san(m): m for m in legal}

            if user_move in legal_uci:
                board.push(legal_uci[user_move])
            elif user_move in legal_san:
                board.push(legal_san[user_move])
            else:
                while user_move not in legal_uci and user_move not in legal_san:
                    print("That wasn't a valid move. Try again.")
                    user_move = input('Your move: ').strip()
                    if user_move.lower() == 'quit':
                        break
                if user_move.lower() != 'quit':
                    if user_move in legal_uci:
                        board.push(legal_uci[user_move])
                    else:
                        board.push(legal_san[user_move])
        else:
            ai_player = 'w' if board.turn == chess.WHITE else 'b'
            ai_move = ai_function(board.fen(), player=ai_player)
            print(f'AI move: {ai_move}')
            board.push_uci(ai_move)

    show_board(board)
    if board.outcome() is not None:
        result = board.result()
        winner = {'1-0': 'White wins', '0-1': 'Black wins', '1/2-1/2': 'Draw'}
        print(winner.get(result, result))


# Strong default engine: fast PST + quiescence minimax search.
def bot_move(fen, player='w', model=model, use_minimax=True, depth=2, nn_leaf=False):
    \"\"\"Pick the strongest move. Defaults to minimax with quiescence search.\"\"\"
    return play_nn(fen, model, use_minimax=use_minimax, depth=depth,
                   nn_leaf=nn_leaf, player=player)


# Example: play as White against the bot
# play_game(bot_move, human_player='w')
""")

# --------------------------------------------------------------------------- #
# Section 6.2 - Stockfish game
# --------------------------------------------------------------------------- #
md("""### 6.2 Play vs Stockfish""")

code("""def get_stockfish_engine(stockfish_path=STOCKFISH_PATH):
    \"\"\"Start a Stockfish engine. Auto-discovers the binary if path is None.\"\"\"
    if stockfish_path is None:
        candidates = ["stockfish", "stockfish.exe", "./stockfish", "./stockfish.exe"]
        for candidate in candidates:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(candidate)
                print(f"Found Stockfish at: {candidate}")
                return engine
            except Exception:
                continue
        raise FileNotFoundError(
            "Stockfish not found. Download it from https://stockfishchess.org and "
            "set STOCKFISH_PATH in Section 1."
        )
    return chess.engine.SimpleEngine.popen_uci(stockfish_path)


def play_against_stockfish(ai_function, bot_is_white=False, stockfish_elo=1320,
                           time_limit=0.1, stockfish_path=STOCKFISH_PATH):
    \"\"\"Play a single game: the AI vs Stockfish, with board visualization.\"\"\"
    board = chess.Board()
    engine = None
    try:
        engine = get_stockfish_engine(stockfish_path)
        if stockfish_elo is not None:
            engine.configure({
                "UCI_LimitStrength": True,
                "UCI_Elo": max(int(stockfish_elo), 1320),  # Stockfish min Elo is 1320
            })

        bot_color_chr = 'w' if bot_is_white else 'b'
        while board.outcome() is None:
            show_board(board)
            if board.turn == chess.WHITE:
                if bot_is_white:
                    move = ai_function(board.fen(), player='w')
                    print(f"AI (White) move: {move}")
                    board.push_uci(move)
                else:
                    info = engine.play(board, chess.engine.Limit(time=time_limit))
                    print(f"Stockfish (White) move: {info.move}")
                    board.push(info.move)
            else:
                if bot_is_white:
                    info = engine.play(board, chess.engine.Limit(time=time_limit))
                    print(f"Stockfish (Black) move: {info.move}")
                    board.push(info.move)
                else:
                    move = ai_function(board.fen(), player='b')
                    print(f"AI (Black) move: {move}")
                    board.push_uci(move)

        show_board(board)
        return board.result()
    finally:
        if engine is not None:
            engine.quit()


# Example:
# print(play_against_stockfish(bot_move, bot_is_white=False, stockfish_elo=1320))
""")

# --------------------------------------------------------------------------- #
# Section 6.3 - Rating the bot
# --------------------------------------------------------------------------- #
md("""### 6.3 Rating the bot against Stockfish

Plays a series of games at increasing Stockfish strengths and tracks Elo.
""")

code("""def calculate_elo(old_rating, opponent_rating, result, k_factor=32):
    \"\"\"Update Elo after one game.\n    result: 1.0 = win, 0.5 = draw, 0.0 = loss.\"\"\"
    expected = 1 / (1 + 10 ** ((opponent_rating - old_rating) / 400))
    return old_rating + k_factor * (result - expected)


def simulate_game(ai_function, stockfish_elo, bot_is_white=False,
                  time_limit=0.1, stockfish_path=STOCKFISH_PATH):
    \"\"\"Play one game vs Stockfish; returns 1.0/0.5/0.0 for the bot.\"\"\"
    board = chess.Board()
    engine = None
    try:
        engine = get_stockfish_engine(stockfish_path)
        engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": max(int(stockfish_elo), 1320),
        })

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                if bot_is_white:
                    board.push_uci(ai_function(board.fen(), player='w'))
                else:
                    info = engine.play(board, chess.engine.Limit(time=time_limit))
                    board.push(info.move)
            else:
                if bot_is_white:
                    info = engine.play(board, chess.engine.Limit(time=time_limit))
                    board.push(info.move)
                else:
                    board.push_uci(ai_function(board.fen(), player='b'))

        result = board.result()
    finally:
        if engine is not None:
            engine.quit()

    # Result is from White's perspective; convert to the bot's points.
    if bot_is_white:
        bot_points = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
    else:
        bot_points = {"1-0": 0.0, "0-1": 1.0, "1/2-1/2": 0.5}[result]
    return bot_points


def rate_bot(ai_function, initial_rating=1500,
             stockfish_elos=(1320, 1360, 1400, 1600, 1800),
             num_games=3, time_limit=0.1, stockfish_path=STOCKFISH_PATH):
    \"\"\"Rate the bot by playing several games at each Stockfish Elo.\"\"\"
    bot_rating = initial_rating
    results_summary = {}

    for elo in stockfish_elos:
        wins = draws = losses = 0
        for i in range(num_games):
            bot_is_white = (i % 2 == 0)
            result = simulate_game(
                ai_function, elo, bot_is_white=bot_is_white,
                time_limit=time_limit, stockfish_path=stockfish_path,
            )
            if result == 1.0:
                wins += 1
            elif result == 0.5:
                draws += 1
            else:
                losses += 1
            bot_rating = calculate_elo(bot_rating, elo, result)
            print(f"Elo {elo} game {i+1}: "
                  f"{'Win' if result == 1 else 'Draw' if result == 0.5 else 'Loss'} "
                  f"-> rating {bot_rating:.0f}")

        results_summary[elo] = (wins, draws, losses)
        print(f"Elo {elo}: {num_games} games -> "
              f"{wins}W / {draws}D / {losses}L")

    print("\\n" + "=" * 40)
    print(f"Final bot rating: {bot_rating:.0f}")
    print("=" * 40)
    return bot_rating, results_summary


# -----------------------------------------------------------------------------
# Uncomment to rate the bot. This takes a while (needs Stockfish downloaded).
# -----------------------------------------------------------------------------
# final_rating, summary = rate_bot(bot_move, num_games=3)
if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
    print(f"Stockfish found at: {STOCKFISH_PATH}")
    print("Uncomment the rate_bot(...) line above to rate the bot "
          "(takes a few minutes).")
else:
    print("Rating needs Stockfish. Download it and set STOCKFISH_PATH, then "
          "run the rating cell above.")
""")

# --------------------------------------------------------------------------- #
# Final cell
# --------------------------------------------------------------------------- #
md("""## Summary of improvements

- **Encoding**: 8x8x19 with turn + castling + en-passant channels
- **Model**: deeper PyTorch net with BatchNorm + Dropout, Adam optimizer
- **Data**: proper shuffled train/val/test split
- **Move selection**: batched neural evaluation + classical minimax with
  alpha-beta, move ordering, PST and quiescence search
- **Gameplay**: Jupyter + CLI support, play as White or Black, vs Stockfish
- **Rating**: Elo rating against Stockfish with alternating colors
- **Persistence**: models can be saved and reloaded
""")


# --------------------------------------------------------------------------- #
# Write notebook
# --------------------------------------------------------------------------- #
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = r"C:\Users\Admin\Downloads\AI_Chess_Model-main\AI_Chess_Model-main\chess_main.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to {output_path} with {len(cells)} cells")