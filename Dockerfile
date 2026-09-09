# Deploy image for the AI Chess Bot (Render free web service / any Docker host).
# Deliberately small: only the runtime files are copied (no Stockfish, no CSV).

FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first (a CPU wheel is much smaller than the CUDA bundle).
RUN pip install --no-cache-dir torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Files needed to play: the engine, its notebook source, the app and the model.
COPY chess_web.py chess_bot.py chess_main.ipynb chess_model.pt ./

# Render injects PORT; keep a sane default for local `docker run`.
ENV PORT=7860 \
    OMP_NUM_THREADS=2 \
    TORCH_THREADS=2

EXPOSE 7860

# 1 worker (keeps the process under 512 MB), 4 threads, bound to $PORT.
CMD gunicorn -b 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 60 chess_web:app