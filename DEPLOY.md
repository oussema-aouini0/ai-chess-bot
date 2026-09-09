# Deploying the chess bot publicly on Render (free, no credit card)

Turns the local Flask board into a public URL anyone can open and play.
~4 MB of files get deployed — **no Stockfish, no training data**.

All Render-specific files are already in this repo:

| File | Role |
|---|---|
| `render.yaml` | Render Blueprint: free web service, Docker, `/healthz`, env vars |
| `Dockerfile` | Runtime image (~300 MB, CPU-only torch), binds to `$PORT` |
| `requirements-deploy.txt` | Runtime deps (`gunicorn` included) |
| `.dockerignore` | Keeps Stockfish + CSV out of the build context |
| `chess_web.py`, `chess_bot.py`, `chess_main.ipynb`, `chess_model.pt` | The app + engine + weights |

## Free-tier facts (2026, verified)

- **$0/month, no credit card.** Hobby workspace + free web service.
- 512 MB RAM / 0.1 CPU — we measured the app at **~370 MB RSS**, and moves
  stay well under the limit.
- **Spins down after 15 min idle** → first visitor after a pause waits ~1 min
  while it wakes. Free plan only; `$7/mo` removes that.
- 750 instance-hours/month (≈ 30 days of always-running) and 500 build minutes.

## 1. Put the code on GitHub (free)

1. Create an account at <https://github.com> and a **new repository**
   (public or private — Render can read private if you authorize it).
2. Upload this folder as the repo's content (or `git init` locally and push).
   Make sure these files are at the **repo root**:
   `render.yaml`, `Dockerfile`, `requirements-deploy.txt`, `.dockerignore`,
   `chess_web.py`, `chess_bot.py`, `chess_main.ipynb`, `chess_model.pt`.
   (`stockfish/` and the CSV do **not** need to be uploaded at all.)

Local one-liner if you use Git:

```bash
git init
git add -A
git commit -m "AI chess bot"
# then: create the repo on github.com and push:
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git branch -M main
git push -u origin main
```

## 2. Deploy to Render (free)

1. Sign up at <https://render.com> (GitHub login works; **no card**).
2. In the dashboard: **New + → Blueprint**.
3. Pick your GitHub repo; Render reads `render.yaml` and creates the
   **chess-bot** web service on the **free** plan automatically.
4. Click **Apply**; watch the build log (torch CPU wheel download → a few
   minutes on the first build).
5. When it says **Live**, open the service URL —
   `https://chess-bot.onrender.com`.

Every `git push` to the repo re-deploys automatically.

## 3. Verify

- Open the public URL, play a game as White, then as Black.
- `https://chess-bot.onrender.com/healthz` should return `ok`.
- Test the exposure: every running process is confined to the app container —
  the engine only parses FENs and runs inference; nothing is written to disk.

## Optional tweaks

- **Concurrency**: `MAX_CONCURRENT` (parallel AI searches, default 4) and
  `MAX_DEPTH` (default 4) are set in `render.yaml`; adjust and re-push.
- **Always-on**: the free instance sleeps after 15 min idle. Upgrading the
  compute plan in the Dashboard stops the sleep, but that adds cost.
- **Custom domain**: Render Dashboard → service → *Settings* → *Custom Domains*
  (free TLS).

## Local check before pushing

Simulate the production WSGI flow on Windows (waitress = same gunicorn-style
WSGI, Windows-compatible):

```bash
C:/Users/Admin/anaconda3/python.exe -m pip install waitress
C:/Users/Admin/anaconda3/python.exe -m waitress --listen=127.0.0.1:5000 --threads=6 chess_web:app
curl http://127.0.0.1:5000/healthz    # -> 200 ok
```

On Linux/macOS (or inside the container) the real command is the one in the
Dockerfile:

```bash
gunicorn -b 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 60 chess_web:app
```

## How to deploy from a Windows machine

You do not need Docker on your PC — Render builds the image on their servers
from the repo. Just push the files to GitHub, and Render does the rest.