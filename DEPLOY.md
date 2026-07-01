# Deploy SHL Assessment Recommender

## Option A — Render (recommended, free tier)

### 1. Push code to GitHub

```powershell
cd "c:\Users\Mridhul\Downloads\AI Project"
git init
git add .
git commit -m "SHL conversational assessment recommender"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/shl-recommender.git
git push -u origin main
```

### 2. Create Render Web Service

1. Go to [https://render.com](https://render.com) and sign up.
2. **New → Web Service** → connect your GitHub repo.
3. Settings:
   - **Runtime:** Docker
   - **Dockerfile path:** `./Dockerfile`
   - **Health check path:** `/health`
   - **Plan:** Free
4. Optional env var: `OPENAI_API_KEY` (works without it).
5. Click **Deploy**.

First deploy takes ~5–10 minutes (downloads embedding model).

### 3. Smoke-test public URL

Replace `YOUR-APP` with your Render hostname:

```powershell
curl https://YOUR-APP.onrender.com/health

curl -X POST https://YOUR-APP.onrender.com/chat `
  -H "Content-Type: application/json" `
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"I need an assessment\"}]}"
```

Expected:
- `/health` → `{"status":"ok"}`
- `/chat` vague opener → `recommendations: []` and a clarifying question

### 4. Submit to SHL

In the SHL form, submit:
- **Public URL:** `https://YOUR-APP.onrender.com`
- **Approach doc:** `APPROACH.md` (export to PDF if required)

---

## Option B — Railway

1. Push to GitHub (same as above).
2. [https://railway.app](https://railway.app) → New Project → Deploy from GitHub.
3. Railway auto-detects `Dockerfile`.
4. Set port `8000`, generate public domain.
5. Test `/health` and `/chat` as above.

---

## Option C — Local Docker (test before cloud)

Install Docker Desktop, then:

```powershell
cd "c:\Users\Mridhul\Downloads\AI Project"
docker build -t shl-recommender .
docker run -p 8000:8000 shl-recommender
```

Test: `http://localhost:8000/health`

---

## Pre-submit checklist

Run locally before submitting:

```powershell
python scripts/build_catalog.py
python scripts/eval_chat.py
python -m pytest tests/ -v
```

Confirm:
- [ ] Public `/health` returns 200
- [ ] Public `/chat` returns valid JSON schema
- [ ] `data/eval_results.json` shows strong Recall@10
- [ ] All pytest probes pass
- [ ] `APPROACH.md` attached

---

## Cold start note

First `/chat` after idle may take 30–90s (loads sentence-transformers model). SHL allows up to **2 minutes** for `/health` on cold start. Subsequent calls are fast.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Build fails on Render | Check Dockerfile uses Python 3.11; `requirements.txt` pinned |
| `/health` OK but `/chat` timeout | Wait for model load; retry; upgrade to paid tier if needed |
| Out of memory on free tier | Reduce concurrent requests; use Render starter plan |
| 502 on first request | Normal cold start — retry after 60s |
