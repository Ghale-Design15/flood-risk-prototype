# React frontend (Vercel) — River Murray Flood Risk

Single-screen operations view for the Week 11 demo. It consumes the same API as the
Streamlit dashboard (see `docs/API_CONTRACT.md`), so both tell the same story.

What is on the screen:

- **Model dropdown** built from `GET /models` (the ensemble is the default).
- **Risk verdict**: flood probability and band for the selected horizon.
- **LGA map**: the three council areas from `data/lga_boundaries.geojson`. Murray Bridge
  is tinted with the current risk band; the others stay neutral because only Murray
  Bridge has a trained model.
- **Why this score**: the four model inputs behind the prediction.
- **Alert authorisation**: two-step confirm, then `POST /alerts` (hash-chained audit log
  and SendGrid email on the backend).

Predictions come from `POST /predict_series`. Nothing is scored in the browser.

## Run locally

The backend must be running first (from the repo root):

```bash
python -m uvicorn main:app --app-dir backend --port 8077
```

Then:

```bash
cd frontend
npm install
cp .env.example .env.local        # then set VITE_API_URL=http://127.0.0.1:8077
npm run dev
```

Open the URL Vite prints (default <http://localhost:5173>).

## Deploy to Vercel

The app lives in a subfolder of the team repo, so point Vercel at that folder.

1. Push the branch to GitHub.
2. <https://vercel.com> → **Add New** → **Project** → import `jumisaji/flood-risk-prototype`.
3. **Root Directory**: `frontend` (this is the important one).
   Framework preset **Vite**, build `npm run build`, output `dist` are detected automatically.
4. **Environment Variables**: add `VITE_API_URL` = the Render API URL
   (for example `https://flood-risk-prototype-week10.onrender.com`), no trailing slash.
5. **Deploy**. Vercel gives the public URL.

`VITE_API_URL` is read at **build** time, so after changing it use **Redeploy**, not just
a refresh.

## Notes for the demo

- The free Render instance sleeps after about 15 minutes idle. The screen pings `/health`
  on load and shows "Warming up" rather than blocking, but run `python backend/warmup.py`
  a few minutes before presenting so the first prediction is instant.
- `/alerts` needs `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` on the backend. Without them
  it returns 503 and the screen says "Authorised locally", which is the honest state.
- The API must allow the Vercel origin via CORS. The backend currently allows all origins,
  so nothing to change.
