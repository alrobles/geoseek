"""Similarity explorer backend.

Serves precomputed Clay v1.5 embeddings (lat/lon tokens zeroed —
imagery-only) over the xnicher Rhipicephalus microplus accessible-area
grid and answers k-NN cosine-similarity queries. No torch, no model
weights: similarity is a matmul over a normalized embedding matrix.

Run:   uvicorn demo.app:app --host 0.0.0.0 --port 8600
Data:  demo/data/grid_embeddings.npz   (scripts/05_build_demo_grid.py)
       DEMO_NPZ env var overrides the path (dev data, tests).
"""

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles

DEMO = Path(__file__).resolve().parent

NPZ = Path(os.environ.get("DEMO_NPZ",
                         DEMO / "data" / "grid_embeddings.npz"))
z = np.load(NPZ)
IDS = z["ids"].astype(np.int64)
LONLAT = z["lonlat"].astype(np.float64)
BIO1, RH, DT = z["bio1"], z["rh"], z["datetime"]
E = z["emb_mean"].astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
IDPOS = {int(i): p for p, i in enumerate(IDS.tolist())}
COSLAT = np.cos(np.radians(LONLAT[:, 1]))

# optional suitability surface (scripts/06_suitability.py): presence-
# vs-background probe scores per node — an embedding-signal layer,
# not a validated SDM.
_suit = DEMO / "data" / f"suitability_{NPZ.stem.split('_')[0]}.npz"
SUIT = {}
if _suit.exists():
    s = np.load(_suit)
    SUIT = {int(i): float(p) for i, p in
            zip(s["ids"].tolist(), s["prob"].tolist())}

app = FastAPI(title="EO-FM similarity explorer")


def carto_key():
    """CARTO basemap key for the client (public-embed key, like a
    Mapbox token): env CARTO_KEY, else file at CARTO_KEY_FILE, else
    demo/data/carto-key; empty string if none."""
    if os.environ.get("CARTO_KEY"):
        return os.environ["CARTO_KEY"].strip()
    for p in (os.environ.get("CARTO_KEY_FILE"), DEMO / "data" / "carto-key"):
        if p and Path(p).exists():
            return Path(p).read_text().strip()
    return ""


@app.get("/api/config")
def config():
    return {"carto_key": carto_key()}


def row(p, sim=None):
    d = {"id": int(IDS[p]), "lon": float(LONLAT[p, 0]),
         "lat": float(LONLAT[p, 1])}
    if sim is not None:
        d["sim"] = round(float(sim), 4)
    if SUIT:
        d["suit"] = round(SUIT.get(int(IDS[p]), np.nan), 3)
    return d


@app.get("/api/points")
def points():
    d = {"ids": IDS.tolist(),
         "lonlat": np.round(LONLAT, 5).tolist(),
         "n": len(IDS)}
    if SUIT:
        d["suit"] = [SUIT.get(int(i)) for i in IDS.tolist()]
    return d


@app.get("/api/similar")
def similar(lon: float, lat: float, k: int = Query(48, ge=1, le=500)):
    # snap the click to the nearest grid node (cos-lat weighted)
    d = ((LONLAT[:, 0] - lon) * COSLAT) ** 2 + (LONLAT[:, 1] - lat) ** 2
    qi = int(d.argmin())
    sims = E @ E[qi]
    top = np.argpartition(-sims, min(k + 1, len(sims) - 1))[:k + 1]
    top = top[np.argsort(-sims[top])]
    top = [p for p in top if p != qi][:k]
    return {"query": row(qi),
            "results": [row(p, sims[p]) for p in top]}


@app.get("/api/meta/{pid}")
def meta(pid: int):
    p = IDPOS.get(pid)
    if p is None:
        return {"error": "unknown id"}
    return {**row(p), "bio1": float(BIO1[p]), "rh": float(RH[p]),
            "datetime": str(DT[p])}


app.mount("/", StaticFiles(directory=DEMO / "static", html=True),
          name="static")
