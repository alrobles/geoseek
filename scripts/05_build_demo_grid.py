"""Build the demo-grid assets for the live similarity explorer.

Source points: the 9,784 accessible-area (M) coordinates bundled in
xnicher (data/m_full.csv). For each point:
  fetch   S2 chip (PC sentinel-2-l2a, Clay spec) -> data/chips/grid/<ID>.npz
  embed   Clay v1.5 with lat/lon tokens ZEROED   -> imagery-only vectors
  thumbs  NIR-R-G thumbnail                      -> demo/static/thumbs/<ID>.jpg

Final output: demo/data/grid_embeddings.npz
  ids (N,), lonlat (N,2), emb_mean (N,1024), emb_cls (N,1024),
  bio1 (N,), rh (N,), datetime (N,) str

All phases are resumable: existing chips, embedded ids (checkpoint file)
and thumbnails are skipped on rerun.

Usage: python scripts/05_build_demo_grid.py [fetch|embed|thumbs|all]
                                             [--limit N]
"""

import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, DEMO = ROOT / "data", ROOT / "demo"
GRID_DIR = DATA / "chips" / "grid"
THUMB_DIR = DEMO / "static" / "thumbs"
PARTIAL = DEMO / "data" / "grid_emb_partial.npz"
FINAL = DEMO / "data" / "grid_embeddings.npz"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fetch_mod = _load("fetch_chips", ROOT / "scripts" / "01_fetch_chips.py")


def grid_points():
    m = pd.read_csv(DATA / "m_full.csv")
    return m.rename(columns={"ID": "id"})[["id", "lon", "lat", "bio1", "rh"]]


# ------------------------------------------------------------------ fetch
def fetch_one(row):
    pid, lon, lat = int(row["id"]), float(row["lon"]), float(row["lat"])
    out_path = GRID_DIR / f"{pid}.npz"
    if out_path.exists():
        return pid, "cached", None
    try:
        items = fetch_mod.best_items(fetch_mod.PC, "sentinel-2-l2a",
                                     lon, lat)
        if not items:
            return pid, "no_item", None
        for item in items:
            chip = fetch_mod.chip_from_item(
                item, fetch_mod.CLAY_BANDS, fetch_mod.CLAY_SIZE,
                fetch_mod.CLAY_RES, lon, lat)
            zero_frac = float((chip == 0).mean())
            if zero_frac > 0.9:
                continue
            np.savez_compressed(
                out_path, pixels=chip.astype(np.int16),
                item_id=item.id,
                cloud=item.properties.get("eo:cloud_cover"),
                datetime=item.datetime.isoformat() if item.datetime else "",
                zero_frac=zero_frac,
            )
            return pid, "ok", item.id
        return pid, "empty_chips", items[0].id
    except Exception as e:  # noqa: BLE001
        return pid, f"error:{type(e).__name__}:{e}", None


def do_fetch(pts):
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    rows = [r for _, r in pts.iterrows()]
    manifest, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(fetch_one, r) for r in rows]
        for i, fut in enumerate(as_completed(futs)):
            pid, status, item_id = fut.result()
            manifest.append(dict(id=pid, model="grid", status=status,
                                 item_id=item_id))
            if (i + 1) % 100 == 0 or i == len(rows) - 1:
                done = sum(1 for m in manifest
                           if m["status"] in ("ok", "cached"))
                print(f"[fetch] {i + 1}/{len(rows)} ok+cached={done} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(manifest).to_csv(DATA / "manifest_grid.csv", index=False)
    print("[fetch] statuses:",
          pd.Series(m["status"] for m in manifest).value_counts().to_dict())


# ------------------------------------------------------------------ embed
def do_embed(pts, block=512):
    embed_mod = _load("embed", ROOT / "scripts" / "02_embed.py")
    done_ids = set()
    parts = {"ids": [], "cls": [], "mean": []}
    if PARTIAL.exists():
        z = np.load(PARTIAL)
        done_ids = set(z["ids"].tolist())
        parts = {k: [z[k]] for k in ("ids", "cls", "mean")}
        print(f"[embed] resuming at {len(done_ids)}")
    todo = pts[~pts["id"].isin(done_ids)]
    print(f"[embed] {len(todo)} points to embed")
    DEMO_DATA = PARTIAL.parent
    DEMO_DATA.mkdir(parents=True, exist_ok=True)
    for i0 in range(0, len(todo), block):
        chunk = todo.iloc[i0:i0 + block]
        ids, cls, mean = embed_mod.run_clay(
            chunk, batch=16, zero_latlon=True, chips_dir=GRID_DIR)
        parts["ids"].append(np.asarray(ids))
        parts["cls"].append(cls)
        parts["mean"].append(mean)
        np.savez(PARTIAL, ids=np.concatenate(parts["ids"]),
                 cls=np.concatenate(parts["cls"]),
                 mean=np.concatenate(parts["mean"]))
        n = len(np.concatenate(parts["ids"]))
        print(f"[embed] checkpoint {n}/{len(pts)}", flush=True)

    z = np.load(PARTIAL)
    pos = {pid: i for i, pid in enumerate(z["ids"].tolist())}
    keep = [pos[pid] for pid in pts["id"] if pid in pos]
    meta = pts[pts["id"].isin(set(z["ids"].tolist()))].set_index("id")
    np.savez(
        FINAL,
        ids=z["ids"][keep].astype(np.int64),
        lonlat=meta[["lon", "lat"]].values.astype(np.float32),
        emb_mean=z["mean"][keep].astype(np.float32),
        emb_cls=z["cls"][keep].astype(np.float32),
        bio1=meta["bio1"].values.astype(np.float32),
        rh=meta["rh"].values.astype(np.float32),
        datetime=np.array([str(np.load(GRID_DIR / f"{i}.npz")["datetime"])
                           for i in z["ids"][keep]]),
    )
    print(f"[embed] wrote {FINAL} n={len(keep)}")


# ------------------------------------------------------------------ thumbs
def thumb_one(path, size=128):
    from PIL import Image
    z = np.load(path)
    px = z["pixels"].astype(np.float32)[[6, 2, 1]]  # NIR, R, G
    lo, hi = np.percentile(px, 2), np.percentile(px, 98)
    rgb = np.clip((px - lo) / (hi - lo + 1e-9), 0, 1)
    img = Image.fromarray(
        (rgb.transpose(1, 2, 0) * 255).astype(np.uint8))
    img.resize((size, size), Image.BILINEAR).save(
        THUMB_DIR / f"{path.stem}.jpg", quality=82)


def do_thumbs(pts):
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    files = [GRID_DIR / f"{pid}.npz" for pid in pts["id"]]
    files = [f for f in files if f.exists()
             and not (THUMB_DIR / f"{f.stem}.jpg").exists()]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(thumb_one, files))
    print(f"[thumbs] {len(files)} rendered -> {THUMB_DIR}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    phase = args[0] if args else "all"
    limit = (int(sys.argv[sys.argv.index("--limit") + 1])
             if "--limit" in sys.argv else None)
    pts = grid_points()
    if limit:
        pts = pts.head(limit)
    if phase in ("fetch", "all"):
        do_fetch(pts)
    if phase in ("embed", "all"):
        do_embed(pts)
    if phase in ("thumbs", "all"):
        do_thumbs(pts)
