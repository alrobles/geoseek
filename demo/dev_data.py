"""Build dev assets for the similarity explorer from the existing
668-point clay_noll embeddings — lets the app run before the full
M-grid build finishes. Grid ids are offset +1e6 so thumbnails never
collide with real grid thumbs.

Run:  python demo/dev_data.py   ->  demo/data/dev.npz + dev thumbs
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, DEMO = ROOT / "data", ROOT / "demo"
OFFSET = 1_000_000

spec = importlib.util.spec_from_file_location(
    "grid_build", ROOT / "scripts" / "05_build_demo_grid.py")
gb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gb)


def main():
    z = np.load(DATA.parent / "results" / "embeddings_clay_noll.npz")
    pts = pd.read_csv(DATA / "points.csv").set_index("id")
    env = pd.read_csv(DATA / "env.csv").set_index("id")
    ids = z["ids"]
    dev_ids = ids + OFFSET
    dts = [str(np.load(DATA / "chips" / "clay" / f"{i}.npz")["datetime"])
           for i in ids]
    np.savez(DEMO / "data" / "dev.npz",
             ids=dev_ids.astype(np.int64),
             lonlat=pts.loc[ids, ["lon", "lat"]].values.astype(np.float32),
             emb_mean=z["emb_mean"].astype(np.float32),
             emb_cls=z["emb_cls"].astype(np.float32),
             bio1=env.loc[ids, "bio1"].values.astype(np.float32),
             rh=env.loc[ids, "rh"].values.astype(np.float32),
             datetime=np.array(dts))
    # thumbs under dev ids
    gb.THUMB_DIR.mkdir(parents=True, exist_ok=True)
    for i, did in zip(ids, dev_ids):
        out = gb.THUMB_DIR / f"{did}.jpg"
        if out.exists():
            continue
        src = DATA / "chips" / "clay" / f"{i}.npz"
        from PIL import Image
        px = np.load(src)["pixels"].astype(np.float32)[[6, 2, 1]]
        lo, hi = np.percentile(px, 2), np.percentile(px, 98)
        rgb = np.clip((px - lo) / (hi - lo + 1e-9), 0, 1)
        Image.fromarray((rgb.transpose(1, 2, 0) * 255).astype(np.uint8)) \
            .resize((128, 128), Image.BILINEAR).save(out, quality=82)
    print(f"dev data: {len(ids)} points -> {DEMO/'data'/'dev.npz'}")


if __name__ == "__main__":
    main()
