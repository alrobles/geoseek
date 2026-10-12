"""Fetch Sentinel-2 (Clay spec) and HLS-S30 (Prithvi spec) chips for each point.

- Clay  : 10 bands, 256x256 px @10 m, centred on the point  (2.56 km window)
- Prithvi: 6 bands, 224x224 px @30 m, centred on the point  (6.72 km window)

Each point gets the lowest-cloud item inside a fixed date window.
Chips are cached as .npz under data/chips/{clay,prithvi}/<id>.npz
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import stackstac

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DATE_RANGE = "2024-01-01/2025-07-01"
MAX_CLOUD = 40.0
# PC sentinel-2-l2a assets, mapped to Clay's band_order:
# B02 blue, B03 green, B04 red, B05-07 rededge1-3, B08 nir, B8A nir08,
# B11 swir16, B12 swir22
CLAY_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
              "B11", "B12"]
PRITHVI_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]
CLAY_SIZE, CLAY_RES = 256, 10          # 2.56 km
PRITHVI_SIZE, PRITHVI_RES = 224, 30  # 6.72 km

E84 = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
PC = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)


def m_to_deg_half(half_m, lat):
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * math.cos(math.radians(lat)))
    return dlon, dlat


def best_items(catalog, collection, lon, lat, n=3):
    search = catalog.search(
        collections=[collection],
        datetime=DATE_RANGE,
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        max_items=50,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
    )
    items = list(search.items())
    items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    return items[:n]


def utm_epsg(lon, lat):
    zone = int((lon + 180.0) / 6.0) + 1
    return (32700 if lat < 0 else 32600) + zone


def chip_from_item(item, bands, size, res, lon, lat, retries=3):
    # request one extra pixel of margin; centre-crop guarantees size x size
    half = size * res / 2.0 + res
    dlon, dlat = m_to_deg_half(half, lat)
    bounds = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    epsg = utm_epsg(lon, lat)
    for attempt in range(retries):
        try:
            arr = stackstac.stack(
                [item], assets=bands, bounds_latlon=bounds,
                resolution=res, epsg=epsg, fill_value=0,
            )
            chip = arr.compute().values  # [time, band, y, x]
            chip = chip.reshape(-1, chip.shape[2], chip.shape[3])
            # centre-crop to the requested pixel window
            _, h, w = chip.shape
            if h < size or w < size:
                raise ValueError(f"chip too small {chip.shape}")
            y0, x0 = (h - size) // 2, (w - size) // 2
            return chip[:, y0:y0 + size, x0:x0 + size]
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def fetch_point(row, model):
    pid, lon, lat = int(row["id"]), float(row["lon"]), float(row["lat"])
    out_dir = DATA / "chips" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pid}.npz"
    if out_path.exists():
        return pid, "cached", None, None

    if model == "clay":
        # Element84's sentinel-s2-l2a bucket is RequesterPays and returns
        # zeros without signed reads; PC serves the same L2A product for
        # free with per-item SAS signing.
        catalog, collection, bands, size, res = (
            PC, "sentinel-2-l2a", CLAY_BANDS, CLAY_SIZE, CLAY_RES)
    else:
        catalog, collection, bands, size, res = (
            PC, "hls2-s30", PRITHVI_BANDS, PRITHVI_SIZE, PRITHVI_RES)

    try:
        items = best_items(catalog, collection, lon, lat)
        if not items:
            return pid, "no_item", None, None
        for item in items:
            chip = chip_from_item(item, bands, size, res, lon, lat)
            zero_frac = float((chip == 0).mean())
            if zero_frac > 0.9:
                continue  # empty read (e.g. unsigned requester-pays asset)
            np.savez_compressed(
                out_path, pixels=chip.astype(np.int16),
                item_id=item.id,
                cloud=item.properties.get("eo:cloud_cover"),
                datetime=item.datetime.isoformat() if item.datetime else "",
                zero_frac=zero_frac,
            )
            return pid, "ok", item.id, item.properties.get("eo:cloud_cover")
        return pid, "empty_chips", items[0].id, None
    except Exception as e:  # noqa: BLE001
        return pid, f"error:{type(e).__name__}:{e}", None, None


def main(model):
    (DATA / "chips" / model).mkdir(parents=True, exist_ok=True)
    pts = pd.read_csv(DATA / "points.csv")
    rows = [row for _, row in pts.iterrows()]
    manifest = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_point, row, model): i
                for i, row in enumerate(rows)}
        for i, fut in enumerate(as_completed(futs)):
            pid, status, item_id, cloud = fut.result()
            manifest.append(dict(id=pid, model=model, status=status,
                                 item_id=item_id, cloud=cloud))
            if (i + 1) % 25 == 0 or i == len(rows) - 1:
                done = sum(1 for m in manifest
                           if m["status"] in ("ok", "cached"))
                print(f"[{model}] {i + 1}/{len(rows)}  ok+cached={done}  "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(manifest).to_csv(DATA / f"manifest_{model}.csv", index=False)
    bad = [m for m in manifest if m["status"].startswith("error")]
    print(f"[{model}] done. statuses:",
          pd.Series(m["status"] for m in manifest).value_counts().to_dict())


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "clay")
