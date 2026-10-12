"""Extract frozen embeddings for every cached chip.

- clay   -> data/chips/clay/*.npz     (10x256x256) -> CLS embedding, 1024-d
- prithvi-> data/chips/prithvi/*.npz  (6x224x224)  -> CLS + mean-pooled tokens

Output: results/embeddings_<model>.npz with arrays
  ids      (N,)   point ids
  emb_cls  (N,D)
  emb_mean (N,D)  patch-token mean (clay: all tokens incl. cls)
"""

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
sys.path.insert(0, str(ROOT / "vendor" / "prithvi"))

DEVICE = "cpu"
torch.set_num_threads(int(__import__("os").environ.get("EMB_THREADS", 4)))


# ---------------------------------------------------------------- clay
def clay_embedder():
    from claymodel.finetune.embedder.factory import Embedder
    emb = Embedder(img_size=256,
                   ckpt_path=str(ROOT / "checkpoints" / "clay_v15_encoder.pt"),
                   device=DEVICE)
    meta = yaml.safe_load(open(ROOT / "vendor" / "clay" / "metadata.yaml"))
    s2 = meta["sentinel-2-l2a"]
    order = s2["band_order"]
    mean = torch.tensor([s2["bands"]["mean"][b] for b in order])
    std = torch.tensor([s2["bands"]["std"][b] for b in order])
    waves = torch.tensor([s2["bands"]["wavelength"][b] for b in order],
                         dtype=torch.float32)
    gsd = torch.tensor([float(s2["gsd"])], dtype=torch.float32)
    return emb, mean, std, waves, gsd


def clay_time_latlon(dt_str, lat, lon):
    import datetime
    d = datetime.datetime.fromisoformat(dt_str.replace("Z", "+00:00")) \
        if dt_str else datetime.datetime(2024, 6, 1)
    week = d.isocalendar().week * 2 * math.pi / 52
    hour = d.hour * 2 * math.pi / 24
    la, lo = math.radians(lat), math.radians(lon)
    time4 = [math.sin(week), math.cos(week), math.sin(hour), math.cos(hour)]
    latlon4 = [math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)]
    return time4, latlon4


def run_clay(pts, batch=16, zero_latlon=False, chips_dir=None):
    chips_dir = chips_dir or DATA / "chips" / "clay"
    emb, mean, std, waves, gsd = clay_embedder()
    enc = emb.clay_encoder
    ids, cls_out, mean_out = [], [], []
    rows = list(pts.iterrows())
    for i0 in range(0, len(rows), batch):
        chunk = rows[i0:i0 + batch]
        pxs, times, latlons, chunk_ids = [], [], [], []
        for _, row in chunk:
            pid = int(row["id"])
            f = chips_dir / f"{pid}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            px = torch.from_numpy(z["pixels"].astype(np.float32))
            pxs.append((px - mean[:, None, None]) / std[:, None, None])
            t4, l4 = clay_time_latlon(str(z["datetime"]),
                                      float(row["lat"]), float(row["lon"]))
            times.append(t4)
            latlons.append([0.0] * 4 if zero_latlon else l4)
            chunk_ids.append(pid)
        if not chunk_ids:
            continue
        B = len(chunk_ids)
        cube = {
            "pixels": torch.stack(pxs),
            "time": torch.tensor(times, dtype=torch.float32),
            "latlon": torch.tensor(latlons, dtype=torch.float32),
            "waves": waves,
            "gsd": gsd,
        }
        with torch.no_grad():
            patches, _ = enc.to_patch_embed(cube["pixels"], cube["waves"])
            patches = enc.add_encodings(
                patches, cube["time"], cube["latlon"], cube["gsd"])
            cls_t = enc.cls_token.expand(B, -1, -1)
            tokens = enc.transformer(torch.cat((cls_t, patches), dim=1))
        ids.extend(chunk_ids)
        cls_out.extend(tokens[:, 0].numpy())
        mean_out.extend(tokens[:, 1:].mean(1).numpy())
        if len(ids) % 64 < batch:
            print(f"[clay] {len(ids)}", flush=True)
    return ids, np.stack(cls_out), np.stack(mean_out)


# ---------------------------------------------------------------- prithvi
def prithvi_encoder():
    import json
    from prithvi_mae import PrithviViT
    cfg = json.load(open(ROOT / "vendor" / "prithvi" / "config.json"))
    pre = cfg["pretrained_cfg"]
    enc = PrithviViT(
        img_size=224, patch_size=tuple(pre["patch_size"]),
        num_frames=1, in_chans=pre["in_chans"],
        embed_dim=pre["embed_dim"], depth=pre["depth"],
        num_heads=pre["num_heads"], mlp_ratio=pre["mlp_ratio"],
        coords_encoding=pre.get("coords_encoding", []),
    )
    sd = torch.load(ROOT / "checkpoints" / "Prithvi_EO_V2_300M.pt",
                    map_location="cpu", weights_only=True)
    enc_sd = {re.sub(r"^encoder\.", "", k): v for k, v in sd.items()
              if k.startswith("encoder.")
              and k != "encoder.pos_embed"}  # pos_embed is for 4 frames;
    # the deterministic sincos init for 1 frame is kept instead
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    print("[prithvi] missing:", len(missing), "unexpected:", len(unexpected))
    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()
    mean = torch.tensor(pre["mean"], dtype=torch.float32)
    std = torch.tensor(pre["std"], dtype=torch.float32)
    return enc, mean, std


def run_prithvi(pts):
    enc, mean, std = prithvi_encoder()
    ids, cls_out, mean_out = [], [], []
    for _, row in pts.iterrows():
        pid = int(row["id"])
        f = DATA / "chips" / "prithvi" / f"{pid}.npz"
        if not f.exists():
            continue
        px = torch.from_numpy(np.load(f)["pixels"].astype(np.float32))
        px = (px - mean[:, None, None]) / std[:, None, None]
        with torch.no_grad():
            feats = enc.forward_features(px.unsqueeze(0))  # [1, 1+N, 1024]
        x = feats[-1][0]
        ids.append(pid)
        cls_out.append(x[0].numpy())
        mean_out.append(x[1:].mean(0).numpy())
        if len(ids) % 25 == 0:
            print(f"[prithvi] {len(ids)}", flush=True)
    return ids, np.stack(cls_out), np.stack(mean_out)


def main(model):
    RES.mkdir(exist_ok=True)
    pts = pd.read_csv(DATA / "points.csv")
    zero = model.endswith("_noll")
    model = model.replace("_noll", "")
    if model == "clay":
        ids, emb_cls, emb_mean = run_clay(pts, zero_latlon=zero)
        tag = "clay_noll" if zero else "clay"
    else:
        ids, emb_cls, emb_mean = run_prithvi(pts)
        tag = model
    np.savez(RES / f"embeddings_{tag}.npz",
             ids=np.array(ids), emb_cls=emb_cls, emb_mean=emb_mean)
    print(f"[{tag}] saved {emb_cls.shape} embeddings")


if __name__ == "__main__":
    main(sys.argv[1])
