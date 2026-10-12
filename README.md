# geoseek

Prototype: **frozen Earth-observation foundation-model embeddings for a
presence-vs-background task**, comparing two open EO foundation models —
[Clay v1.5](https://huggingface.co/made-with-clay/Clay) and
[Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) —
against a conventional bioclimatic baseline.

Built as a GeoAI support deliverable: end-to-end STAC data access,
model-specific preprocessing, embedding extraction, lightweight probing
and evaluation under **spatially blocked cross-validation**, all on CPU.

**Live demo:** [geodemo.ecoseek.org](https://geodemo.ecoseek.org) —
click the map to explore landscape similarity in Clay's embedding
space, with a suitability layer scored by the frozen-embedding probe.

## Task

Species: *Rhipicephalus microplus* (cattle tick) — the published
case study of the [`xnicher`](https://github.com/alrobles/xnicher) R
package (JOSS submission; Robles Fernández & Jiménez).

- **334 presence records** + **334 background points** sampled inside the
  accessible area *M* bundled with `xnicher::rhipicephalus`
- Features per point:
  - `clay_cls` / `clay_mean` — Clay v1.5 CLS / mean-pooled embedding
    (1024-d), Sentinel-2 L2A 10-band chip, 256×256 px @ 10 m
    (2.56 km window)
  - `clay_noll_mean` — same Clay embedding with the lat/lon tokens
    zeroed (geolocation-ablation control)
  - `prithvi_cls` / `prithvi_mean` — Prithvi-EO-2.0 CLS / mean-pooled
    tokens (1024-d), HLS-S30 6-band chip, 224×224 px @ 30 m (6.72 km window)
  - `bioclim` — bio1 + relative humidity, the two covariates of the
    published case study

## Pipeline

| step | script | what it does |
|---|---|---|
| 1 | `scripts/01_fetch_chips.py` | lowest-cloud item per point via STAC — Planetary Computer `sentinel-2-l2a` (Clay) and `hls2-s30` (Prithvi) — centred chips cached to `.npz`; chips >90 % zeros are retried with the next-best item |
| 2 | `scripts/02_embed.py` | frozen-encoder inference, no fine-tuning: Clay `Embedder` (official `claymodel` path) and `PrithviViT.forward_features` |
| 3 | `scripts/03_evaluate.py` | logistic-regression probe under 5 spatial blocks (k-means on coordinates); ROC-AUC + AP |
| 4 | `scripts/04_figures.py` | study-area map, AUC comparison, embedding PCA |

Data window: one lowest-cloud scene per point within
2024-01-01 → 2025-07-01, `eo:cloud_cover < 40`.

## Results

n = 667 points (334 presences / 333 background; one Guianas background
point lacked HLS coverage), logistic-regression probe, 5 spatially
blocked folds:

| feature set | ROC-AUC (logreg) | ROC-AUC (RF) |
|---|---|---|
| **clay_mean** | **0.853 ± 0.054** | **0.869 ± 0.043** |
| clay + bioclim | 0.851 ± 0.053 | 0.868 ± 0.043 |
| clay_noll_mean (lat/lon zeroed) | 0.852 ± 0.059 | 0.866 ± 0.048 |
| clay_cls | 0.852 ± 0.049 | 0.862 ± 0.053 |
| prithvi_mean | 0.818 ± 0.044 | 0.835 ± 0.030 |
| prithvi_mean + bioclim | 0.816 ± 0.047 | 0.831 ± 0.032 |
| prithvi_cls | 0.785 ± 0.088 | 0.830 ± 0.015 |
| lonlat only | 0.639 ± 0.078 | 0.563 ± 0.069 |
| bioclim (bio1 + rh) | 0.502 ± 0.098 | 0.579 ± 0.076 |

`results/metrics.csv` · `figures/auc_comparison.png` ·
`figures/study_area.png` · `figures/embedding_pca.png` ·
`figures/example_chips.png`

**Reading.** Clay v1.5's frozen embeddings carry the strongest signal
for this task (0.85–0.87 AUC under spatially blocked CV, consistent
across probes and pooling choices), with Prithvi-EO-2.0 close behind
(0.79–0.84); both sit far above the two-covariate bioclim baseline
(0.50–0.58). The `clay_noll` control — same chips, lat/lon tokens
zeroed — loses at most ~0.004 AUC, while a probe on raw coordinates
alone reaches only 0.64: Clay's score comes from the imagery, not
from its geolocation tokens. This is a single-probe comparison on one
ecological task — evidence about *embedding readiness for this
workflow*, not a general model ranking.

## Honest scope notes

- **Presence-vs-background ≠ presence-absence.** This evaluates whether
  the frozen embeddings carry *predictive signal* about locations where
  the species is recorded — it is an embedding-quality probe, not a
  validated species distribution model.
- The two models see different footprints by design (2.56 km @ 10 m vs
  6.72 km @ 30 m): the comparison is "each model used per its documented
  input spec", not pixel-identical inputs.
- Presence records are occurrence aggregates without observation dates;
  imagery is a fixed 2024–2025 window. Treating records as
  contemporary-enough is a documented prototype simplification.
- Clay additionally encodes explicit lat/lon + acquisition-time tokens
  (`time`, `latlon` in its datacube); Prithvi-EO-2.0 has
  `coords_encoding: []` and sees imagery only. The `clay_noll` ablation
  (lat/lon zeroed) keeps 0.87 AUC against 0.64 for a raw-coordinates
  probe, so geolocation does not explain Clay's score.
- HLS-S30 had no item for one background point in the Guianas inside
  the date window (`no_item`, id 635).
- Checkpoints are git-ignored. Clay encoder weights
  (`checkpoints/clay_v15_encoder.pt`) were extracted from the public
  `v1.5/clay-v1.5.ckpt` on Hugging Face; Prithvi weights are the public
  `Prithvi_EO_V2_300M.pt`. The vendored `vendor/prithvi/prithvi_mae.py`
  is the model implementation shipped inside the official HF repo.

## Reproduce

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install "git+https://github.com/Clay-foundation/model.git"

# points/env are exported from the xnicher package (see scripts/00_*.R)
python scripts/01_fetch_chips.py clay
python scripts/01_fetch_chips.py prithvi
# download checkpoints per "Honest scope notes"
python scripts/02_embed.py clay && python scripts/02_embed.py clay_noll
python scripts/02_embed.py prithvi
python scripts/03_evaluate.py && python scripts/04_figures.py
```

Everything runs CPU-only; the embedding step is the bottleneck
(~1-2 h for 668 points on 8 cores).

## Why this design

The intended pattern for EO foundation models in a research-support
context is exactly this: **frozen embeddings + a lightweight probe**,
because it is cheap, reproducible on modest hardware, and isolates
what the representation already knows. Fine-tuning a 300 M-parameter
MAE for a 668-point task would be the wrong tool — and would still not
outperform an honest spatial-block evaluation of embeddings plus a
two-covariate baseline.
