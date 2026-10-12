"""Spatial-block CV evaluation of frozen EO-FM embeddings for the
presence-vs-background task on the xnicher Rhipicephalus microplus case.

Feature sets:
  clay_cls       - Clay v1.5 CLS embedding (1024-d)
  prithvi_cls    - Prithvi-EO-2.0-300M CLS token (1024-d)
  prithvi_mean   - Prithvi mean-pooled patch tokens (1024-d)
  bioclim        - bio1 + rh (the 2 covariates of the published case study)
  clay+bioclim, prithvi_mean+bioclim - concatenated features

Folds: 5 spatial blocks via k-means on (lon, lat) - each fold a
spatially contiguous hold-out region (no random pixel leakage).

Probe: StandardScaler + LogisticRegression (linear probe).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
N_FOLDS = 5
SEED = 20261011


def load_features():
    pts = pd.read_csv(DATA / "points.csv").set_index("id")
    env = pd.read_csv(DATA / "env.csv").set_index("id")[["bio1", "rh"]]
    out = {}
    for model, keys in [("clay", ["emb_cls", "emb_mean"]),
                        ("prithvi", ["emb_cls", "emb_mean"]),
                        ("clay_noll", ["emb_mean"])]:
        f = RES / f"embeddings_{model}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        for k in keys:
            out[f"{model}_{k.split('_')[1]}"] = \
                pd.DataFrame(z[k], index=z["ids"]) \
                  .join(pts[["label", "lon", "lat"]]).dropna()
    bioclim = env.join(pts[["label", "lon", "lat"]]).dropna()
    return out, bioclim


def spatial_folds(coords, n=N_FOLDS):
    km = KMeans(n_clusters=n, n_init=20, random_state=SEED).fit(coords)
    return km.labels_


PROBES = {
    "logreg": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
    "rf": lambda: RandomForestClassifier(
        300, random_state=0, n_jobs=8),
}


def evaluate(X, y, folds, probe="logreg"):
    aucs, aps = [], []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if te.sum() < 10 or len(np.unique(y[te])) < 2:
            continue
        clf = PROBES[probe]()
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p))
    return np.array(aucs), np.array(aps)


def main():
    feats, bioclim = load_features()

    # align ids present in every embedding set
    common = bioclim.index
    for df in feats.values():
        common = common.intersection(df.index)
    pts = pd.read_csv(DATA / "points.csv").set_index("id").loc[common]
    folds = spatial_folds(pts[["lon", "lat"]].values)
    y = pts["label"].values

    drop = ["label", "lon", "lat"]
    sets = {
        "bioclim (bio1+rh)": bioclim.loc[common, ["bio1", "rh"]].values,
        "lonlat only": pts[["lon", "lat"]].values,
    }
    for k, df in feats.items():
        sets[k] = df.loc[common].drop(columns=drop).values
    sets["clay+bioclim"] = np.hstack(
        [sets["clay_mean"], sets["bioclim (bio1+rh)"]])
    sets["prithvi_mean+bioclim"] = np.hstack(
        [sets["prithvi_mean"], sets["bioclim (bio1+rh)"]])

    rows = []
    for name, X in sets.items():
        for probe in PROBES:
            aucs, aps = evaluate(X, y, folds, probe)
            rows.append(dict(feature_set=name, probe=probe,
                             n_features=X.shape[1],
                             auc_mean=aucs.mean(), auc_sd=aucs.std(),
                             ap_mean=aps.mean(), ap_sd=aps.std(),
                             folds=len(aucs)))
            print(f"{name:24s} {probe:7s} "
                  f"AUC {aucs.mean():.3f}±{aucs.std():.3f}  "
                  f"AP {aps.mean():.3f}±{aps.std():.3f}")
    res = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
    res.to_csv(RES / "metrics.csv", index=False)
    np.savez(RES / "fold_assignments.npz",
             ids=common.values, folds=folds)
    print("\n", res.to_string(index=False))
    print(f"\nn = {len(common)} points ({int(y.sum())} pres / "
          f"{int((1 - y).sum())} bg), {N_FOLDS} spatial folds")


if __name__ == "__main__":
    main()
