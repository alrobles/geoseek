"""Train the presence-vs-background probe on the 668 labelled points
and score every node of the demo grid -> per-node suitability values
served by the demo app as a choropleth layer.

Probe: StandardScaler + LogisticRegression on clay_noll emb_mean —
the imagery-only feature set (AUC 0.852 under spatial-block CV), same
protocol as scripts/03_evaluate.py. This is an *embedding-signal*
surface, not a validated species distribution model.

Inputs : results/embeddings_clay_noll.npz (train), data/points.csv,
         demo/data/{dev,grid_embeddings}.npz (score targets)
Output : demo/data/suitability_<tag>.npz  {ids, prob}

Run: python scripts/06_suitability.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, RES, DDATA = (ROOT / "data", ROOT / "results",
                    ROOT / "demo" / "data")


def train():
    z = np.load(RES / "embeddings_clay_noll.npz")
    pts = pd.read_csv(DATA / "points.csv").set_index("id")
    y = pts.loc[z["ids"], "label"].values
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, C=1.0))
    clf.fit(z["emb_mean"], y)
    # (in-sample fit is not informative at 1024-d on 668 pts — the CV
    # estimate lives in results/metrics.csv)
    return clf


def score(clf, npz_path):
    z = np.load(npz_path)
    if "emb_mean" not in z:
        return
    prob = clf.predict_proba(z["emb_mean"])[:, 1].astype(np.float32)
    out = DDATA / f"suitability_{npz_path.stem.split('_')[0]}.npz"
    np.savez(out, ids=z["ids"].astype(np.int64), prob=prob)
    print(f"[suit] {npz_path.name}: n={len(prob)} "
          f"p=[{prob.min():.3f},{prob.mean():.3f},{prob.max():.3f}] -> {out}")


def main():
    clf = train()
    for f in sorted(DDATA.glob("*.npz")):
        if f.stem.startswith("suitability"):
            continue
        score(clf, f)


if __name__ == "__main__":
    main()
