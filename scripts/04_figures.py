"""Figures for the geoseek report.

1. study area map      - presences vs background, coloured by spatial fold
2. fold AUC bars       - per-fold + mean ROC-AUC for each feature set
3. embedding PCA       - first two PCs of Clay vs Prithvi embeddings
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)


def map_figure():
    pts = pd.read_csv(DATA / "points.csv")
    folds = np.load(RES / "fold_assignments.npz")
    fmap = dict(zip(folds["ids"], folds["folds"]))
    pts["fold"] = pts["id"].map(fmap)

    fig, ax = plt.subplots(figsize=(8, 7))
    for f, grp in pts[pts.label == 0].groupby("fold"):
        ax.scatter(grp.lon, grp.lat, s=8, alpha=.35, label=f"fold {f}")
    for f, grp in pts[pts.label == 1].groupby("fold"):
        ax.scatter(grp.lon, grp.lat, s=14, marker="^",
                   edgecolors="k", linewidths=.2, alpha=.85)
    pres = pts[pts.label == 1]
    ax.scatter([], [], marker="^", c="k", label="presence")
    ax.scatter([], [], c="gray", alpha=.35, label="background (M)")
    ax.set(xlabel="longitude", ylabel="latitude",
           title="Rhipicephalus microplus presences & accessible-area "
                 "background, coloured by spatial CV fold")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "study_area.png", dpi=160)


def auc_figure():
    met = pd.read_csv(RES / "metrics.csv")
    piv = met.pivot_table(index="feature_set", columns="probe",
                          values=["auc_mean", "auc_sd"])
    order = piv[("auc_mean", "rf")].sort_values().index
    piv = piv.loc[order]
    y = np.arange(len(piv))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for off, probe, col in [(-.2, "logreg", "#4878cf"), (.2, "rf", "#d65f5f")]:
        ax.barh(y + off, piv[("auc_mean", probe)], height=.38,
                xerr=piv[("auc_sd", probe)], color=col, alpha=.85,
                label=probe)
    ax.set_yticks(y, piv.index)
    ax.axvline(.5, color="k", ls="--", lw=.8)
    ax.legend(frameon=False, loc="lower right")
    ax.set(xlabel="spatial-block CV ROC-AUC (mean ± sd)",
           title="Presence-vs-background: frozen EO-FM embeddings vs bioclim")
    fig.tight_layout()
    fig.savefig(FIG / "auc_comparison.png", dpi=160)


def pca_figure():
    pts = pd.read_csv(DATA / "points.csv").set_index("id")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model, key in [(axes[0], "clay", "emb_cls"),
                           (axes[1], "prithvi", "emb_mean")]:
        z = np.load(RES / f"embeddings_{model}.npz")
        y = pts.loc[z["ids"], "label"].values
        X = StandardScaler().fit_transform(z[key])
        pc = PCA(n_components=2, random_state=0).fit_transform(X)
        ax.scatter(pc[y == 0, 0], pc[y == 0, 1], s=7, alpha=.3,
                   label="background")
        ax.scatter(pc[y == 1, 0], pc[y == 1, 1], s=10, alpha=.7,
                   marker="^", label="presence")
        ax.set(title=f"{model} ({key})", xlabel="PC1")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("PC2")
    fig.suptitle("Frozen embeddings, first two PCs (presence vs background)")
    fig.tight_layout()
    fig.savefig(FIG / "embedding_pca.png", dpi=160)


def chips_figure():
    """False-colour (NIR-R-G) previews of what each model sees."""
    pts = pd.read_csv(DATA / "points.csv").set_index("id")
    ids = [2, 200, 400]

    def vis(x):
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1).transpose(1, 2, 0)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6.5))
    for j, pid in enumerate(ids):
        row = pts.loc[pid]
        # clay band_order: blue0 green1 red2 ... nir6 ; prithvi: B02..B07
        c = np.load(DATA / f"chips/clay/{pid}.npz")["pixels"][[6, 2, 1]]
        p = np.load(DATA / f"chips/prithvi/{pid}.npz")["pixels"][[3, 2, 1]]
        axes[0, j].imshow(vis(c.astype(float)))
        axes[0, j].set_title(f"id {pid} S2 2.56 km (clay)", fontsize=9)
        axes[1, j].imshow(vis(p.astype(float)))
        axes[1, j].set_title(f"id {pid} HLS 6.72 km (prithvi)", fontsize=9)
        for ax in (axes[0, j], axes[1, j]):
            ax.axis("off")
    fig.suptitle("False-colour chips (NIR-R-G) at example points")
    fig.tight_layout()
    fig.savefig(FIG / "example_chips.png", dpi=150)


if __name__ == "__main__":
    map_figure()
    auc_figure()
    pca_figure()
    chips_figure()
    print("figures written to", FIG)
