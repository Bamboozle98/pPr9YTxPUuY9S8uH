# src/analysis/cluster_populations.py
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA


from src.data.LoadData import load_data


def _stack_X(X_train, X_test):
    # handles sparse and dense
    if sp.issparse(X_train) or sp.issparse(X_test):
        return sp.vstack([X_train, X_test])
    return np.vstack([X_train, X_test])


def _stack_y(y_train, y_test):
    if y_train is None or y_test is None:
        return None
    return np.concatenate([np.asarray(y_train), np.asarray(y_test)])


def choose_k_by_silhouette(Z, k_min=2, k_max=10, random_state=72, n_init=20):
    best = {"k": None, "silhouette": -np.inf}
    rows = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(Z)
        sil = silhouette_score(Z, labels)
        inertia = km.inertia_

        rows.append({"k": k, "silhouette": sil, "inertia": inertia})
        if sil > best["silhouette"]:
            best = {"k": k, "silhouette": sil}

    scores = pd.DataFrame(rows).sort_values("k")
    return best["k"], scores


def top_features_by_cluster(X_scaled, labels, feature_names, top_n=15):
    """
    X_scaled: scaled feature matrix (sparse or dense) in ORIGINAL feature space
    labels: cluster labels for each row in X_scaled
    Returns dict cluster_id -> DataFrame(feature, mean)
    """
    out = {}
    n_clusters = int(labels.max()) + 1

    # For efficiency: compute per-cluster mean without densifying huge matrices.
    # Sparse mean over rows: slice cluster rows, then mean(axis=0)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        Xc = X_scaled[idx]
        mu = np.asarray(Xc.mean(axis=0)).ravel()  # works for sparse/dense
        df = pd.DataFrame({"feature": feature_names, "mean": mu}).sort_values("mean", ascending=False)
        out[c] = df.head(top_n).reset_index(drop=True)

    return out


def plot_clusters_2d(X_all, clusters, out_png="clusters_2d.png", random_state=72, y_all=None):
    """
    X_all: full feature matrix (sparse or dense)
    clusters: cluster labels for each row
    y_all: optional true labels (0/1) for a second plot
    """
    # 2D projection that is safe for sparse matrices
    if sp.issparse(X_all):
        proj = TruncatedSVD(n_components=2, random_state=random_state)
    else:
        proj = PCA(n_components=2, random_state=random_state)

    X_2d = proj.fit_transform(X_all)

    # Plot clusters
    plt.figure()
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, s=10, alpha=0.8)
    plt.title("2D Projection of Samples (colored by cluster)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"Saved cluster plot -> {out_png}")

    # Optional: visualize true labels (if you want to compare structure vs target)
    if y_all is not None:
        out_png2 = out_png.replace(".png", "_labels.png")
        plt.figure()
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_all, s=10, alpha=0.8)
        plt.title("2D Projection of Samples (colored by label)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        plt.savefig(out_png2, dpi=200)
        plt.close()
        print(f"Saved label plot -> {out_png2}")


def main():
    ap = argparse.ArgumentParser(description="Cluster dataset populations (standalone).")
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=10)
    ap.add_argument("--pca_components", type=int, default=10,
                    help="Number of PCA components. Set to 0 to disable PCA.")
    ap.add_argument("--random_state", type=int, default=72)
    ap.add_argument("--n_init", type=int, default=20)
    ap.add_argument("--top_n_features", type=int, default=15)
    ap.add_argument("--out_csv", type=str, default="cluster_assignments.csv")
    args = ap.parse_args()

    # Your existing loader
    X_train, X_test, y_train, y_test, encoder = load_data()
    feature_names = encoder.get_feature_names_out().tolist()

    X_all = _stack_X(X_train, X_test)
    y_all = _stack_y(y_train, y_test)

    # Scale: with_mean=False is required for sparse
    scaler = StandardScaler(with_mean=False)

    if args.pca_components and args.pca_components > 0:
        reducer = PCA(n_components=args.pca_components, random_state=args.random_state)
        pre = Pipeline([("scaler", scaler), ("pca", reducer)])
    else:
        pre = Pipeline([("scaler", scaler)])

    # Transform once for k-selection
    Z = pre.fit_transform(X_all)

    best_k, scores = choose_k_by_silhouette(
        Z,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
        n_init=args.n_init,
    )

    print("\n=== K Selection (Silhouette + Inertia) ===")
    print(scores.to_string(index=False))
    print(f"\nChosen k (best silhouette): {best_k}")

    # Fit final clustering model
    kmeans = KMeans(n_clusters=best_k, n_init=args.n_init, random_state=args.random_state)
    clusters = kmeans.fit_predict(Z)
    # Visualize using ORIGINAL feature space (scaled is also fine, but keep it simple)
    # If you want scaled instead: pass scaler.fit_transform(X_all)
    plot_clusters_2d(X_all, clusters, out_png="clusters_2d.png", random_state=args.random_state, y_all=y_all)

    # Save assignments (train/test row order is preserved)
    n_train = X_train.shape[0]
    split = np.array(["train"] * n_train + ["test"] * (X_all.shape[0] - n_train))

    out = pd.DataFrame({
        "row_index": np.arange(X_all.shape[0]),
        "split": split,
        "cluster": clusters,
    })

    if y_all is not None:
        out["label"] = y_all

    out.to_csv(args.out_csv, index=False)
    print(f"\nSaved cluster assignments -> {args.out_csv}")

    # Cluster sizes
    print("\n=== Cluster Counts ===")
    print(out["cluster"].value_counts().sort_index().to_string())

    # Cluster vs label distribution (if labels exist)
    if y_all is not None:
        print("\n=== Cluster vs Label (row-normalized) ===")
        ct = pd.crosstab(out["cluster"], out["label"], normalize="index")
        print(ct.to_string())

    # Feature profiles per cluster (in original feature space)
    # We profile using SCALED features (before PCA), which is easiest to interpret.
    X_scaled = scaler.fit_transform(X_all)

    profiles = top_features_by_cluster(
        X_scaled=X_scaled,
        labels=clusters,
        feature_names=feature_names,
        top_n=args.top_n_features,
    )

    print("\n=== Top Feature Profiles Per Cluster (scaled mean) ===")
    for c, df in profiles.items():
        print(f"\n--- Cluster {c} ---")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
