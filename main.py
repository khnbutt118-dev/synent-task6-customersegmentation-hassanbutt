"""Synent Data Science Internship - Task 6: Customer Segmentation (Mall Customers, K-Means)
Author: Hassan Butt
Run: python main.py
"""

import sys, os, re, json, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


DATASETS = {'mall': {'files': ['Mall_Customers.csv'],
          'kaggle': 'vjchoudhary7/customer-segmentation-tutorial-in-python',
          'name': 'Mall Customer Segmentation dataset',
          'page': 'https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python'}}



def _outdir(repo_dir):
    p = Path(repo_dir) / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, repo_dir, name):
    fig.savefig(_outdir(repo_dir) / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def save_json(obj, repo_dir, name):
    (_outdir(repo_dir) / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _read_csv(path, spec):
    kw = {"encoding": spec.get("encoding", "utf-8")}
    if spec.get("no_header_file") and Path(path).name == spec["no_header_file"]:
        kw["header"] = None
        kw["names"] = spec["names"]
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        kw["encoding"] = "latin-1"
        return pd.read_csv(path, **kw)


def _builtin(key):
    if key == "iris":
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d.frame.drop(columns=["target"])
        df["species"] = [d.target_names[i] for i in d.target]
        return df
    raise KeyError(key)


def get_data(key, repo_dir):
    """Load a dataset: local data/ folder -> built-in -> kagglehub -> public URL."""
    spec = DATASETS[key]
    data_dir = Path(repo_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in spec["files"]:
        p = data_dir / name
        if p.exists():
            print(f"  [data] using local file {p}")
            return _read_csv(p, spec)
    if spec.get("builtin"):
        print("  [data] using built-in copy")
        return _builtin(key)
    if spec.get("kaggle"):
        try:
            import kagglehub
            folder = Path(kagglehub.dataset_download(spec["kaggle"]))
            cands = list(folder.rglob("*.csv"))
            pick = None
            for want in spec["files"]:
                for c in cands:
                    if c.name.lower() == want.lower():
                        pick = c
                        break
                if pick:
                    break
            if pick is None and cands:
                pick = cands[0]
            if pick is not None:
                dest = data_dir / pick.name
                shutil.copy(pick, dest)
                print(f"  [data] downloaded from Kaggle: {dest}")
                return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] kagglehub download not available ({type(e).__name__}: {e})")
    if spec.get("url"):
        try:
            import urllib.request
            dest = data_dir / (spec.get("no_header_file") or spec["files"][0])
            urllib.request.urlretrieve(spec["url"], dest)
            print(f"  [data] downloaded from public URL: {dest}")
            return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] public URL download failed ({type(e).__name__}: {e})")
    raise FileNotFoundError(
        f"Dataset '{spec['name']}' not found.\n"
        f"  -> Download it from {spec['page']}\n"
        f"  -> Save the CSV as one of {spec['files']} inside: {data_dir.resolve()}\n"
        f"  -> Then run the script again.")


def _to_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


def _snake(col):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()


def task6_run(repo_dir):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    df = get_data("mall", repo_dir)
    ren = {}
    for c in df.columns:
        lc = str(c).lower()
        if "income" in lc:
            ren[c] = "annual_income"
        elif "spending" in lc:
            ren[c] = "spending_score"
        elif lc in ("genre", "gender"):
            ren[c] = "gender"
        elif lc == "age":
            ren[c] = "age"
        elif "id" in lc:
            ren[c] = "customer_id"
    df = df.rename(columns=ren).drop_duplicates()
    df = df.dropna(subset=["annual_income", "spending_score", "age"])
    shape = [int(df.shape[0]), int(df.shape[1])]
    X = df[["annual_income", "spending_score"]]
    Xs = StandardScaler().fit_transform(X)

    ks = range(2, 11)
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Xs, km.labels_))
    best_k = list(ks)[int(np.argmax(sil))]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(list(ks), inertia, marker="o")
    ax[0].set_title("Elbow method (inertia)")
    ax[0].set_xlabel("k")
    ax[1].plot(list(ks), sil, marker="o", color="#27ae60")
    ax[1].axvline(best_k, ls="--", color="grey")
    ax[1].set_title(f"Silhouette score (best k = {best_k})")
    ax[1].set_xlabel("k")
    save_fig(fig, repo_dir, "elbow_and_silhouette.png")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xs)
    df["cluster"] = km.labels_
    sil_final = float(silhouette_score(Xs, km.labels_))
    prof = df.groupby("cluster").agg(customers=("cluster", "size"), avg_age=("age", "mean"),
                                     avg_income=("annual_income", "mean"), avg_spending=("spending_score", "mean")).round(1)
    mi, si = df["annual_income"].mean(), df["annual_income"].std()
    ms, ss = df["spending_score"].mean(), df["spending_score"].std()
    names = {}
    for cl, r in prof.iterrows():
        zi, zs = (r["avg_income"] - mi) / si, (r["avg_spending"] - ms) / ss
        if abs(zi) < 0.5 and abs(zs) < 0.5:
            names[cl] = "Average customers"
        elif zi >= 0 and zs >= 0:
            names[cl] = "High income, high spending (premium)"
        elif zi >= 0 and zs < 0:
            names[cl] = "High income, low spending (careful)"
        elif zi < 0 and zs >= 0:
            names[cl] = "Low income, high spending (impulsive)"
        else:
            names[cl] = "Low income, low spending (budget)"
    prof["segment"] = pd.Series(names)
    df["segment"] = df["cluster"].map(names)
    prof.to_csv(_outdir(repo_dir) / "segment_profiles.csv")
    df.to_csv(_outdir(repo_dir) / "customers_with_segments.csv", index=False)

    cent = pd.DataFrame(StandardScaler().fit(X).inverse_transform(km.cluster_centers_), columns=["annual_income", "spending_score"])
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.scatterplot(data=df, x="annual_income", y="spending_score", hue="segment", palette="tab10", s=70, ax=ax)
    ax.scatter(cent["annual_income"], cent["spending_score"], c="black", marker="X", s=220, label="centroids")
    ax.set_title(f"K-Means customer segments (k = {best_k})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save_fig(fig, repo_dir, "customer_segments.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=df, x="cluster", y="age", ax=ax[0], palette="tab10")
    ax[0].set_title("Age distribution per cluster")
    prof["customers"].plot(kind="bar", ax=ax[1], color="#34495e")
    ax[1].set_title("Customers per cluster")
    ax[1].tick_params(axis="x", rotation=0)
    save_fig(fig, repo_dir, "cluster_age_and_size.png")

    biggest = prof["customers"].idxmax()
    target = [c for c, n in names.items() if n.startswith("High income, high")]
    insights = [
        f"The silhouette score is highest at k = {best_k} ({sil_final:.2f}), so {best_k} segments were used.",
        f"The largest segment is '{names[biggest]}' with {int(prof.loc[biggest, 'customers'])} customers.",
    ]
    if target:
        t = target[0]
        insights.append(f"The premium segment (high income, high spending) has {int(prof.loc[t, 'customers'])} customers with an average age of {prof.loc[t, 'avg_age']:.0f}, a priority for loyalty offers.")
    low_high = [c for c, n in names.items() if n.startswith("High income, low")]
    if low_high:
        t = low_high[0]
        insights.append(f"The 'high income, low spending' segment ({int(prof.loc[t, 'customers'])} customers) is a candidate for targeted campaigns, since these customers have the means to spend more.")
    return {"headline": {"Customers": shape[0], "Clusters (k)": best_k, "Silhouette score": round(sil_final, 2),
                         "Largest segment": names[biggest]},
            "insights": insights, "dataset_shape": shape,
            "files": ["elbow_and_silhouette.png", "customer_segments.png", "cluster_age_and_size.png",
                      "segment_profiles.csv", "customers_with_segments.csv"]}



if __name__ == "__main__":
    res = task6_run(Path(__file__).parent)
    print(json.dumps(res, indent=2, default=str))
