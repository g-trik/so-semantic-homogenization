"""
11_robustness_v3.py
===================
Four question-level checks, with tag-level rows re-run alongside so a single
estimator is used throughout.

  A. Placebo test        - November 2020 as a fake treatment date, question
                           level; panels rebuilt from the embeddings
  B. Pre-period balance  - cross-sectional OLS on pre-period clusters, HC1
  C. Median metric       - median pairwise cosine similarity as the dependent
                           variable, all six cells
  D. Bash-only control   - R excluded from the control group

The placebo block uses tenure at time of posting for the contributor control,
matching the main specifications. Reputation is not used here: the dump records
it only as a snapshot at its release date, so it is correlated with time since
posting and therefore with the pre/post distinction the estimate is identified
from.

Output: /root/v3/robustness_v3.csv

Usage:
  python3 11_robustness_v3.py
"""

import warnings

import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

OUT = "/root/v3/robustness_v3.csv"
PLACEBO = pd.Timestamp("2020-11-01")
ACTUAL = pd.Timestamp("2022-11-01")
MIN_PER_PERIOD = 2
MEDIAN_TENURE = 5.12          # sample median, set in 09_metrics_v3.py

TAG = {"prose_only": "/root/v3/metrics_tag_prose_v3.parquet",
       "prose_code": "/root/v3/metrics_tag_prosecode_v3.parquet",
       "code_only":  "/root/v3/metrics_tag_code_v3.parquet"}

QL = {"prose_only": "/root/v3/metrics_ql_prose_v3.parquet",
      "prose_code": "/root/v3/metrics_ql_prosecode_v3.parquet",
      "code_only":  "/root/v3/metrics_ql_code_v3.parquet"}

QL_META = {
    "prose_only": ("/root/v3/meta_ql_prose_v3_tenure.parquet",
                   "/root/v3/emb_ql_prose_v3.npy"),
    "prose_code": ("/root/v3/meta_ql_prosecode_v3_tenure.parquet",
                   "/root/v3/emb_ql_prosecode_v3.npy"),
    "code_only":  ("/root/v3/meta_ql_code_v3_tenure.parquet",
                   "/root/v3/emb_ql_code_v3.npy"),
}


def stars(p):
    if not np.isfinite(p):
        return ""
    return ("***" if p < .001 else "**" if p < .01 else
            "*" if p < .05 else "." if p < .10 else "")


def h1_fit(df, level, dv="AvgCosineSim"):
    """H1 specification on a stored panel."""
    df = df.copy()
    if "Group" in df.columns:
        df["Treated"] = (df["Group"] == "treatment").astype(int)
    if level == "tag":
        df["Month_dt"] = pd.to_datetime(df["Month"] + "-01")
        df["Post"] = (df["Month_dt"] >= ACTUAL).astype(int)
        fe, clus = "Tag + Month_dt", "Tag"
        ctrls = ["LogAvgBodyLength", "ShareBelowMedianTenure"]
    else:
        fe, clus = "ParentId + Post", "ParentId"
        ctrls = ["LogNAnswers", "LogAvgBodyLength", "ShareBelowMedianTenure"]
    df["PxT"] = df["Post"] * df["Treated"]
    res = pf.feols(f"{dv} ~ {' + '.join(['PxT'] + ctrls)} | {fe}",
                   data=df, vcov={"CRV1": clus})
    t = res.tidy().loc["PxT"]
    return (float(t["Estimate"]), float(t["Std. Error"]),
            float(t["Pr(>|t|)"]), int(res._N))


rows = []

# ---------------------------------------------------------------- A. placebo
print("=" * 96)
print("A. PLACEBO TEST (fake treatment November 2020, question level)")
print("=" * 96)

for ct, (meta_path, emb_path) in QL_META.items():
    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    meta["emb_idx"] = meta.index
    meta["CreationDate"] = pd.to_datetime(meta["CreationDate"])

    pre = meta[meta["CreationDate"] < ACTUAL].copy()
    pre["FakePost"] = (pre["CreationDate"] >= PLACEBO).astype(int)

    a = pre[pre.FakePost == 0].groupby("ParentId").size()
    b = pre[pre.FakePost == 1].groupby("ParentId").size()
    valid = (set(a[a >= MIN_PER_PERIOD].index)
             & set(b[b >= MIN_PER_PERIOD].index))
    pre = pre[pre.ParentId.isin(valid)].copy()

    emb = np.load(emb_path, mmap_mode="r")
    recs = []
    for (pid, fp), g in pre.groupby(["ParentId", "FakePost"]):
        if len(g) < MIN_PER_PERIOD:
            continue
        v = np.asarray(emb[g.emb_idx.values], dtype=float)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n < 1e-8] = 1.0
        v = v / n
        s = v @ v.T
        ten = g["TenureYrs"].dropna()
        recs.append(dict(
            ParentId=pid,
            Post=fp,
            AvgCosineSim=s[np.triu_indices(len(v), k=1)].mean(),
            LogNAnswers=np.log(len(g)),
            LogAvgBodyLength=np.log(g["BodyLength"].mean() + 1),
            ShareBelowMedianTenure=(float((ten < MEDIAN_TENURE).mean())
                                    if len(ten) else np.nan),
            Treated=int(g["Group"].iloc[0] == "treatment")))

    d = pd.DataFrame(recs)
    d["PxT"] = d["Post"] * d["Treated"]
    rhs = ["PxT", "LogNAnswers", "LogAvgBodyLength", "ShareBelowMedianTenure"]
    dd = d.dropna(subset=rhs + ["AvgCosineSim"])

    r = pf.feols(f"AvgCosineSim ~ {' + '.join(rhs)} | ParentId + Post",
                 data=dd, vcov={"CRV1": "ParentId"})
    t = r.tidy().loc["PxT"]
    bb, se, p, n = (float(t["Estimate"]), float(t["Std. Error"]),
                    float(t["Pr(>|t|)"]), int(r._N))
    print(f"  QL {ct:<12} b={bb:+.5f}{stars(p):<4} se={se:.5f} p={p:.4f} "
          f"N={n:,}  ({'PASS' if p > 0.05 else 'FLAG'})")
    rows.append(dict(check="placebo", level="ql", content=ct,
                     b=bb, se=se, p=p, N=n))

# -------------------------------------------------------- B. level balance
print()
print("=" * 96)
print("B. PRE-PERIOD LEVEL BALANCE (cross-sectional OLS, HC1)")
print("=" * 96)

for ct, path in QL.items():
    d = pd.read_parquet(path)
    d = d[d.Post == 0].copy()
    d["Treated"] = (d["Group"] == "treatment").astype(int)
    cols = ["Treated", "LogNAnswers", "LogAvgBodyLength",
            "ShareBelowMedianTenure"]
    n0 = len(d)
    d = d.dropna(subset=cols + ["AvgCosineSim"]).copy()
    if len(d) < n0:
        print(f"    (dropped {n0 - len(d)} rows with missing covariates)")

    m = sm.OLS(d["AvgCosineSim"], sm.add_constant(d[cols])).fit(cov_type="HC1")
    bb, se = m.params["Treated"], m.bse["Treated"]
    p, n = m.pvalues["Treated"], int(m.nobs)
    print(f"  QL {ct:<12} b={bb:+.5f}{stars(p):<4} se={se:.5f} p={p:.4f} "
          f"N={n:,}  ({'balanced' if p > 0.05 else 'IMBALANCE'})")
    rows.append(dict(check="level_balance", level="ql", content=ct,
                     b=bb, se=se, p=p, N=n))

# ----------------------------------------------------------- C. median metric
print()
print("=" * 96)
print("C. MEDIAN PAIRWISE COSINE SIMILARITY")
print("=" * 96)

for level, paths in (("tag", TAG), ("ql", QL)):
    for ct, path in paths.items():
        d = pd.read_parquet(path)
        if "MedianCosineSim" not in d.columns:
            print(f"  {level} {ct}: MedianCosineSim missing")
            continue
        bb, se, p, n = h1_fit(d, level, dv="MedianCosineSim")
        print(f"  {level:<4}{ct:<12} b={bb:+.5f}{stars(p):<4} se={se:.5f} "
              f"p={p:.4f} N={n:,}")
        rows.append(dict(check="median", level=level, content=ct,
                         b=bb, se=se, p=p, N=n, dv="MedianCosineSim"))

# -------------------------------------------------------------- D. bash only
print()
print("=" * 96)
print("D. BASH-ONLY CONTROL (R excluded)")
print("=" * 96)

for level, paths in (("tag", TAG), ("ql", QL)):
    for ct, path in paths.items():
        d = pd.read_parquet(path)
        db = d[(d.Group == "treatment")
               | ((d.Group == "control") & (d.Tag == "bash"))].copy()
        bm, _, pm, _ = h1_fit(d, level)
        bb, se, p, n = h1_fit(db, level)
        print(f"  {level:<4}{ct:<12} main={bm:+.5f}{stars(pm):<4}"
              f"   bash-only={bb:+.5f}{stars(p):<4} p={p:.4f} N={n:,}")
        rows.append(dict(check="bash_only", level=level, content=ct,
                         b=bb, se=se, p=p, N=n, b_main=bm, p_main=pm))

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nSaved -> {OUT}")
