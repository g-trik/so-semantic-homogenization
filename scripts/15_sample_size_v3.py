"""
15_sample_size_v3.py
====================
Sample-size sensitivity of the H1 tag-level treatment effect.

The tag-level design caps each (tag, month) cluster at 100 answers. This script
re-estimates H1 at caps of 25, 50, 75 and 100 to establish that the chosen cap
does not drive the result.

At caps below 100 the cluster is subsampled and the similarity metric and
covariates are recomputed from the embeddings. At the full cap no subsampling is
required, so the stored panel is used directly and the row is therefore
identical to the main estimate rather than an independent draw.

Output: /root/v3/sample_size_v3.csv

Usage:
  python3 15_sample_size_v3.py
"""

import warnings

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")

OUT = '/root/v3/sample_size_v3.csv'
CAPS = [25, 50, 75, 100]
FULL_CAP = 100
MIN_CLUSTER = 5
SEED = 42
MEDIAN_TENURE = 5.12          # sample median, set in 09_metrics_v3.py
MAX_TENURE_YRS = 25
POST_FROM = pd.Timestamp('2022-11-01')

SOURCE = {
    'prose_only': ('/root/v3/emb_tag_prose_v3.npy',
                   '/root/v3/meta_tag_prose_v3.parquet',
                   '/root/v3/metrics_tag_prose_v3.parquet'),
    'prose_code': ('/root/v3/emb_tag_prosecode_v3.npy',
                   '/root/v3/meta_tag_prosecode_v3.parquet',
                   '/root/v3/metrics_tag_prosecode_v3.parquet'),
    'code_only':  ('/root/v3/emb_tag_code_v3.npy',
                   '/root/v3/meta_tag_code_v3.parquet',
                   '/root/v3/metrics_tag_code_v3.parquet'),
}

CONTROLS = ['LogAvgBodyLength', 'ShareBelowMedianTenure']


def stars(p):
    return ("***" if p < .001 else "**" if p < .01 else
            "*" if p < .05 else "." if p < .10 else "")


def add_did_terms(d):
    """Post, Treated and their interaction from Month and Group."""
    d['Month_dt'] = pd.to_datetime(d['Month'] + '-01')
    d['Post'] = (d['Month_dt'] >= POST_FROM).astype(int)
    d['Treated'] = (d['Group'] == 'treatment').astype(int)
    d['PxT'] = d['Post'] * d['Treated']
    return d


def tenure_at_posting(meta):
    """Years between account creation and the month of posting."""
    post_dt = pd.to_datetime(meta['CreationDate'].astype(str) + '-01',
                             errors='coerce')
    acct_dt = pd.to_datetime(meta['AccountCreated'], errors='coerce')
    ten = (post_dt - acct_dt).dt.days / 365.25
    return ten.where(ten.between(0, MAX_TENURE_YRS))


def build_panel(emb, meta, cap, rng):
    """Subsample each cluster to `cap`, then recompute metrics and covariates."""
    rows = []
    for (tag, month), g in meta.groupby(['Tag', 'CreationDate'], sort=False):
        if len(g) > cap:
            g = g.sample(n=cap, random_state=rng)
        if len(g) < MIN_CLUSTER:
            continue
        v = emb[g.index.to_numpy()]
        upper = (v @ v.T)[np.triu_indices(len(v), k=1)]
        ten = g['TenureYrs'].dropna()
        rows.append({
            'Tag': tag,
            'Month': month,
            'Group': g['Group'].iloc[0],
            'AvgCosineSim': float(upper.mean()),
            'AvgBodyLength': float(g['BodyLength'].mean()),
            'ShareBelowMedianTenure': (float((ten < MEDIAN_TENURE).mean())
                                       if len(ten) else np.nan),
        })

    d = pd.DataFrame(rows)
    d['LogAvgBodyLength'] = np.log(d['AvgBodyLength'] + 1)
    return add_did_terms(d)


def estimate(d):
    rhs = ['PxT'] + CONTROLS
    dd = d.dropna(subset=rhs + ['AvgCosineSim'])
    r = pf.feols(f"AvgCosineSim ~ {' + '.join(rhs)} | Tag + Month_dt",
                 data=dd, vcov={"CRV1": "Tag"}).tidy().loc['PxT']
    return float(r['Estimate']), float(r['Pr(>|t|)']), len(dd)


def main():
    print(f"{'cap':>5}  " + "".join(f"{c:<24}" for c in SOURCE))
    print('-' * 80)

    rows = []
    n_obs = None
    for cap in CAPS:
        line = f"{cap:>5}  "
        for content, (emb_path, meta_path, panel_path) in SOURCE.items():
            if cap >= FULL_CAP:
                d = add_did_terms(pd.read_parquet(panel_path).copy())
            else:
                emb = np.load(emb_path)
                meta = pd.read_parquet(meta_path).reset_index(drop=True)
                meta['TenureYrs'] = tenure_at_posting(meta)
                d = build_panel(emb, meta, cap, np.random.RandomState(SEED))

            b, p, n_obs = estimate(d)
            line += f"{b:+.4f} ({p:.3f}){stars(p):<4}   "
            rows.append(dict(cap=cap, content=content, b=b, p=p, N=n_obs))
        print(line, flush=True)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nN per regression: {n_obs} cluster-months")
    print(f"Saved -> {OUT}")


if __name__ == '__main__':
    main()
