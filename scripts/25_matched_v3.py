"""
25_matched_v3.py
================
Sensitivity of the question-level estimates to the treatment-control ratio.

The full sample carries 8,407 treatment questions against 415 control, a ratio
of 20.3:1. This script re-estimates H1 and the reputation moderator on matched
subsamples at ratios of 3:1, 5:1, 8:1 and 12:1, holding the control group fixed
and selecting treatment questions nearest to it.

Matching is nearest-neighbour without replacement on standardised pre-period
LogNAnswers and LogAvgBodyLength: for each control question the closest
unselected treatment questions are taken in turn until the target count is
reached.

Alongside each estimate the script reports the correlation between Post x M and
Post x Treated x M and the variance inflation factor on the triple interaction,
so the collinearity can be read against the ratio directly.

Output: /root/v3/matched_v3.csv

Usage:
  python3 25_matched_v3.py
"""

import warnings

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

OUT = '/root/v3/matched_v3.csv'
MODERATOR = 'LogAvgReputation'
RATIOS = [3, 5, 8, 12, None]          # None = full sample
MATCH_ON = ['LogNAnswers', 'LogAvgBodyLength']

CELLS = {'prose only':   '/root/v3/metrics_ql_prose_v3.parquet',
         'prose & code': '/root/v3/metrics_ql_prosecode_v3.parquet',
         'code only':    '/root/v3/metrics_ql_code_v3.parquet'}


def fit(d, rhs, term):
    dd = d.dropna(subset=rhs + ['AvgCosineSim'])
    r = pf.feols(f"AvgCosineSim ~ {' + '.join(rhs)} | ParentId + Post",
                 data=dd, vcov={"CRV1": "ParentId"}).tidy().loc[term]
    return float(r['Estimate']), float(r['Pr(>|t|)'])


def vif(df, cols, target):
    """Variance inflation factor via auxiliary regression."""
    y = df[target].to_numpy(float)
    others = [c for c in cols if c != target]
    x = np.c_[np.ones(len(df)), df[others].to_numpy(float)]
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    ss_res = ((y - x @ b) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return 1 / (1 - r2) if r2 < 1 else np.inf


def nearest_treatment(control_z, treatment_z, treatment_ids, n_wanted):
    """Nearest treatment questions to the control set, without replacement."""
    tree = cKDTree(treatment_z)
    k = min(max(1, n_wanted // len(control_z) + 4), len(treatment_z))
    _, idx = tree.query(control_z, k=k)
    idx = np.atleast_2d(idx)

    taken, order = set(), []
    for col in range(idx.shape[1]):
        for i in idx[:, col]:
            if i not in taken:
                taken.add(i)
                order.append(i)
            if len(taken) >= n_wanted:
                break
        if len(taken) >= n_wanted:
            break
    return treatment_ids[np.array(order[:n_wanted])]


def main():
    print(f'{"content":<14}{"ratio":>6}{"treat":>8}{"ctrl":>6}{"corr":>8}'
          f'{"VIF":>7}{"H1 b":>9}{"p":>7}{"H4red":>9}{"p":>7}'
          f'{"H4sat":>9}{"p":>7}')
    print('-' * 97)

    rows = []
    for name, path in CELLS.items():
        full = pd.read_parquet(path).copy()
        full['Treated'] = (full.Group == 'treatment').astype(int)

        pre = full[full.Post == 0].set_index('ParentId')[MATCH_ON]
        ctrl_ids = full.loc[full.Group == 'control', 'ParentId'].unique()
        trt_ids = full.loc[full.Group == 'treatment', 'ParentId'].unique()

        mu, sd = pre.mean(), pre.std()
        z_ctrl = ((pre.loc[ctrl_ids] - mu) / sd).to_numpy()
        z_trt = ((pre.loc[trt_ids] - mu) / sd).to_numpy()

        for ratio in RATIOS:
            if ratio is None:
                d, label, keep = full.copy(), 'full', trt_ids
            else:
                n_wanted = ratio * len(ctrl_ids)
                keep = nearest_treatment(z_ctrl, z_trt, trt_ids, n_wanted)
                d = full[full.ParentId.isin(set(keep))
                         | (full.Group == 'control')].copy()
                label = f'{ratio}:1'

            m = MODERATOR
            d['PxT'] = d.Post * d.Treated
            d['PxM'] = d.Post * d[m]
            d['TxM'] = d.Treated * d[m]
            d['PxTxM'] = d.Post * d.Treated * d[m]

            h1 = fit(d, ['PxT', 'LogNAnswers', 'LogAvgBodyLength',
                         'ShareBelowMedianTenure'], 'PxT')
            red = fit(d, [m, 'PxT', 'PxTxM', 'LogNAnswers',
                          'LogAvgBodyLength'], 'PxTxM')
            sat_cols = [m, 'PxT', 'PxM', 'TxM', 'PxTxM',
                        'LogNAnswers', 'LogAvgBodyLength']
            sat = fit(d, sat_cols, 'PxTxM')

            ds = d.dropna(subset=sat_cols + ['AvgCosineSim'])
            corr = np.corrcoef(ds.PxM, ds.PxTxM)[0, 1]
            v = vif(ds, sat_cols, 'PxTxM')

            print(f'{name:<14}{label:>6}{len(keep):>8,}{len(ctrl_ids):>6}'
                  f'{corr:>8.3f}{v:>7.1f}{h1[0]:>+9.4f}{h1[1]:>7.3f}'
                  f'{red[0]:>+9.4f}{red[1]:>7.3f}{sat[0]:>+9.4f}{sat[1]:>7.3f}')

            rows.append(dict(content=name, ratio=label, treat=len(keep),
                             ctrl=len(ctrl_ids), corr=corr, vif=v,
                             h1_b=h1[0], h1_p=h1[1],
                             h4_red_b=red[0], h4_red_p=red[1],
                             h4_sat_b=sat[0], h4_sat_p=sat[1]))
        print()

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'Saved -> {OUT}')


if __name__ == '__main__':
    main()
