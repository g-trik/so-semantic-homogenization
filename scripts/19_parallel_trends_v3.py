"""
19_parallel_trends_v3.py
========================
Parallel-trends and pre-period balance tests, under two contributor controls.

At tag level, where clusters are observed across multiple pre-period months,
the coefficient on TimeIndex x Treated tests for a differential linear trend.
Tag fixed effects are included but month fixed effects are not: they would
absorb the time index and leave the interaction unidentified.

At question level each question appears in exactly two periods, so a
within-question pre-trend cannot be estimated. Pre-period level balance is
assessed instead by a cross-sectional regression of baseline similarity on the
treatment indicator, with robust (HC1) standard errors and no fixed effects.

Both tests are run twice, once with the tenure-based contributor control used
in the main specifications and once with the share of answerers holding
reputation below 100, so the result can be read against either measure.

Output: /root/v3/parallel_trends_v3.csv

Usage:
  python3 19_parallel_trends_v3.py
"""

import warnings

import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

OUT = '/root/v3/parallel_trends_v3.csv'
PRE_END = pd.Timestamp('2022-11-01')
BASE_YEAR = 2019

CONTROLS = [('tenure', 'ShareBelowMedianTenure'),
            ('novice', 'ShareNewUsers')]

TAG = [('Tag prose only',   '/root/v3/metrics_tag_prose_v3.parquet'),
       ('Tag prose & code', '/root/v3/metrics_tag_prosecode_v3.parquet'),
       ('Tag code only',    '/root/v3/metrics_tag_code_v3.parquet')]

QL = [('QL prose only',   '/root/v3/metrics_ql_prose_v3.parquet'),
      ('QL prose & code', '/root/v3/metrics_ql_prosecode_v3.parquet'),
      ('QL code only',    '/root/v3/metrics_ql_code_v3.parquet')]


def stars(p):
    return ("***" if p < .001 else "**" if p < .01 else
            "*" if p < .05 else "." if p < .10 else "")


def tag_slope(d, control):
    """Coefficient on TimeIndex x Treated over the pre-period."""
    rhs = ['TimeIndex', 'TxTreated', 'LogAvgBodyLength', control]
    dd = d.dropna(subset=rhs + ['AvgCosineSim'])
    r = pf.feols(f"AvgCosineSim ~ {' + '.join(rhs)} | Tag",
                 data=dd, vcov={"CRV1": "Tag"}).tidy().loc['TxTreated']
    return float(r['Estimate']), float(r['Pr(>|t|)']), len(dd)


def ql_balance(d, control):
    """Coefficient on Treated in the pre-period cross-section."""
    cols = ['Treated', 'LogNAnswers', 'LogAvgBodyLength', control]
    dd = d.dropna(subset=cols + ['AvgCosineSim'])
    m = sm.OLS(dd['AvgCosineSim'],
               sm.add_constant(dd[cols])).fit(cov_type='HC1')
    return float(m.params['Treated']), float(m.pvalues['Treated']), len(dd)


def run(label, test_name, d, estimator, fmt):
    """Estimate under both contributor controls and print one row."""
    out = {}
    for key, control in CONTROLS:
        if control not in d.columns:
            out[key] = (np.nan, np.nan, 0)
            continue
        out[key] = estimator(d, control)

    t, n = out['tenure'], out['novice']
    print(f"{label:<20}{test_name:<22}{t[0]:>+13.6f}{t[1]:>9.{fmt}f}"
          f"{stars(t[1]):<3}{n[0]:>+14.6f}{n[1]:>9.{fmt}f}{stars(n[1])}")

    return dict(cell=label, test=test_name,
                b_tenure=t[0], p_tenure=t[1],
                b_novice=n[0], p_novice=n[1], N=t[2])


def main():
    print("PARALLEL TRENDS AND PRE-PERIOD BALANCE")
    print("=" * 88)
    print(f"{'Cell':<20}{'Test':<22}{'tenure coef':>13}{'p':>9}"
          f"{'novice coef':>14}{'p':>9}")
    print('-' * 88)

    rows = []

    for label, path in TAG:
        d = pd.read_parquet(path).copy()
        d['Month_dt'] = pd.to_datetime(d['Month'] + '-01')
        d = d[d['Month_dt'] < PRE_END].copy()
        d['Treated'] = (d['Group'] == 'treatment').astype(int)
        d['TimeIndex'] = ((d['Month_dt'].dt.year - BASE_YEAR) * 12
                          + d['Month_dt'].dt.month - 1)
        d['TxTreated'] = d['TimeIndex'] * d['Treated']
        rows.append(run(label, 'Pre-trend slope', d, tag_slope, 3))

    print()

    for label, path in QL:
        d = pd.read_parquet(path).copy()
        d = d[d['Post'] == 0].copy()
        d['Treated'] = (d['Group'] == 'treatment').astype(int)
        rows.append(run(label, 'Pre-period balance', d, ql_balance, 4))

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")


if __name__ == '__main__':
    main()
