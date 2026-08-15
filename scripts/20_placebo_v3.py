"""
20_placebo_v3.py
================
Tag-level placebo test.

Re-estimates the H1 specification on the pre-period subsample (January 2019 to
October 2022) with November 2020 as a fake treatment date, two years before
ChatGPT's actual release. A non-significant coefficient supports the reading
that the main estimates reflect the post-November 2022 shock rather than
pre-existing differences between the groups.

The question-level placebo requires rebuilding the panels from the embeddings
under the fake period split, and is run by 11_robustness_v3.py.

Output: /root/v3/placebo_tag_v3.csv

Usage:
  python3 20_placebo_v3.py
"""

import warnings

import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")

OUT = '/root/v3/placebo_tag_v3.csv'
PLACEBO_DATE = pd.Timestamp('2020-11-01')
ACTUAL_DATE = pd.Timestamp('2022-11-01')
CONTROLS = ['LogAvgBodyLength', 'ShareBelowMedianTenure']

CELLS = [('Tag prose-only',   '/root/v3/metrics_tag_prose_v3.parquet'),
         ('Tag prose & code', '/root/v3/metrics_tag_prosecode_v3.parquet'),
         ('Tag code-only',    '/root/v3/metrics_tag_code_v3.parquet')]


def stars(p):
    return ("***" if p < .001 else "**" if p < .01 else
            "*" if p < .05 else "." if p < .10 else "")


def main():
    print(f"{'Cell':<20}{'Coefficient':>13}{'Std. error':>12}"
          f"{'p-value':>10}{'N':>8}")
    print('-' * 65)

    rows = []
    for label, path in CELLS:
        d = pd.read_parquet(path).copy()
        d['Month_dt'] = pd.to_datetime(d['Month'] + '-01')
        d = d[d['Month_dt'] < ACTUAL_DATE].copy()

        d['Treated'] = (d['Group'] == 'treatment').astype(int)
        d['PlaceboPost'] = (d['Month_dt'] >= PLACEBO_DATE).astype(int)
        d['PxT'] = d['PlaceboPost'] * d['Treated']

        rhs = ['PxT'] + CONTROLS
        dd = d.dropna(subset=rhs + ['AvgCosineSim'])
        r = pf.feols(f"AvgCosineSim ~ {' + '.join(rhs)} | Tag + Month_dt",
                     data=dd, vcov={"CRV1": "Tag"}).tidy().loc['PxT']

        b, se = float(r['Estimate']), float(r['Std. Error'])
        p = float(r['Pr(>|t|)'])
        print(f"{label:<20}{b:>+13.4f}{se:>12.4f}{p:>10.3f}"
              f"{len(dd):>8,}  {stars(p)}")
        rows.append(dict(cell=label, b=b, se=se, p=p, N=len(dd)))

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")


if __name__ == '__main__':
    main()
