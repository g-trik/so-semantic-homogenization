"""
23_appendix_h_v3.py
===================
Full regression output for every specification reported in the thesis.

Produces 36 tables: for each of the six analytical cells, H1 without moderator
terms, followed by each applicable moderator hypothesis under both the reduced
and the saturated form. H2 is estimated at question level only, since the
tag-level sampling cap leaves no usable variation in cluster answer volume.

Each table reports the coefficient, cluster-robust standard error, t-statistic
and p-value for every regressor, together with the estimation sample size,
overall, adjusted and within R-squared, the fixed effects absorbed and the
clustering level.

Usage:
  python3 23_appendix_h_v3.py
  python3 23_appendix_h_v3.py > appendix_h.txt
"""

import warnings

import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")

POST_FROM = pd.Timestamp('2022-11-01')
TENURE_CTRL = 'ShareBelowMedianTenure'

TAG = {'prose only':   '/root/v3/metrics_tag_prose_v3.parquet',
       'prose & code': '/root/v3/metrics_tag_prosecode_v3.parquet',
       'code only':    '/root/v3/metrics_tag_code_v3.parquet'}

QL = {'prose only':   '/root/v3/metrics_ql_prose_v3.parquet',
      'prose & code': '/root/v3/metrics_ql_prosecode_v3.parquet',
      'code only':    '/root/v3/metrics_ql_code_v3.parquet'}

LABEL = {'PxT':   'Post x Treated',
         'PxM':   'Post x M',
         'TxM':   'Treated x M',
         'PxTxM': 'Post x Treated x M'}


def sig(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def prep(path, level, moderator=None):
    """Load a panel and construct the interaction terms."""
    d = pd.read_parquet(path).copy()
    d['Treated'] = (d['Group'] == 'treatment').astype(int)

    if level == 'tag':
        d['Month_dt'] = pd.to_datetime(d['Month'] + '-01')
        d['Post'] = (d['Month_dt'] >= POST_FROM).astype(int)
        fe, cluster = 'Tag + Month_dt', 'Tag'
    else:
        fe, cluster = 'ParentId + Post', 'ParentId'

    d['PxT'] = d['Post'] * d['Treated']
    if moderator:
        d['PxM'] = d['Post'] * d[moderator]
        d['TxM'] = d['Treated'] * d[moderator]
        d['PxTxM'] = d['Post'] * d['Treated'] * d[moderator]
    return d, fe, cluster


def table(number, title, d, rhs, fe, cluster, moderator=None):
    """Estimate and print one regression table."""
    dd = d.dropna(subset=rhs + ['AvgCosineSim'])
    res = pf.feols(f"AvgCosineSim ~ {' + '.join(rhs)} | {fe}",
                   data=dd, vcov={"CRV1": cluster})
    tidy = res.tidy()

    print(f"\nTable A{number}: {title}")
    print(f"{'Coefficient':<34}{'Estimate':>12}{'Std. Err.':>12}"
          f"{'t-stat':>9}{'p-value':>10}  Sig")

    for term in rhs:
        if term not in tidy.index:
            continue
        r = tidy.loc[term]
        name = LABEL.get(term, term)
        if moderator and term in ('PxM', 'TxM', 'PxTxM'):
            name = name.replace(' M', f' {moderator}')
        p = float(r['Pr(>|t|)'])
        p_str = '<0.001' if p < 0.001 else f'{p:.3f}'
        print(f"{name:<34}{float(r['Estimate']):>+12.6f}"
              f"{float(r['Std. Error']):>12.6f}"
              f"{float(r['t value']):>9.3f}{p_str:>10}  {sig(p)}")

    def stat(attr):
        v = getattr(res, attr, None)
        return float(v) if v is not None else float('nan')

    fe_label = fe.replace('Month_dt', 'Month').replace('ParentId', 'Question')
    print(f"N = {int(res._N):,}. R2 = {stat('_r2'):.4f} "
          f"adj R2 = {stat('_adj_r2'):.4f} "
          f"within R2 = {stat('_r2_within'):.4f}. "
          f"FE: {fe_label}. Cluster: {cluster.replace('ParentId', 'Question')}.")

    return number + 1


def main():
    number = 1

    for level, paths in (('tag', TAG), ('ql', QL)):
        level_name = 'Tag-level' if level == 'tag' else 'QL'
        base = (['LogAvgBodyLength', TENURE_CTRL] if level == 'tag'
                else ['LogNAnswers', 'LogAvgBodyLength', TENURE_CTRL])
        rep_ctrls = (['LogAvgBodyLength'] if level == 'tag'
                     else ['LogNAnswers', 'LogAvgBodyLength'])

        if level == 'tag':
            specs = [('H3', 'ShareAccepted', base),
                     ('H4', 'LogAvgReputation', rep_ctrls)]
        else:
            specs = [('H2', 'LogNAnswers', ['LogAvgBodyLength', TENURE_CTRL]),
                     ('H3', 'ShareAccepted', base),
                     ('H4', 'LogAvgReputation', rep_ctrls)]

        for content, path in paths.items():
            d, fe, cluster = prep(path, level)
            number = table(
                number,
                f"{level_name} H1 ({content}) — Post x Treated, no moderator",
                d, ['PxT'] + base, fe, cluster)

            for hyp, moderator, ctrls in specs:
                dm, fe_m, cluster_m = prep(path, level, moderator)
                forms = (
                    ('reduced',   [moderator, 'PxT', 'PxTxM'] + ctrls),
                    ('saturated', [moderator, 'PxT', 'PxM', 'TxM', 'PxTxM']
                                  + ctrls),
                )
                for form, rhs in forms:
                    number = table(
                        number,
                        f"{level_name} {hyp} ({content}) — "
                        f"moderator {moderator}, {form}",
                        dm, rhs, fe_m, cluster_m, moderator=moderator)

    print("\n\nSignificance codes: *** p < 0.001, ** p < 0.01, * p < 0.05.")


if __name__ == '__main__':
    main()
