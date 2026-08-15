"""
Minimum-cluster-size sensitivity: H1 re-estimated at tag-level thresholds of
3, 5, 10, 20, 30, 50. No resampling — clusters below each threshold are simply
dropped from the existing panel.
"""
import warnings, numpy as np, pandas as pd, pyfixest as pf
warnings.filterwarnings("ignore")

CELLS = {'prose_only':'/root/v3/metrics_tag_prose_v3.parquet',
         'prose_code':'/root/v3/metrics_tag_prosecode_v3.parquet',
         'code_only': '/root/v3/metrics_tag_code_v3.parquet'}
THRESHOLDS = [3, 5, 10, 20, 30, 50]

def st(p):
    return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "." if p<.10 else ""

panels = {}
for ct, path in CELLS.items():
    d = pd.read_parquet(path).copy()
    d['Month_dt'] = pd.to_datetime(d.Month + '-01')
    d['Post'] = (d.Month_dt >= pd.Timestamp('2022-11-01')).astype(int)
    d['Treated'] = (d.Group == 'treatment').astype(int)
    d['PxT'] = d.Post * d.Treated
    panels[ct] = d

print("minimum observed cluster size per cell:")
for ct, d in panels.items():
    print(f"  {ct:<12}{int(d.NAnswers.min())}")
print()

print(f"{'min':>5}  " + "".join(f"{c:<24}" for c in CELLS) + "clusters")
print('-'*95)
rows = []
for thr in THRESHOLDS:
    line = f"{thr:>5}  "
    n_last = None
    for ct, d in panels.items():
        dd = d[d.NAnswers >= thr]
        rhs = ['PxT','LogAvgBodyLength','ShareBelowMedianTenure']
        dd = dd.dropna(subset=rhs+['AvgCosineSim'])
        r = pf.feols(f"AvgCosineSim ~ {' + '.join(rhs)} | Tag + Month_dt",
                     data=dd, vcov={"CRV1":"Tag"}).tidy().loc['PxT']
        b, p = float(r['Estimate']), float(r['Pr(>|t|)'])
        line += f"{b:+.4f} ({p:.3f}){st(p):<4}   "
        n_last = len(dd)
        rows.append(dict(threshold=thr, content=ct, b=b, p=p, clusters=len(dd)))
    print(line + f"{n_last}", flush=True)

pd.DataFrame(rows).to_csv('/root/v3/min_cluster_v3.csv', index=False)
print("\nSaved -> /root/v3/min_cluster_v3.csv")
