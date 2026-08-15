"""
10_master_v3.py
===============
Primary regressions for all four hypotheses across the six analytical cells.

H1 is the main treatment effect, the coefficient on Post x Treated, estimated
without moderator terms. H2, H3 and H4 are moderator hypotheses, each reported
under two specifications:

  REDUCED    omits the two-way interactions Post x M and Treated x M
  SATURATED  includes both, alongside the triple interaction

H4t repeats the reputation moderator with account tenure at time of posting in
place of reputation, which is not subject to the dump-date recording problem.

A final block re-estimates H1 on the code only cells after excluding clusters
in which more than half of answers derive their code solely from inline spans.

Entity and time fixed effects are included throughout, with standard errors
clustered at the entity level: tag for tag-level regressions, question for
question-level regressions.

Output: /root/v3/master_results_v3.csv

Usage:
  python3 10_master_v3.py
"""
import warnings, numpy as np, pandas as pd, pyfixest as pf
warnings.filterwarnings("ignore")

TAG={'prose_only':'/root/v3/metrics_tag_prose_v3.parquet',
     'prose_code':'/root/v3/metrics_tag_prosecode_v3.parquet',
     'code_only':'/root/v3/metrics_tag_code_v3.parquet'}
QL ={'prose_only':'/root/v3/metrics_ql_prose_v3.parquet',
     'prose_code':'/root/v3/metrics_ql_prosecode_v3.parquet',
     'code_only':'/root/v3/metrics_ql_code_v3.parquet'}
TENURE_CTRL = 'ShareBelowMedianTenure'

def st(p):
    if not np.isfinite(p): return ""
    return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "." if p<.10 else ""

def prep(path, level, M=None):
    d=pd.read_parquet(path).copy()
    d["Treated"]=(d["Group"]=="treatment").astype(int)
    if level=="tag":
        d["Month_dt"]=pd.to_datetime(d["Month"]+"-01")
        d["Post"]=(d["Month_dt"]>=pd.Timestamp("2022-11-01")).astype(int)
        fe,clus="Tag + Month_dt","Tag"
    else:
        fe,clus="ParentId + Post","ParentId"
    d["PxT"]=d["Post"]*d["Treated"]
    if M:
        d["PxM"]=d["Post"]*d[M]; d["TxM"]=d["Treated"]*d[M]
        d["PxTxM"]=d["Post"]*d["Treated"]*d[M]
    return d,fe,clus

def fit(d,rhs,fe,clus,term):
    dd=d.dropna(subset=rhs+['AvgCosineSim'])
    r=pf.feols(f"AvgCosineSim ~ {' + '.join(rhs)} | {fe}",
               data=dd, vcov={"CRV1":clus}).tidy()
    if term not in r.index: return None
    q=r.loc[term]
    return float(q["Estimate"]),float(q["Std. Error"]),float(q["Pr(>|t|)"]),len(dd)

rows=[]
print("="*84); print("H1  main treatment effect"); print("="*84)
for lv,paths in (("tag",TAG),("ql",QL)):
    for ct,path in paths.items():
        d,fe,clus=prep(path,lv)
        ctrls=(["LogAvgBodyLength",TENURE_CTRL] if lv=="tag"
               else ["LogNAnswers","LogAvgBodyLength",TENURE_CTRL])
        b,se,p,n=fit(d,["PxT"]+ctrls,fe,clus,"PxT")
        print(f"  {lv:<4}{ct:<12}b={b:+.5f}{st(p):<4}se={se:.5f} p={p:.4f} N={n:,}")
        rows.append(dict(hyp="H1",spec="H1",level=lv,content=ct,b=b,se=se,p=p,N=n))

SPECS=[("H2","ql","LogNAnswers",["LogAvgBodyLength",TENURE_CTRL]),
       ("H3","tag","ShareAccepted",["LogAvgBodyLength",TENURE_CTRL]),
       ("H3","ql","ShareAccepted",["LogNAnswers","LogAvgBodyLength",TENURE_CTRL]),
       ("H4","tag","LogAvgReputation",["LogAvgBodyLength"]),
       ("H4","ql","LogAvgReputation",["LogNAnswers","LogAvgBodyLength"]),
       ("H4t","tag","LogAvgTenure",["LogAvgBodyLength"]),
       ("H4t","ql","LogAvgTenure",["LogNAnswers","LogAvgBodyLength"])]

for hyp,lv,M,ctrls in SPECS:
    paths=TAG if lv=="tag" else QL
    print(); print("="*84); print(f"{hyp} ({lv})  M={M}"); print("="*84)
    for ct,path in paths.items():
        d=pd.read_parquet(path).copy()
        if M=="LogAvgTenure":
            if "AvgTenureYrs" not in d.columns:
                print(f"  {ct:<12}AvgTenureYrs missing"); continue
            d["LogAvgTenure"]=np.log(d["AvgTenureYrs"]+1)
        if M not in d.columns:
            print(f"  {ct:<12}{M} missing"); continue
        d["Treated"]=(d["Group"]=="treatment").astype(int)
        if lv=="tag":
            d["Month_dt"]=pd.to_datetime(d["Month"]+"-01")
            d["Post"]=(d["Month_dt"]>=pd.Timestamp("2022-11-01")).astype(int)
            fe,clus="Tag + Month_dt","Tag"
        else:
            fe,clus="ParentId + Post","ParentId"
        d["PxT"]=d["Post"]*d["Treated"]
        d["PxM"]=d["Post"]*d[M]
        d["TxM"]=d["Treated"]*d[M]
        d["PxTxM"]=d["Post"]*d["Treated"]*d[M]
        for spec,rhs in (("REDUCED",[M,"PxT","PxTxM"]+ctrls),
                         ("SATURATED",[M,"PxT","PxM","TxM","PxTxM"]+ctrls)):
            r=fit(d,rhs,fe,clus,"PxTxM")
            if r is None:
                print(f"  {ct:<12}{spec:<10}not estimable"); continue
            b,se,p,n=r
            print(f"  {ct:<12}{spec:<10}b={b:+.5f}{st(p):<4}p={p:.4f} N={n:,}")
            rows.append(dict(hyp=hyp,spec=spec,level=lv,content=ct,b=b,se=se,p=p,N=n))

# ---- inline-only code sensitivity ----
print(); print("="*84); print("H1 excluding clusters >50% inline-only code"); print("="*84)
for lv,paths in (("tag",TAG),("ql",QL)):
    path=paths['code_only']
    d,fe,clus=prep(path,lv)
    if 'ShareInlineOnlyCode' not in d.columns:
        print(f"  {lv}: flag missing"); continue
    ctrls=(["LogAvgBodyLength",TENURE_CTRL] if lv=="tag"
           else ["LogNAnswers","LogAvgBodyLength",TENURE_CTRL])
    full=fit(d,["PxT"]+ctrls,fe,clus,"PxT")
    sub=fit(d[d.ShareInlineOnlyCode<=0.5],["PxT"]+ctrls,fe,clus,"PxT")
    print(f"  {lv:<4}all clusters   b={full[0]:+.5f}{st(full[2]):<4}p={full[2]:.4f} N={full[3]:,}")
    print(f"  {lv:<4}excl inline>50 b={sub[0]:+.5f}{st(sub[2]):<4}p={sub[2]:.4f} N={sub[3]:,}")
    rows.append(dict(hyp="H1_no_inline",spec="H1",level=lv,content="code_only",
                     b=sub[0],se=sub[1],p=sub[2],N=sub[3]))

pd.DataFrame(rows).to_csv('/root/v3/master_results_v3.csv',index=False)

print("\nSaved -> /root/v3/master_results_v3.csv")
