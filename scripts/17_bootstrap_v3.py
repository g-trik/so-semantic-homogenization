"""
Wild cluster bootstrap for the H1 tag-level treatment effect.
Webb six-point weights, null imposed, B = 9,999, clustered by tag.

With nine clusters (seven treatment, two control) cluster-robust standard
errors over-reject; the bootstrap provides small-sample-valid inference
(Cameron, Gelbach & Miller, 2008).
"""
import warnings, numpy as np, pandas as pd, pyfixest as pf
warnings.filterwarnings("ignore")

CELLS = [
 ('Tag prose only',   '/root/v3/metrics_tag_prose_v3.parquet'),
 ('Tag prose & code', '/root/v3/metrics_tag_prosecode_v3.parquet'),
 ('Tag code only',    '/root/v3/metrics_tag_code_v3.parquet'),
]
B, SEED = 9999, 12345
CTRLS = ['LogAvgBodyLength', 'ShareBelowMedianTenure']

# Webb six-point distribution
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                  np.sqrt(0.5),  1.0,  np.sqrt(1.5)])

def demean(X, y, ent, tim):
    """Two-way within transformation by alternating projection."""
    X, y = X.copy().astype(float), y.copy().astype(float)
    for _ in range(200):
        prev = y.copy()
        for g in (ent, tim):
            df = pd.DataFrame(X); df['_y'] = y; df['_g'] = g
            m = df.groupby('_g').transform('mean')
            X = X - m.iloc[:, :X.shape[1]].to_numpy()
            y = y - m['_y'].to_numpy()
        if np.max(np.abs(y - prev)) < 1e-12:
            break
    return X, y

def cluster_t(X, y, cid):
    """OLS with CRVE, returning t on the first regressor."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(cid):
        m = cid == c
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G = len(np.unique(cid))
    adj = G / (G - 1)
    V = XtX_inv @ meat @ XtX_inv * adj
    return b, b[0] / np.sqrt(V[0, 0]), u

print(f"WILD CLUSTER BOOTSTRAP   Webb 6-pt, null imposed, B={B:,}, seed={SEED}")
print("=" * 84)
print(f"{'Cell':<20}{'beta1':>11}{'CRVE p':>10}{'Boot p':>10}{'clusters':>10}{'N':>8}")
print('-' * 84)

rows = []
for label, path in CELLS:
    d = pd.read_parquet(path).copy()
    d['Month_dt'] = pd.to_datetime(d.Month + '-01')
    d['Post'] = (d.Month_dt >= pd.Timestamp('2022-11-01')).astype(int)
    d['Treated'] = (d.Group == 'treatment').astype(int)
    d['PxT'] = d.Post * d.Treated
    rhs = ['PxT'] + CTRLS
    d = d.dropna(subset=rhs + ['AvgCosineSim']).reset_index(drop=True)

    # CRVE p-value from pyfixest, for consistency with the main tables
    res = pf.feols(f"AvgCosineSim ~ {' + '.join(rhs)} | Tag + Month_dt",
                   data=d, vcov={"CRV1": "Tag"}).tidy().loc['PxT']
    b1, p_crve = float(res['Estimate']), float(res['Pr(>|t|)'])

    # within-transform, then bootstrap on the demeaned system
    X = d[rhs].to_numpy(float)
    y = d['AvgCosineSim'].to_numpy(float)
    Xd, yd = demean(X, y, d['Tag'].to_numpy(), d['Month_dt'].to_numpy())
    cid = pd.factorize(d['Tag'])[0]
    _, t_obs, _ = cluster_t(Xd, yd, cid)

    # restricted model: impose beta1 = 0
    Xr = Xd[:, 1:]
    br = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ yd)
    yhat_r = Xr @ br
    ur = yd - yhat_r

    rng = np.random.default_rng(SEED)
    G = cid.max() + 1
    t_boot = np.empty(B)
    for r in range(B):
        w = rng.choice(WEBB, size=G)[cid]
        y_star = yhat_r + ur * w
        _, t_star, _ = cluster_t(Xd, y_star, cid)
        t_boot[r] = t_star

    p_boot = (np.sum(np.abs(t_boot) >= abs(t_obs)) + 1) / (B + 1)
    print(f"{label:<20}{b1:>+11.4f}{p_crve:>10.3f}{p_boot:>10.3f}{G:>10}{len(d):>8}")
    rows.append(dict(cell=label, beta1=b1, crve_p=p_crve, boot_p=p_boot,
                     t_obs=t_obs, clusters=G, N=len(d)))

pd.DataFrame(rows).to_csv('/root/v3/bootstrap_v3.csv', index=False)
print("\nSaved -> /root/v3/bootstrap_v3.csv")
