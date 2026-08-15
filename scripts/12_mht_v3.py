"""
12_mht_v3.py
============
Multiple-hypothesis correction across the 21 primary tests, applied separately
under each moderator specification.

The design comprises 21 tests: H1 in all six cells, H3 and H4 in all six, and
H2 in the three question-level cells. H1 contains no moderator, so its six
estimates are common to both panels; the fifteen moderator tests differ between
the reduced and saturated forms.

Reports Benjamini-Hochberg q-values, controlling the false discovery rate, with
monotonicity enforced so q never decreases as the raw p-value rises. The
Bonferroni threshold at a 5% family-wise error rate is also reported.

Correcting across all four hypotheses as a single family is the conservative
choice; correcting within hypothesis families would give a smaller K and a less
stringent threshold.

Input : /root/v3/master_results_v3.csv, written by 10_master_v3.py
Output: /root/v3/mht_both.txt and /root/v3/mht_results_v3.csv

Usage:
  python3 12_mht_v3.py
"""

import numpy as np
import pandas as pd

MASTER = '/root/v3/master_results_v3.csv'
OUT_TXT = '/root/v3/mht_both.txt'
OUT_CSV = '/root/v3/mht_results_v3.csv'
ALPHA = 0.05

LABEL = {'prose_only': 'prose only',
         'prose_code': 'prose & code',
         'code_only':  'code only'}


def benjamini_hochberg(p):
    """BH q-values with monotonicity enforced. Input must be sorted ascending."""
    k = len(p)
    q = p * k / np.arange(1, k + 1)
    return np.minimum.accumulate(q[::-1])[::-1].clip(max=1.0)


def panel(df, title, handle):
    d = df.dropna(subset=['p']).sort_values('p').reset_index(drop=True)
    k = len(d)
    d['rank'] = np.arange(1, k + 1)
    d['q_bh'] = benjamini_hochberg(d['p'].to_numpy())
    d['content'] = d['content'].replace(LABEL)

    bonferroni = ALPHA / k
    lines = [
        '',
        '=' * 66,
        f'{title}   (K = {k})',
        '=' * 66,
        f'{"rank":>5}{"hyp":>5}{"level":>7}{"content":>15}'
        f'{"beta":>12}{"p":>9}{"q_BH":>9}',
        '-' * 62,
    ]
    for _, r in d.iterrows():
        mark = '  *' if r.q_bh < ALPHA else ''
        lines.append(f'{int(r["rank"]):>5}{r.hyp:>5}{r.level:>7}'
                     f'{r.content:>15}{r.b:>+12.5f}{r.p:>9.4f}'
                     f'{r.q_bh:>9.3f}{mark}')
    lines += [
        '',
        f'Bonferroni threshold : {bonferroni:.5f}',
        f'surviving BH at {ALPHA:.0%}   : {(d.q_bh < ALPHA).sum()}',
        f'surviving Bonferroni : {(d.p < bonferroni).sum()}',
    ]

    text = '\n'.join(lines)
    print(text)
    handle.write(text + '\n')
    return d


def main():
    m = pd.read_csv(MASTER)

    h1 = (m.hyp == 'H1') & (m.spec == 'H1')
    mods = m.hyp.isin(['H2', 'H3', 'H4'])

    reduced = m[h1 | (mods & (m.spec == 'REDUCED'))].copy()
    saturated = m[h1 | (mods & (m.spec == 'SATURATED'))].copy()

    with open(OUT_TXT, 'w') as f:
        r = panel(reduced, 'REDUCED specification for moderators', f)
        s = panel(saturated, 'SATURATED specification for moderators', f)

    r['panel'] = 'reduced'
    s['panel'] = 'saturated'
    pd.concat([r, s], ignore_index=True).to_csv(OUT_CSV, index=False)

    print(f'\nSaved -> {OUT_TXT}')
    print(f'Saved -> {OUT_CSV}')


if __name__ == '__main__':
    main()
