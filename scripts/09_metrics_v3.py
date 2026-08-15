"""
09_metrics_v3.py
================
Builds the cluster-level panels for all six analytical cells: mean and median
pairwise cosine similarity plus covariates.

Clusters are (tag, month) pairs at tag level and (question, period) pairs at
question level. Similarity is computed over the upper triangle of the pairwise
matrix, so each pair contributes once.

Tenure at time of posting is derived as posting date minus account creation
date. Unlike reputation, which the dump records only as a snapshot at its
release date, tenure at posting is fixed regardless of when the dump was taken:

  AvgTenureYrs            mean cluster tenure in years
  LogAvgTenure            log(AvgTenureYrs + 1)
  ShareBelowMedianTenure  share of answerers below the sample median

The median cutoff is computed once from the first cell processed and reused
across all six, so the threshold is constant.

Reputation is nullable: answers whose author account has been deleted carry
pd.NA. AvgReputation and ShareNewUsers use the known-owner subset as
denominator, so deleted accounts are not misclassified as novice contributors.

Usage:
  python3 09_metrics_v3.py
  python3 09_metrics_v3.py --level ql --content code
"""

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

MIN_TAG_CLUSTER = 5
MIN_QL_PERIOD   = 2
POST_FROM       = '2022-11'
MAX_TENURE_YRS  = 25          # sanity bound on date arithmetic

CELLS = {
 ('tag','prose'):      {'emb':'/root/v3/emb_tag_prose_v3.npy',
                        'meta':'/root/v3/meta_tag_prose_v3.parquet',
                        'out':'/root/v3/metrics_tag_prose_v3.parquet'},
 ('tag','prose_code'): {'emb':'/root/v3/emb_tag_prosecode_v3.npy',
                        'meta':'/root/v3/meta_tag_prosecode_v3.parquet',
                        'out':'/root/v3/metrics_tag_prosecode_v3.parquet'},
 ('tag','code'):       {'emb':'/root/v3/emb_tag_code_v3.npy',
                        'meta':'/root/v3/meta_tag_code_v3.parquet',
                        'out':'/root/v3/metrics_tag_code_v3.parquet'},
 ('ql','prose'):       {'emb':'/root/v3/emb_ql_prose_v3.npy',
                        'meta':'/root/v3/meta_ql_prose_v3.parquet',
                        'out':'/root/v3/metrics_ql_prose_v3.parquet'},
 ('ql','prose_code'):  {'emb':'/root/v3/emb_ql_prosecode_v3.npy',
                        'meta':'/root/v3/meta_ql_prosecode_v3.parquet',
                        'out':'/root/v3/metrics_ql_prosecode_v3.parquet'},
 ('ql','code'):        {'emb':'/root/v3/emb_ql_code_v3.npy',
                        'meta':'/root/v3/meta_ql_code_v3.parquet',
                        'out':'/root/v3/metrics_ql_code_v3.parquet'},
}

# set on the first cell processed, reused thereafter
MEDIAN_TENURE = None


class MetricsBuilder:
    """Builds one cluster-level panel from embeddings plus metadata."""

    def __init__(self, level, content):
        if (level, content) not in CELLS:
            raise ValueError(f'unknown cell {(level, content)}')
        cfg = CELLS[(level, content)]
        self.level, self.content = level, content
        self.emb_path, self.meta_path, self.out = cfg['emb'], cfg['meta'], cfg['out']

    # ---------------- similarity ----------------
    @staticmethod
    def _pairwise(emb):
        """Mean and median pairwise cosine similarity, upper triangle only."""
        sim = emb @ emb.T
        upper = sim[np.triu_indices(len(emb), k=1)]
        return float(upper.mean()), float(np.median(upper))

    # ---------------- covariates ----------------
    @staticmethod
    def _covariates(g):
        rep   = g['Reputation']
        known = g['HasOwner'] if 'HasOwner' in g.columns else rep.notna()
        n_known = int(known.sum())
        rep_known = rep[known].astype('float64')

        out = {'NAnswers':        len(g),
               'ShareAccepted':   float(g['AcceptedFlag'].mean()),
               'AvgBodyLength':   float(g['BodyLength'].mean()),
               'NKnownOwner':     n_known,
               'ShareKnownOwner': n_known / len(g)}

        if n_known:
            out['AvgReputation'] = float(rep_known.mean())
            out['ShareNewUsers'] = float((rep_known < 100).mean())
        else:
            out['AvgReputation'] = np.nan
            out['ShareNewUsers'] = np.nan

        if 'TenureYrs' in g.columns:
            t = g['TenureYrs'].dropna()
            out['AvgTenureYrs'] = float(t.mean()) if len(t) else np.nan
            out['ShareBelowMedianTenure'] = (
                float((t < MEDIAN_TENURE).mean()) if len(t) else np.nan)
        if 'CodeFromInlineOnly' in g.columns:
            out['ShareInlineOnlyCode'] = float(g['CodeFromInlineOnly'].mean())
        return out

    # ---------------- panels ----------------
    def _tag_panel(self, emb, meta):
        rows = []
        for (tag, month), g in tqdm(meta.groupby(['Tag','CreationDate'], sort=False),
                                    desc=f'  tag/{self.content}'):
            if len(g) < MIN_TAG_CLUSTER:
                continue
            m, md = self._pairwise(emb[g.index.to_numpy()])
            rows.append({'Tag': tag, 'Month': month, 'Group': g['Group'].iloc[0],
                         'AvgCosineSim': m, 'MedianCosineSim': md,
                         **self._covariates(g)})
        df = pd.DataFrame(rows)
        df['Post'] = (df['Month'] >= POST_FROM).astype(int)
        return df

    def _ql_panel(self, emb, meta):
        if 'Post' not in meta.columns:
            meta['Post'] = (meta['CreationDate'] >= POST_FROM).astype(int)
        rows = []
        for (qid, period), g in tqdm(meta.groupby(['ParentId','Post'], sort=False),
                                     desc=f'  ql/{self.content}'):
            if len(g) < MIN_QL_PERIOD:
                continue
            m, md = self._pairwise(emb[g.index.to_numpy()])
            rows.append({'ParentId': qid, 'Post': period,
                         'Tag': g['Tag'].iloc[0], 'Group': g['Group'].iloc[0],
                         'AvgCosineSim': m, 'MedianCosineSim': md,
                         **self._covariates(g)})
        df = pd.DataFrame(rows)
        both = df.groupby('ParentId').size()
        return df[df['ParentId'].isin(both[both == 2].index)].copy()

    # ---------------- run ----------------
    def _load_meta(self):
        """Loads metadata and attaches tenure at posting."""
        global MEDIAN_TENURE
        meta = pd.read_parquet(self.meta_path).reset_index(drop=True)
        if 'AccountCreated' in meta.columns:
            post_dt = pd.to_datetime(meta['CreationDate'].astype(str) + '-01',
                                     errors='coerce')
            acct_dt = pd.to_datetime(meta['AccountCreated'], errors='coerce')
            ten = (post_dt - acct_dt).dt.days / 365.25
            ten[~ten.between(0, MAX_TENURE_YRS)] = np.nan
            meta['TenureYrs'] = ten
            if MEDIAN_TENURE is None:
                MEDIAN_TENURE = float(ten.median())
                print(f'  median tenure cutoff: {MEDIAN_TENURE:.2f} years',
                      flush=True)
        return meta

    def run(self):
        print('=' * 72)
        print(f'METRICS — {self.level} level, {self.content}')
        print('=' * 72)

        emb  = np.load(self.emb_path)
        meta = self._load_meta()
        assert len(emb) == len(meta), \
            f'length mismatch: emb {len(emb)} vs meta {len(meta)}'
        print(f'  embeddings       : {emb.shape}')
        print(f'  metadata         : {len(meta):,}', flush=True)

        df = (self._tag_panel(emb, meta) if self.level == 'tag'
              else self._ql_panel(emb, meta))

        df['LogNAnswers']      = np.log(df['NAnswers'])
        df['LogAvgBodyLength'] = np.log(df['AvgBodyLength'] + 1)
        df['LogAvgReputation'] = np.log(df['AvgReputation'] + 1)
        if 'AvgTenureYrs' in df.columns:
            df['LogAvgTenure'] = np.log(df['AvgTenureYrs'] + 1)
        df['Treated'] = (df['Group'] == 'treatment').astype(int)

        df.to_parquet(self.out)
        self._report(df)

    def _report(self, df):
        print(f'  cluster obs      : {len(df):,}')
        if self.level == 'tag':
            print(f'  tags             : {df["Tag"].nunique()}')
            print(f'  months           : {df["Month"].nunique()}')
        else:
            g = df.groupby('ParentId')['Group'].first()
            t, c = (g == 'treatment').sum(), (g == 'control').sum()
            print(f'  questions        : {df["ParentId"].nunique():,}')
            print(f'  treatment/control: {t:,} / {c:,}   ({t/c:.1f}:1)'
                  if c else '  control empty')
        print(f'  mean cos sim     : {df["AvgCosineSim"].mean():.4f}')
        print(f'  median cos sim   : {df["MedianCosineSim"].mean():.4f}')
        print(f'  unknown-owner    : '
              f'{(1 - df["ShareKnownOwner"].mean())*100:.2f}% of answers')
        if 'AvgTenureYrs' in df.columns:
            print(f'  mean tenure      : {df["AvgTenureYrs"].mean():.2f} years')
        if 'ShareInlineOnlyCode' in df.columns:
            print(f'  inline-only code : {df["ShareInlineOnlyCode"].mean()*100:.1f}%')
        print(f'  saved            : {self.out}')
        print('=' * 72); print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--level',   choices=['tag','ql','all'], default='all')
    ap.add_argument('--content', choices=['prose','prose_code','code','all'],
                    default='all')
    a = ap.parse_args()
    levels   = ['tag','ql'] if a.level == 'all' else [a.level]
    contents = ['prose','prose_code','code'] if a.content == 'all' else [a.content]
    for lv in levels:
        for ct in contents:
            MetricsBuilder(lv, ct).run()
