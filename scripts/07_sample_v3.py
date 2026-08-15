"""
07_sample_v3.py
===============
Tag classification and panel construction for all six analytical cells,
restricted to the common subsample.

Classification rule: a question qualifies only if its tag list contains exactly
one treatment tag and no control tag, or exactly one control tag and no
treatment tag. Questions straddling both groups are dropped, so no question
contributes to both arms (SUTVA).

Answers are restricted to the PostId set present in all three corpora, built by
06_common_ids.py, so every cell contains identical answers and prose/code
contrasts vary content only, not sample.

At tag level the draw of up to N_PER_CLUSTER answers per (tag, month) cluster
is made once and reused across the three content types, so the three cells hold
exactly the same answers. At question level no cap is applied, but a question
enters the panel only if it has at least MIN_PER_PERIOD answers in both the
pre- and post-period.

Usage:
  python3 07_sample_v3.py
  python3 07_sample_v3.py --level ql --content code
"""

import argparse
import numpy as np
import pandas as pd

TREATMENT = ['python', 'c++', 'java', 'javascript', 'typescript', 'php', 'ruby']
CONTROL   = ['bash', 'r']

N_PER_CLUSTER  = 100
MIN_PER_PERIOD = 2
POST_FROM      = '2022-11'
SEED           = 42

COMMON_IDS = '/root/v3/common_ids.parquet'

CORPORA = {'prose':      '/root/answers_with_tags.parquet',
           'prose_code': '/root/answers_with_tags_withcode.parquet',
           'code':       '/root/answers_with_tags_codeonly.parquet'}

OUT_TAG = {'prose':      '/root/v3/answers_sampled_v3.parquet',
           'prose_code': '/root/v3/answers_sampled_withcode_v3.parquet',
           'code':       '/root/v3/answers_sampled_codeonly_v3.parquet'}

OUT_QL  = {'prose':      '/root/v3/answers_ql_v3.parquet',
           'prose_code': '/root/v3/answers_ql_withcode_v3.parquet',
           'code':       '/root/v3/answers_ql_codeonly_v3.parquet'}

BASE_COLS = ['PostId', 'ParentId', 'OwnerUserId', 'CreationDate', 'Score',
             'BodyLength', 'AcceptedFlag', 'Reputation', 'HasOwner',
             'AccountCreated', 'HasCode', 'Tags']

# shared tag-level draw, populated by the first tag cell processed
_SHARED_TAG_IDS = None


class TagSampler:
    """Classifies questions by tag and builds one analytical panel."""

    def __init__(self, content, level, common_ids=None,
                 treatment=TREATMENT, control=CONTROL,
                 n_per_cluster=N_PER_CLUSTER, min_per_period=MIN_PER_PERIOD,
                 post_from=POST_FROM, seed=SEED):
        if content not in CORPORA:
            raise ValueError(f'content must be one of {list(CORPORA)}')
        if level not in ('tag', 'ql'):
            raise ValueError("level must be 'tag' or 'ql'")
        self.content, self.level = content, level
        self.src = CORPORA[content]
        self.out = (OUT_TAG if level == 'tag' else OUT_QL)[content]
        self.common = common_ids
        self.treatment_set, self.control_set = set(treatment), set(control)
        self.n_per_cluster = n_per_cluster
        self.min_per_period = min_per_period
        self.post_from = post_from
        self.seed = seed

    # ---------------- classification ----------------
    def classify(self, tag_list):
        """Exactly one treatment tag XOR exactly one control tag."""
        t_count = c_count = 0
        t_match = c_match = None
        for t in tag_list:
            if t in self.treatment_set:
                t_count += 1; t_match = t
            elif t in self.control_set:
                c_count += 1; c_match = t
            if t_count > 1 or c_count > 1:
                return None, None
        if t_count == 1 and c_count == 0:
            return t_match, 'treatment'
        if c_count == 1 and t_count == 0:
            return c_match, 'control'
        return None, None

    # ---------------- loading ----------------
    def _load(self):
        cols = list(BASE_COLS)
        if self.content == 'code':
            cols.append('CodeFromInlineOnly')
        df = pd.read_parquet(self.src, columns=cols)
        n0 = len(df)
        print(f'  answers loaded   : {n0:,}', flush=True)
        if self.common is not None:
            df = df[df['PostId'].isin(self.common)].copy()
            print(f'  common subsample : {len(df):,} '
                  f'({100*len(df)/n0:.1f}% of {n0:,})', flush=True)
        return df

    def _assign_groups(self, df):
        q = df[['ParentId', 'Tags']].drop_duplicates(subset='ParentId')
        print(f'  unique questions : {len(q):,}', flush=True)
        res = [self.classify(tl) for tl in q['Tags'].values]
        q = q.assign(Tag=[r[0] for r in res], Group=[r[1] for r in res])
        qual = q[q['Tag'].notna()][['ParentId', 'Tag', 'Group']]
        print(f'  qualifying       : {len(qual):,} '
              f'({100*len(qual)/len(q):.1f}% of questions)', flush=True)
        df = df.merge(qual, on='ParentId', how='inner')
        print(f'  answers retained : {len(df):,}', flush=True)
        return df

    # ---------------- panels ----------------
    def _build_tag_panel(self, df):
        global _SHARED_TAG_IDS
        df = df.copy()
        df['Month'] = df['CreationDate'].astype(str).str[:7]

        if _SHARED_TAG_IDS is None:
            rng = np.random.RandomState(self.seed)
            parts = []
            for _, g in df.groupby(['Tag', 'Month'], sort=False):
                parts.append(g if len(g) <= self.n_per_cluster
                             else g.sample(n=self.n_per_cluster, random_state=rng))
            out = pd.concat(parts, ignore_index=True)
            _SHARED_TAG_IDS = set(out['PostId'])
            print(f'  drew shared tag sample: {len(out):,} answers', flush=True)
        else:
            out = df[df['PostId'].isin(_SHARED_TAG_IDS)].copy()
            print(f'  reused shared tag sample: {len(out):,} answers', flush=True)

        print(f'  clusters         : '
              f'{out.groupby(["Tag","Month"]).ngroups:,}', flush=True)
        return out

    def _build_ql_panel(self, df):
        df = df.copy()
        df['Post'] = (df['CreationDate'].astype(str) >= self.post_from).astype(int)
        pre  = df.loc[df['Post'] == 0].groupby('ParentId').size()
        post = df.loc[df['Post'] == 1].groupby('ParentId').size()
        valid = (set(pre[pre >= self.min_per_period].index)
                 & set(post[post >= self.min_per_period].index))
        print(f'  panel-valid qs   : {len(valid):,}', flush=True)
        out = df[df['ParentId'].isin(valid)].copy()
        print(f'  answers retained : {len(out):,}', flush=True)
        return out

    def _attach_body(self, out):
        want = set(out['PostId'])
        body = pd.read_parquet(self.src, columns=['PostId', 'Body'])
        body = body[body['PostId'].isin(want)]
        out = out.merge(body, on='PostId', how='left')
        missing = out['Body'].isna().sum()
        if missing:
            print(f'  WARNING missing body: {missing:,}', flush=True)
        return out

    # ---------------- run ----------------
    def run(self):
        print('=' * 72)
        print(f'SAMPLE — {self.level} level, {self.content}')
        print('=' * 72)
        df = self._load()
        df = self._assign_groups(df)
        out = (self._build_tag_panel(df) if self.level == 'tag'
               else self._build_ql_panel(df))
        del df
        out = self._attach_body(out)
        out.to_parquet(self.out)
        self._report(out)

    def _report(self, out):
        print(f'  saved            : {self.out}')
        if self.level == 'ql':
            t = out.loc[out.Group == 'treatment', 'ParentId'].nunique()
            c = out.loc[out.Group == 'control',   'ParentId'].nunique()
            print(f'  treatment qs     : {t:,}')
            print(f'  control qs       : {c:,}')
            print(f'  imbalance        : {t/c:.1f}:1' if c else '  control empty')
            by = out.groupby(['Group', 'Tag'])['ParentId'].nunique()
        else:
            by = out.groupby(['Group', 'Tag'])['PostId'].count()
        print(); print(by.to_string()); print('=' * 72); print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--level',   choices=['tag','ql','all'], default='all')
    ap.add_argument('--content', choices=['prose','prose_code','code','all'],
                    default='all')
    a = ap.parse_args()

    common = set(pd.read_parquet(COMMON_IDS)['PostId'])
    print(f'common PostId set: {len(common):,}\n', flush=True)

    levels   = ['tag','ql'] if a.level == 'all' else [a.level]
    contents = list(CORPORA) if a.content == 'all' else [a.content]
    for lv in levels:
        for ct in contents:
            TagSampler(ct, lv, common_ids=common).run()
