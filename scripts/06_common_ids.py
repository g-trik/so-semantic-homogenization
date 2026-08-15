"""
06_common_ids.py
================
Builds the common subsample: the set of PostIds present in all three corpora.

An answer qualifies only if it yields at least the minimum character count
under both the prose only and the code only extractions, which is equivalent
to appearing in all three corpora. Restricting every analytical cell to this
set means the three content types are three extractions from the same answers,
so any difference between them is attributable to content rather than to
sample composition.

Input : answers_with_tags{,_withcode,_codeonly}.parquet
Output: /root/v3/common_ids.parquet
"""
import pandas as pd

SRC = {'prose':      '/root/answers_with_tags.parquet',
       'prose_code': '/root/answers_with_tags_withcode.parquet',
       'code':       '/root/answers_with_tags_codeonly.parquet'}

sets = {}
for k, p in SRC.items():
    d = pd.read_parquet(p, columns=['PostId', 'HasCode'])
    s = set(d.loc[d.HasCode, 'PostId'])
    sets[k] = s
    print(f'  {k:<12}{len(d):>12,} rows   HasCode=True {len(s):>12,}', flush=True)

common = sets['prose'] & sets['prose_code'] & sets['code']
print(f'\n  intersection : {len(common):,}')
for k, s in sets.items():
    print(f'    lost from {k:<12}{len(s)-len(common):>10,}')

pd.DataFrame({'PostId': sorted(common)}).to_parquet('/root/v3/common_ids.parquet')
print('\n  saved -> /root/v3/common_ids.parquet')
