"""
08_embed_v3.py
==============
Generates embeddings for all six analytical cells with OpenAI's
text-embedding-3-large, then removes failed embeddings and reapplies the panel
filter.

A single embedding model is used for every content type, so the prose/code
contrast is identified on content alone rather than on differences between
model vector spaces.

Answer bodies are truncated before submission: 30,000 characters for prose
only, 25,000 for the two code-containing types, reflecting the higher
token-per-character density of code. Batches that fail after three attempts
fall back to per-answer submission; an answer that still fails is written as a
zero vector, removed in the cleaning step, after which the question-level panel
filter is reapplied so that every retained question still has at least
MIN_PER_PERIOD answers in both periods.

Requires OPENAI_API_KEY in the environment.

Usage:
  python3 08_embed_v3.py --estimate-only
  python3 08_embed_v3.py
"""

import argparse, os, time
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

MODEL          = 'text-embedding-3-large'
DIMS           = 3072
BATCH_SIZE     = 100
PRICE_PER_MTOK = 0.13
MIN_PER_PERIOD = 2
TRUNCATION     = {'prose': 30_000, 'prose_code': 25_000, 'code': 25_000}

CELLS = {
 ('tag','prose'):      {'src':'/root/v3/answers_sampled_v3.parquet',
                        'emb':'/root/v3/emb_tag_prose_v3.npy',
                        'meta':'/root/v3/meta_tag_prose_v3.parquet'},
 ('tag','prose_code'): {'src':'/root/v3/answers_sampled_withcode_v3.parquet',
                        'emb':'/root/v3/emb_tag_prosecode_v3.npy',
                        'meta':'/root/v3/meta_tag_prosecode_v3.parquet'},
 ('tag','code'):       {'src':'/root/v3/answers_sampled_codeonly_v3.parquet',
                        'emb':'/root/v3/emb_tag_code_v3.npy',
                        'meta':'/root/v3/meta_tag_code_v3.parquet'},
 ('ql','prose'):       {'src':'/root/v3/answers_ql_v3.parquet',
                        'emb':'/root/v3/emb_ql_prose_v3.npy',
                        'meta':'/root/v3/meta_ql_prose_v3.parquet'},
 ('ql','prose_code'):  {'src':'/root/v3/answers_ql_withcode_v3.parquet',
                        'emb':'/root/v3/emb_ql_prosecode_v3.npy',
                        'meta':'/root/v3/meta_ql_prosecode_v3.parquet'},
 ('ql','code'):        {'src':'/root/v3/answers_ql_codeonly_v3.parquet',
                        'emb':'/root/v3/emb_ql_code_v3.npy',
                        'meta':'/root/v3/meta_ql_code_v3.parquet'},
}


class Embedder:
    """Embeds one analytical cell and cleans the result."""

    def __init__(self, level, content, client=None):
        if (level, content) not in CELLS:
            raise ValueError(f'unknown cell {(level, content)}')
        cfg = CELLS[(level, content)]
        self.level, self.content = level, content
        self.src, self.emb_out, self.meta_out = cfg['src'], cfg['emb'], cfg['meta']
        self.trunc = TRUNCATION[content]
        self.client = client
        self.n_failed = 0

    def estimate(self):
        d = pd.read_parquet(self.src, columns=['BodyLength'])
        tok = d['BodyLength'].clip(upper=self.trunc).sum() / 4
        return len(d), tok, tok / 1e6 * PRICE_PER_MTOK

    def _embed_one(self, text):
        try:
            r = self.client.embeddings.create(model=MODEL, input=[text])
            return np.array(r.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f'    single failed: {str(e)[:90]}', flush=True)
            self.n_failed += 1
            return np.zeros(DIMS, dtype=np.float32)

    def _embed_batch(self, batch):
        for attempt in range(3):
            try:
                r = self.client.embeddings.create(model=MODEL, input=batch)
                return [np.array(d.embedding, dtype=np.float32) for d in r.data]
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt); continue
                print(f'\n  batch failed, falling back per-text: {str(e)[:90]}',
                      flush=True)
                out = []
                for t in batch:
                    out.append(self._embed_one(t)); time.sleep(0.05)
                return out

    def _embed_all(self, df):
        texts = df['Body'].fillna('').astype(str).str.slice(0, self.trunc).tolist()
        n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        chunks = []
        for b in tqdm(range(n_batches), desc=f'  {self.level}/{self.content}'):
            s = b * BATCH_SIZE
            chunks.extend(self._embed_batch(texts[s:s + BATCH_SIZE]))
            time.sleep(0.05)
        return np.stack(chunks)

    def _clean(self, emb, meta):
        norms = np.linalg.norm(emb, axis=1)
        keep = norms > 1e-6
        n_zero = int((~keep).sum())
        print(f'  zero vectors     : {n_zero:,}', flush=True)
        if n_zero:
            emb, meta = emb[keep], meta[keep].reset_index(drop=True)
        if self.level == 'ql':
            before_q = meta['ParentId'].nunique()
            pre  = meta.loc[meta['Post'] == 0].groupby('ParentId').size()
            post = meta.loc[meta['Post'] == 1].groupby('ParentId').size()
            valid = (set(pre[pre >= MIN_PER_PERIOD].index)
                     & set(post[post >= MIN_PER_PERIOD].index))
            m = meta['ParentId'].isin(valid).to_numpy()
            dropped = before_q - len(valid)
            if dropped:
                print(f'  questions dropped: {dropped:,} '
                      f'(fell below {MIN_PER_PERIOD}/period)', flush=True)
                emb, meta = emb[m], meta[m].reset_index(drop=True)
        return emb, meta

    def run(self):
        print('=' * 72)
        print(f'EMBED — {self.level} level, {self.content}   [{MODEL}]')
        print('=' * 72)
        df = pd.read_parquet(self.src)
        n, tok, cost = self.estimate()
        print(f'  answers          : {n:,}')
        print(f'  est. tokens      : {tok/1e6:.2f}M   est. cost ~${cost:.2f}',
              flush=True)

        emb = self._embed_all(df)
        meta = df.drop(columns=['Body']).reset_index(drop=True)
        assert len(emb) == len(meta), 'embedding/metadata length mismatch'
        if self.n_failed:
            print(f'  API failures     : {self.n_failed:,}', flush=True)

        emb, meta = self._clean(emb, meta)
        norms = np.linalg.norm(emb, axis=1)
        print(f'  final shape      : {emb.shape}')
        print(f'  norms            : min {norms.min():.4f}  max {norms.max():.4f}')

        np.save(self.emb_out, emb)
        meta.to_parquet(self.meta_out)
        print(f'  saved            : {self.emb_out}')
        print(f'                     {self.meta_out}')
        print('=' * 72); print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--level',   choices=['tag','ql','all'], default='all')
    ap.add_argument('--content', choices=['prose','prose_code','code','all'],
                    default='all')
    ap.add_argument('--estimate-only', action='store_true')
    a = ap.parse_args()

    levels   = ['tag','ql'] if a.level == 'all' else [a.level]
    contents = ['prose','prose_code','code'] if a.content == 'all' else [a.content]
    targets  = [(lv, ct) for lv in levels for ct in contents]

    print(f'{"cell":<22}{"answers":>10}{"Mtokens":>10}{"cost":>10}')
    print('-' * 52)
    tot_tok = tot_cost = 0
    for lv, ct in targets:
        n, tok, cost = Embedder(lv, ct).estimate()
        tot_tok += tok; tot_cost += cost
        print(f'{lv+"/"+ct:<22}{n:>10,}{tok/1e6:>10.2f}{cost:>10.2f}')
    print('-' * 52)
    print(f'{"TOTAL":<22}{"":>10}{tot_tok/1e6:>10.2f}{tot_cost:>10.2f}\n')

    if a.estimate_only:
        raise SystemExit(0)

    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise SystemExit('OPENAI_API_KEY not set')
    client = OpenAI(api_key=key)
    for lv, ct in targets:
        Embedder(lv, ct, client=client).run()
