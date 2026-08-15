"""
05_join.py
==========
Joins answer batches with accepted-answer flags, user reputation, account
creation date and question tags, for any of the three corpora.

  prose only    : answers_batch_{i}.parquet    -> answers_with_tags.parquet
  prose & code  : answers_batch_wc_{i}.parquet -> answers_with_tags_withcode.parquet
  code only     : answers_batch_co_{i}.parquet -> answers_with_tags_codeonly.parquet

Reputation is left as a nullable Int64 rather than filled with zero. Answers
whose author account has since been deleted carry no OwnerUserId and would
otherwise be assigned Reputation = 0, placing them below the novice cutoff and
misclassifying them as new contributors. HasOwner marks these rows so that
downstream metrics can use the known-owner subset as denominator.

Question tags are merged here from question_tags.parquet, written by
01_parse_posts.py, so Posts.xml is not rescanned.

Output is streamed to parquet batch by batch, so peak memory is one batch
rather than the whole corpus.

Usage:
  python3 05_join.py
  python3 05_join.py codeonly
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CORPORA = {
    'prose': {
        'glob':     '/root/answers_batch_[0-9]*.parquet',
        'accepted': '/root/accepted_ids.parquet',
        'out':      '/root/answers_with_tags.parquet',
    },
    'withcode': {
        'glob':     '/root/answers_batch_wc_[0-9]*.parquet',
        'accepted': '/root/accepted_ids_wc.parquet',
        'out':      '/root/answers_with_tags_withcode.parquet',
    },
    'codeonly': {
        'glob':     '/root/answers_batch_co_[0-9]*.parquet',
        'accepted': '/root/accepted_ids_co.parquet',
        'out':      '/root/answers_with_tags_codeonly.parquet',
    },
}

USERS_PATH = '/root/users.parquet'
TAGS_PATH  = '/root/question_tags.parquet'


class AnswerJoiner:
    """Joins one corpus with accepted flags, reputation and question tags."""

    def __init__(self, corpus, users_path=USERS_PATH, tags_path=TAGS_PATH):
        if corpus not in CORPORA:
            raise ValueError(f'corpus must be one of {list(CORPORA)}')
        self.corpus = corpus
        cfg = CORPORA[corpus]
        self.files    = sorted(glob.glob(cfg['glob']))
        self.accepted = cfg['accepted']
        self.out      = cfg['out']
        self.users_path = users_path
        self.tags_path  = tags_path
        self.stats = {}

    # ---------------- lookups ----------------
    @staticmethod
    def _sorted_index(keys, values):
        """Sort a key/value pair into numpy arrays for searchsorted lookup."""
        k = pd.to_numeric(keys, errors='coerce')
        m = k.notna()
        k = k[m].astype('int64').to_numpy()
        v = values[m].to_numpy()
        o = np.argsort(k)
        return k[o], v[o]

    @staticmethod
    def _lookup(keys, sorted_keys, values):
        """Vectorised left-join. Returns object array, None where key absent."""
        k = pd.to_numeric(keys, errors='coerce').to_numpy()
        valid = ~np.isnan(k)
        ki = np.where(valid, k, -1).astype('int64')
        pos = np.clip(np.searchsorted(sorted_keys, ki), 0, len(sorted_keys) - 1)
        hit = valid & (sorted_keys[pos] == ki)
        out = np.empty(len(k), dtype=object)
        out[:] = None
        out[hit] = values[pos[hit]]
        return out

    def _load_lookups(self):
        acc = pd.read_parquet(self.accepted)
        self.acc_key = np.sort(pd.to_numeric(acc['PostId'], errors='coerce')
                                 .dropna().astype('int64').to_numpy())
        print(f'  accepted ids   : {len(self.acc_key):,}', flush=True)
        del acc

        u = pd.read_parquet(self.users_path)
        self.u_key, self.u_rep = self._sorted_index(u['UserId'], u['Reputation'])
        self.has_created = 'AccountCreated' in u.columns
        if self.has_created:
            _, self.u_created = self._sorted_index(u['UserId'], u['AccountCreated'])
        print(f'  users          : {len(self.u_key):,}'
              f"{'  (with AccountCreated)' if self.has_created else ''}", flush=True)
        del u

        t = pd.read_parquet(self.tags_path)
        self.t_key, self.t_tags = self._sorted_index(t['ParentId'], t['Tags'])
        print(f'  question tags  : {len(self.t_key):,}', flush=True)
        del t

    # ---------------- per-batch join ----------------
    def _join_batch(self, a):
        pid = pd.to_numeric(a['PostId'], errors='coerce').to_numpy()
        v = ~np.isnan(pid)
        pi = np.where(v, pid, -1).astype('int64')
        pos = np.clip(np.searchsorted(self.acc_key, pi), 0, len(self.acc_key) - 1)
        a['AcceptedFlag'] = (v & (self.acc_key[pos] == pi)).astype('int8')

        rep = self._lookup(a['OwnerUserId'], self.u_key, self.u_rep)
        a['Reputation'] = pd.array(rep, dtype='Int64')
        a['HasOwner'] = a['Reputation'].notna()
        if self.has_created:
            a['AccountCreated'] = self._lookup(a['OwnerUserId'],
                                               self.u_key, self.u_created)

        n_before = len(a)
        a['Tags'] = self._lookup(a['ParentId'], self.t_key, self.t_tags)
        a = a[a['Tags'].notna()].copy()
        self.stats['dropped_no_tags'] = (
            self.stats.get('dropped_no_tags', 0) + n_before - len(a))
        return a

    # ---------------- run ----------------
    def run(self):
        print('=' * 72)
        print(f'JOIN — {self.corpus}')
        print('=' * 72)
        if not self.files:
            raise FileNotFoundError(f'no batches matched for {self.corpus}')
        print(f'  batches        : {len(self.files)}')
        self._load_lookups()

        if os.path.exists(self.out):
            os.remove(self.out)

        writer = None
        total_in = total_out = 0
        try:
            for i, f in enumerate(self.files, 1):
                a = pd.read_parquet(f)
                total_in += len(a)
                a = self._join_batch(a)
                total_out += len(a)

                table = pa.Table.from_pandas(a, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(self.out, table.schema)
                writer.write_table(table)
                print(f'    [{i:>3}/{len(self.files)}] {os.path.basename(f)} '
                      f'-> {len(a):,} rows', flush=True)
                del a, table
        finally:
            if writer is not None:
                writer.close()

        self._report(total_in, total_out)

    def _report(self, total_in, total_out):
        d = pd.read_parquet(self.out, columns=['AcceptedFlag', 'HasOwner',
                                              'Reputation'])
        print()
        print(f'  rows in        : {total_in:,}')
        print(f'  rows out       : {total_out:,}')
        print(f'  dropped (no tags): {self.stats.get("dropped_no_tags", 0):,}')
        print(f'  accepted       : {d.AcceptedFlag.sum():,} '
              f'({100 * d.AcceptedFlag.mean():.2f}%)')
        print(f'  missing owner  : {(~d.HasOwner).sum():,} '
              f'({100 * (~d.HasOwner).mean():.2f}%)  [Reputation is NA]')
        print(f'  median rep     : {d.Reputation.median()}')
        print(f'  saved          : {self.out}')
        print('=' * 72)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus', nargs='?', default='all',
                    choices=['prose', 'withcode', 'codeonly', 'all'])
    args = ap.parse_args()
    targets = list(CORPORA) if args.corpus == 'all' else [args.corpus]
    for c in targets:
        AnswerJoiner(c).run()
        print()
