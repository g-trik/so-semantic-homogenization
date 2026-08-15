"""
01_parse_posts.py
=================
Single-pass parser for the Stack Exchange Posts.xml dump.

Produces all three answer corpora from one traversal so the date filter,
community-owned exclusion and minimum-length rule are applied identically:

  prose only    -> answers_batch_{i}.parquet      + accepted_ids.parquet
  prose & code  -> answers_batch_wc_{i}.parquet   + accepted_ids_wc.parquet
  code only     -> answers_batch_co_{i}.parquet   + accepted_ids_co.parquet

Extraction order (all variants): strip HTML tags FIRST, then decode entities.
Stack Overflow stores angle-bracket code as escaped entities; decoding before
tag-stripping lets the tag regex consume the decoded code as if it were markup.

Code extraction captures <pre> blocks whether or not they nest <code>, then
remaining inline <code> spans. Rows carry HasCode and CodeFromInlineOnly so
downstream steps can build a common subsample and test sensitivity.
"""

import html
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter

import pandas as pd

POSTS_XML  = '/root/Posts.xml'
DATE_START = '2019-01-01'
DATE_END   = '2025-12-31'      # inclusive, compared on the date part only
MIN_CHARS  = 10                # identical across all three corpora
BATCH_SIZE = 250_000

TAG_STRIP   = re.compile(r'<[^>]+>')
TAG_NAMES   = re.compile(r'<([^>]+)>')
PRE_ANY     = re.compile(r'<pre[^>]*>(.*?)</pre>', re.DOTALL | re.IGNORECASE)
INLINE_CODE = re.compile(r'<code[^>]*>(.*?)</code>', re.DOTALL | re.IGNORECASE)
CODE_BLOCK  = re.compile(r'<pre[^>]*>.*?</pre>|<code[^>]*>.*?</code>',
                         re.DOTALL | re.IGNORECASE)


def _normalise(text):
    """Strip residual tags, decode entities, collapse whitespace. Order matters."""
    text = TAG_STRIP.sub(' ', text)
    text = html.unescape(text)
    return ' '.join(text.split()).strip()


class PostsParser:
    """One pass over Posts.xml producing three answer corpora."""

    PREFIX = {'prose': 'answers_batch',
              'withcode': 'answers_batch_wc',
              'codeonly': 'answers_batch_co'}

    def __init__(self, xml_path=POSTS_XML, min_chars=MIN_CHARS,
                 date_start=DATE_START, date_end=DATE_END,
                 batch_size=BATCH_SIZE, out_dir='/root',
                 prefix_suffix='', max_answers=None):
        self.xml_path      = xml_path
        self.min_chars     = min_chars
        self.date_start    = date_start
        self.date_end      = date_end
        self.batch_size    = batch_size
        self.out_dir       = out_dir.rstrip('/')
        self.prefix_suffix = prefix_suffix   # e.g. '_TEST' for dry runs
        self.max_answers   = max_answers     # stop early when testing

        self.accepted_ids = set()
        self.question_tags = {}      # qid -> list[str]
        self.question_meta = {}      # qid -> (answer_count, view_count)
        self.scanned  = 0
        self.t0       = None
        self.stats    = Counter()
        self.buffers  = {k: [] for k in self.PREFIX}
        self.batch_no = {k: 0 for k in self.PREFIX}

    # ---------------- extraction ----------------
    @staticmethod
    def extract_prose(body):
        """Prose only: remove code blocks entirely, then normalise."""
        return _normalise(CODE_BLOCK.sub(' ', body))

    @staticmethod
    def extract_withcode(body):
        """Prose & code: keep all inner text including code block contents."""
        return _normalise(body)

    @staticmethod
    def extract_codeonly(body):
        """
        Code only. Captures every <pre>...</pre> block (whether or not it
        nests <code>), then any remaining inline <code> spans.
        Returns (code_text, from_inline_only).
        """
        if not body:
            return '', False

        pre_blocks = PRE_ANY.findall(body)
        rest = PRE_ANY.sub(' ', body)
        inline = INLINE_CODE.findall(rest)

        blocks = pre_blocks + inline
        if not blocks:
            return '', False

        text = _normalise('\n'.join(blocks))
        return text, (bool(inline) and not pre_blocks)

    # ---------------- I/O ----------------
    def _path(self, key, n):
        return f'{self.out_dir}/{self.PREFIX[key]}{self.prefix_suffix}_{n}.parquet'

    def _flush(self, key, final=False):
        buf = self.buffers[key]
        if not buf:
            return
        if len(buf) >= self.batch_size or final:
            n = self.batch_no[key]
            pd.DataFrame(buf).to_parquet(self._path(key, n))
            print(f'  saved {self._path(key, n)} ({len(buf):,} rows)', flush=True)
            self.batch_no[key] += 1
            self.buffers[key] = []

    # ---------------- main loop ----------------
    def run(self):
        context = ET.iterparse(self.xml_path, events=('start', 'end'))
        _, root = next(context)

        self.t0 = time.time()
        for event, elem in context:
            if event == 'end' and elem.tag == 'row':
                self.scanned += 1
                if self.scanned % 2_000_000 == 0:
                    self._progress()
                ptype = elem.get('PostTypeId')
                if ptype == '1':
                    acc = elem.get('AcceptedAnswerId')
                    if acc:
                        self.accepted_ids.add(acc)
                    self._handle_question(elem)
                elif ptype == '2':
                    self._handle_answer(elem)
                elem.clear()
                root.clear()
                if (self.max_answers and
                        self.stats['answers_in_window'] >= self.max_answers):
                    print(f'\n  stopping early at '
                          f'{self.stats["answers_in_window"]:,} answers', flush=True)
                    break

        for key in self.buffers:
            self._flush(key, final=True)
        self._write_accepted()
        self._write_questions()
        self._report()

    def _progress(self):
        el = time.time() - self.t0
        rate = self.scanned / el if el else 0
        buf = sum(len(b) for b in self.buffers.values())
        matched = self.stats['answers_in_window']
        phase = 'scanning to 2019' if matched == 0 else 'collecting'
        print(f'  [{el/60:6.1f}m] {self.scanned:>12,} rows | '
              f'{matched:>9,} answers | {rate:>9,.0f} rows/s | '
              f'buffer {buf:>7,} | {phase}', flush=True)

    def _handle_question(self, elem):
        """Capture per-question metadata so 06_add_tags need not rescan Posts.xml."""
        qid = elem.get('Id')
        raw = elem.get('Tags', '') or ''
        tags = TAG_NAMES.findall(raw)
        if tags:
            self.question_tags[qid] = tags
        ac = elem.get('AnswerCount')
        vc = elem.get('ViewCount')
        if ac is not None or vc is not None:
            self.question_meta[qid] = (int(ac) if ac else 0,
                                       int(vc) if vc else 0)

    def _handle_answer(self, elem):
        created = elem.get('CreationDate', '')
        if not (self.date_start <= created[:10] <= self.date_end):
            return
        if elem.get('CommunityOwnedDate') is not None:
            self.stats['skipped_community_owned'] += 1
            return

        body = elem.get('Body', '') or ''
        self.stats['answers_in_window'] += 1

        prose = self.extract_prose(body)
        withcode = self.extract_withcode(body)
        codeonly, inline_only = self.extract_codeonly(body)
        has_code = len(codeonly) >= self.min_chars

        if not has_code:
            self.stats['no_usable_code'] += 1
        if has_code and inline_only:
            self.stats['code_from_inline_only'] += 1

        base = {'PostId':       elem.get('Id'),
                'ParentId':     elem.get('ParentId'),
                'OwnerUserId':  elem.get('OwnerUserId'),
                'CreationDate': created[:7],
                'Score':        int(elem.get('Score', 0)),
                'HasCode':      has_code}

        if len(prose) >= self.min_chars:
            self.buffers['prose'].append(
                {**base, 'Body': prose, 'BodyLength': len(prose)})
        else:
            self.stats['prose_below_min'] += 1

        if len(withcode) >= self.min_chars:
            self.buffers['withcode'].append(
                {**base, 'Body': withcode, 'BodyLength': len(withcode)})
        else:
            self.stats['withcode_below_min'] += 1

        if has_code:
            self.buffers['codeonly'].append(
                {**base, 'Body': codeonly, 'BodyLength': len(codeonly),
                 'CodeFromInlineOnly': inline_only})

        for key in self.buffers:
            self._flush(key)

    def _write_questions(self):
        qt = pd.DataFrame({
            'ParentId': list(self.question_tags.keys()),
            'Tags':     list(self.question_tags.values())})
        qt.to_parquet(f'{self.out_dir}/question_tags{self.prefix_suffix}.parquet')
        print(f'  saved {len(qt):,} question tag rows')

        if self.question_meta:
            qm = pd.DataFrame({
                'ParentId':    list(self.question_meta.keys()),
                'AnswerCount': [v[0] for v in self.question_meta.values()],
                'ViewCount':   [v[1] for v in self.question_meta.values()]})
            qm.to_parquet(f'{self.out_dir}/question_meta{self.prefix_suffix}.parquet')
            print(f'  saved {len(qm):,} question meta rows')

    def _write_accepted(self):
        ids = pd.DataFrame({'PostId': list(self.accepted_ids)})
        for name in ('accepted_ids', 'accepted_ids_wc', 'accepted_ids_co'):
            ids.to_parquet(f'{self.out_dir}/{name}{self.prefix_suffix}.parquet')
        print(f'\n  saved {len(self.accepted_ids):,} accepted answer IDs '
              f'(identical across all three corpora)')

    def _report(self):
        print('\n' + '=' * 72)
        print('PARSE SUMMARY')
        print('=' * 72)
        print(f'  window          : {self.date_start} to {self.date_end} (inclusive)')
        print(f'  minimum chars   : {self.min_chars}')
        for k in sorted(self.stats):
            print(f'  {k:<25} : {self.stats[k]:>12,}')
        print()
        for key in self.PREFIX:
            print(f'  {self.PREFIX[key]:<20} : {self.batch_no[key]:>3} batch file(s)')
        print('=' * 72)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true',
                    help='dry run: 50k answers, _TEST file suffix')
    a = ap.parse_args()
    if a.test:
        PostsParser(prefix_suffix='_TEST', max_answers=50_000,
                    batch_size=100_000).run()
    else:
        PostsParser().run()
