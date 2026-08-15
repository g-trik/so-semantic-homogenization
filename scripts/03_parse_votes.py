"""
03_parse_votes.py
=================
Parses Votes.xml into batched parquet files, retaining only accept (1) and
upvote (2) records.
"""
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

def parse_votes(xml_path):
    rows = []
    batch = 0
    for event, elem in tqdm(ET.iterparse(xml_path, events=('end',)), desc='Parsing Votes.xml'):
        if elem.tag == 'row':
            vtype = elem.get('VoteTypeId')
            if vtype in ('1', '2'):
                rows.append({
                    'PostId':     elem.get('PostId'),
                    'VoteTypeId': vtype
                })
            elem.clear()

        if len(rows) >= 1000000:
            pd.DataFrame(rows).to_parquet(f'/root/votes_batch_{batch}.parquet')
            print(f'Saved batch {batch}')
            batch += 1
            rows = []

    if rows:
        pd.DataFrame(rows).to_parquet(f'/root/votes_batch_{batch}.parquet')
        print(f'Saved final batch {batch}')

    print('Done parsing Votes.xml!')

parse_votes('/root/Votes.xml')
