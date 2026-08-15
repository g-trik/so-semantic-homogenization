"""
04_parse_tags.py
================
Parses Tags.xml into /root/tags.parquet, retaining the tag name and its
site-wide post count.
"""
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

def parse_tags(xml_path):
    rows = []
    for event, elem in tqdm(ET.iterparse(xml_path, events=('end',)), desc='Parsing Tags.xml'):
        if elem.tag == 'row':
            rows.append({
                'TagId':    elem.get('Id'),
                'TagName':  elem.get('TagName'),
                'Count':    int(elem.get('Count', 0))
            })
            elem.clear()
    pd.DataFrame(rows).to_parquet('/root/tags.parquet')
    print(f'Done! {len(rows)} tags saved.')

parse_tags('/root/Tags.xml')
