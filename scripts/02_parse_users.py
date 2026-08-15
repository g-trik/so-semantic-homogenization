"""
02_parse_users.py
=================
Parses Users.xml into /root/users.parquet.

Captures Reputation (dump-date snapshot) and CreationDate (account creation).

NOTE: the dump contains no reputation history, so Reputation is the value as of
the dump date, not as of the time a user posted a given answer. Because
reputation is monotonically increasing, pre-period answers are systematically
over-assigned. AccountCreated is not subject to this contamination and supports
a tenure-at-posting robustness check on the reputation moderator (H4).
"""
import xml.etree.ElementTree as ET
import pandas as pd


def parse_users(xml_path='/root/Users.xml', out='/root/users.parquet'):
    rows = []
    context = ET.iterparse(xml_path, events=('start', 'end'))
    _, root = next(context)

    for event, elem in context:
        if event == 'end' and elem.tag == 'row':
            rows.append({
                'UserId':         elem.get('Id'),
                'Reputation':     int(elem.get('Reputation', 0)),
                'AccountCreated': (elem.get('CreationDate') or '')[:10],
            })
            elem.clear()
            root.clear()

    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f'Saved {len(df):,} users -> {out}')
    print(f'  columns          : {list(df.columns)}')
    print(f'  AccountCreated   : {df.AccountCreated.min()} to {df.AccountCreated.max()}')
    print(f'  missing created  : {(df.AccountCreated == "").sum():,}')
    print(f'  median reputation: {df.Reputation.median():,.0f}')


if __name__ == '__main__':
    parse_users()
