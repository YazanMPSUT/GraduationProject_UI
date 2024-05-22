import argparse
import os, sqlite3

parser=argparse.ArgumentParser()

parser.add_argument("--dbpath", help="Path of database to truncate table in")
parser.add_argument("--pickle",help="Filename of pickle to delete")

args=parser.parse_args()

if args.pickle and os.path.isfile(args.pickle):
    os.remove(args.pickle)

if args.dbpath and os.path.isfile(args.dbpath):
    conn = sqlite3.connect(args.dbpath)
    cur = conn.cursor()
    cur.execute( "SELECT name FROM sqlite_schema WHERE type='table';")
    tables = [table[0] for table in cur.fetchall()]
    print(tables)
    for table in tables:
        cur.execute( f"DROP TABLE {table}",)
    conn.commit()
    cur.close()