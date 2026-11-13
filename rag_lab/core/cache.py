"""Explicit local SQLite response cache; never stores OAuth tokens."""
import hashlib, json, sqlite3
from pathlib import Path

class ResponseCache:
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT NOT NULL, usage TEXT NOT NULL)")

    @staticmethod
    def key(model,payload):
        serialized=json.dumps({"model":model,"payload":payload},sort_keys=True,ensure_ascii=False)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get(self,key):
        with sqlite3.connect(self.path) as db:
            row=db.execute("SELECT response, usage FROM responses WHERE key=?",(key,)).fetchone()
        return (row[0],json.loads(row[1])) if row else None

    def put(self,key,text,usage):
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO responses VALUES(?,?,?)",
                       (key,text,json.dumps(usage)))
