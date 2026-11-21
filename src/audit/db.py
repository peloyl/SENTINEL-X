
import sqlite3, json
from pathlib import Path
from datetime import datetime, timezone
def _now(): return datetime.now(timezone.utc).isoformat()
class AuditDB:
    def __init__(self, path):
        self.path=str(path); Path(path).parent.mkdir(parents=True, exist_ok=True); self._init()
    def _conn(self): return sqlite3.connect(self.path)
    def _init(self):
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS transactions(id INTEGER PRIMARY KEY, amount REAL, time REAL, source TEXT);
            CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT, transaction_id INTEGER, model_score REAL, final_score REAL, risk_level TEXT, model_version TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS analyst_decisions(id INTEGER PRIMARY KEY AUTOINCREMENT, transaction_id INTEGER, action TEXT, previous_risk TEXT, comment TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS audit_events(id INTEGER PRIMARY KEY AUTOINCREMENT, transaction_id INTEGER, event TEXT, detail TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY AUTOINCREMENT, transaction_id INTEGER, verified_label INTEGER, source TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS model_versions(version TEXT PRIMARY KEY, report TEXT, created_at TEXT);
            """)
    def log_prediction(self, tx_id, model_score, final_score, level, version):
        with self._conn() as c: c.execute("INSERT INTO predictions(transaction_id,model_score,final_score,risk_level,model_version,created_at) VALUES(?,?,?,?,?,?)",(tx_id,model_score,final_score,level,version,_now()))
    def save_decision(self, tx_id, action, previous_risk, comment=""):
        with self._conn() as c:
            c.execute("INSERT INTO analyst_decisions(transaction_id,action,previous_risk,comment,created_at) VALUES(?,?,?,?,?)",(tx_id,action,previous_risk,comment,_now()))
            c.execute("INSERT INTO audit_events(transaction_id,event,detail,created_at) VALUES(?,?,?,?)",(tx_id,action,comment,_now()))
    def save_feedback(self, tx_id, verified_label, source="manual_verified"):
        with self._conn() as c: c.execute("INSERT INTO feedback(transaction_id,verified_label,source,created_at) VALUES(?,?,?,?)",(tx_id,verified_label,source,_now()))
    def decisions(self):
        import pandas as pd
        with self._conn() as c: return pd.read_sql("SELECT * FROM analyst_decisions ORDER BY created_at DESC", c)
    def audit_events(self):
        import pandas as pd
        with self._conn() as c: return pd.read_sql("SELECT * FROM audit_events ORDER BY created_at DESC", c)
    def counts(self):
        with self._conn() as c:
            return {"decisions": c.execute("SELECT COUNT(*) FROM analyst_decisions").fetchone()[0],
                    "confirmed_fraud": c.execute("SELECT COUNT(*) FROM analyst_decisions WHERE action='Confirm Fraud'").fetchone()[0],
                    "legitimate": c.execute("SELECT COUNT(*) FROM analyst_decisions WHERE action='Mark as Legitimate'").fetchone()[0],
                    "escalated": c.execute("SELECT COUNT(*) FROM analyst_decisions WHERE action='Escalate for Review'").fetchone()[0]}
