"""
memory.py
---------
Lightweight SQLite logging of every agent decision, so the pipeline has an
auditable trail (required for the 'Logs/evaluation sheet' deliverable and
for responsible-use in a sensitive domain).
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "agent_logs.db")


def init_db(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            need_id TEXT,
            status TEXT,
            score REAL,
            message TEXT,
            raw_result TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_decision(guardrail_result: dict, ai_message: str = "", db_path: str = DB_PATH):
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO agent_logs (timestamp, need_id, status, score, message, raw_result) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.utcnow().isoformat(),
            str(guardrail_result.get("need", {}).get("id", "unknown")),
            guardrail_result.get("status"),
            guardrail_result.get("score", 0.0),
            ai_message,
            json.dumps(guardrail_result, default=str),
        ),
    )
    conn.commit()
    conn.close()


def get_recent_logs(limit: int = 20, db_path: str = DB_PATH):
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM agent_logs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
