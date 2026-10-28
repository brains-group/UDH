"""[UDH] sqlite cache of chat-completion responses, keyed on the exact request.

HyGRAG keeps no record of LLM calls and saves its graph only at the end of the build, so a crash would lose
every call made so far. With this cache a rerun replays each finished call (same request -> same stored
response) for free; only calls that never completed go to the API. Enabled when HYGRAG_LLM_CACHE is set to
a file path. Only responses with non-empty content are stored.
"""
import hashlib
import json
import os
import sqlite3
import threading

from openai.types.chat import ChatCompletion


class ResponseCache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
        self.conn.commit()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(request: dict) -> str:
        data = {k: v for k, v in request.items() if k != "timeout"}
        return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()

    def get(self, key: str):
        with self.lock:
            row = self.conn.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return ChatCompletion.model_validate_json(row[0])

    def put(self, key: str, rsp: ChatCompletion):
        if not (rsp.choices and rsp.choices[0].message.content):
            return  # never cache an empty reply; a rerun retries it
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, rsp.model_dump_json()))
            self.conn.commit()


_CACHE = None


def get_cache():
    global _CACHE
    path = os.environ.get("HYGRAG_LLM_CACHE")
    if path and _CACHE is None:
        _CACHE = ResponseCache(path)
    return _CACHE
