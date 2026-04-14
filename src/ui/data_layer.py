"""
Data layer SQLite pour Chainlit — persistence des conversations.
Permet la sidebar avec historique des threads.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Optional

from chainlit.data import BaseDataLayer
from chainlit.types import (
    Feedback,
    ThreadDict,
    PageInfo,
    PaginatedResponse,
)

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("CHAINLIT_DB_PATH", "data/chainlit.db")


class SQLiteDataLayer(BaseDataLayer):
    """Data layer SQLite pour persister les conversations Chainlit."""

    def __init__(self):
        os.makedirs(
            os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True
        )
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_tables()
        logger.info("SQLite data layer initialise : %s", DB_PATH)

    def _init_tables(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                identifier TEXT,
                metadata TEXT,
                created_at TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                name TEXT,
                user_id TEXT,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                id TEXT PRIMARY KEY,
                thread_id TEXT,
                type TEXT,
                name TEXT,
                output TEXT,
                metadata TEXT,
                created_at TEXT
            )
        """)
        self.conn.commit()

    async def get_user(self, identifier: str):
        c = self.conn.cursor()
        c.execute(
            "SELECT id, identifier, metadata FROM users WHERE identifier = ?",
            (identifier,),
        )
        row = c.fetchone()
        if row:
            from chainlit.user import PersistedUser

            return PersistedUser(
                id=row[0],
                identifier=row[1],
                metadata=json.loads(row[2] or "{}"),
                createdAt=datetime.now().isoformat(),
            )
        return None

    async def create_user(self, user):
        import uuid

        user_id = str(uuid.uuid4())
        c = self.conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO users (id, identifier, metadata, created_at) VALUES (?, ?, ?, ?)",
            (
                user_id,
                user.identifier,
                json.dumps(user.metadata or {}),
                datetime.now().isoformat(),
            ),
        )
        self.conn.commit()
        from chainlit.user import PersistedUser

        return PersistedUser(
            id=user_id,
            identifier=user.identifier,
            metadata=user.metadata or {},
            createdAt=datetime.now().isoformat(),
        )

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[list] = None,
    ):
        c = self.conn.cursor()
        c.execute("SELECT id FROM threads WHERE id = ?", (thread_id,))
        now = datetime.now().isoformat()
        if c.fetchone():
            updates = []
            params = []
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))
            updates.append("updated_at = ?")
            params.append(now)
            params.append(thread_id)
            c.execute(f"UPDATE threads SET {', '.join(updates)} WHERE id = ?", params)
        else:
            c.execute(
                "INSERT INTO threads (id, name, user_id, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    name or "",
                    user_id or "",
                    json.dumps(metadata or {}),
                    now,
                    now,
                ),
            )
        self.conn.commit()

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        c = self.conn.cursor()
        c.execute(
            "SELECT id, name, user_id, metadata, created_at FROM threads WHERE id = ?",
            (thread_id,),
        )
        row = c.fetchone()
        if not row:
            return None

        c.execute(
            "SELECT id, type, name, output, created_at FROM steps WHERE thread_id = ? ORDER BY created_at",
            (thread_id,),
        )
        steps = []
        for s in c.fetchall():
            steps.append(
                {
                    "id": s[0],
                    "type": s[1],
                    "name": s[2],
                    "output": s[3],
                    "createdAt": s[4],
                }
            )

        return {
            "id": row[0],
            "name": row[1],
            "userIdentifier": row[2] or "",
            "metadata": json.loads(row[3] or "{}"),
            "createdAt": row[4],
            "steps": steps,
        }

    async def get_thread_author(self, thread_id: str) -> str:
        c = self.conn.cursor()
        c.execute("SELECT user_id FROM threads WHERE id = ?", (thread_id,))
        row = c.fetchone()
        return row[0] if row else ""

    async def list_threads(self, pagination, filters):
        c = self.conn.cursor()
        user_id = filters.user_id if hasattr(filters, "user_id") else None

        # Ne lister que les threads qui ont des steps (pas les threads vides)
        if user_id:
            c.execute(
                "SELECT t.id, t.name, t.metadata, t.created_at FROM threads t "
                "WHERE t.user_id = ? AND EXISTS (SELECT 1 FROM steps s WHERE s.thread_id = t.id) "
                "ORDER BY t.updated_at DESC LIMIT 20",
                (user_id,),
            )
        else:
            c.execute(
                "SELECT t.id, t.name, t.metadata, t.created_at FROM threads t "
                "WHERE EXISTS (SELECT 1 FROM steps s WHERE s.thread_id = t.id) "
                "ORDER BY t.updated_at DESC LIMIT 20"
            )

        threads = []
        for row in c.fetchall():
            threads.append(
                ThreadDict(
                    id=row[0],
                    name=row[1] or "Conversation",
                    metadata=json.loads(row[2] or "{}"),
                    createdAt=row[3],
                    steps=[],
                )
            )

        return PaginatedResponse(
            data=threads,
            pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None),
        )

    async def delete_thread(self, thread_id: str):
        c = self.conn.cursor()
        c.execute("DELETE FROM steps WHERE thread_id = ?", (thread_id,))
        c.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        self.conn.commit()

    async def create_step(self, step_dict):
        c = self.conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO steps (id, thread_id, type, name, output, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                step_dict.get("id", ""),
                step_dict.get("threadId", ""),
                step_dict.get("type", ""),
                step_dict.get("name", ""),
                step_dict.get("output", ""),
                json.dumps(step_dict.get("metadata", {})),
                step_dict.get("createdAt", datetime.now().isoformat()),
            ),
        )
        self.conn.commit()

    async def update_step(self, step_dict):
        await self.create_step(step_dict)

    async def delete_step(self, step_id: str):
        c = self.conn.cursor()
        c.execute("DELETE FROM steps WHERE id = ?", (step_id,))
        self.conn.commit()

    async def create_element(self, element):
        pass

    async def get_element(self, thread_id, element_id):
        return None

    async def delete_element(self, element_id):
        pass

    async def upsert_feedback(self, feedback):
        return ""

    async def delete_feedback(self, feedback_id):
        return True

    async def get_favorite_steps(self, user_id):
        return []

    async def set_step_favorite(self, step_id, user_id, favorite):
        pass

    async def build_debug_url(self, *args, **kwargs):
        return ""

    async def close(self):
        self.conn.close()
