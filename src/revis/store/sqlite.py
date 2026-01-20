"""SQLite implementation of RunStore."""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from revis.types import (
    Artifact,
    Budget,
    Decision,
    Metric,
    Run,
    Session,
    Suggestion,
    TerminationReason,
)


class SQLiteRunStore:
    """SQLite-based run store."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._migrate()  # Run migrations on first connection
        return self._conn

    def initialize(self) -> None:
        """Initialize the database schema."""
        schema_path = Path(__file__).parent / "schema.sql"
        schema = schema_path.read_text()
        self.conn.executescript(schema)
        self.conn.commit()
        self._migrate()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _migrate(self) -> None:
        """Run migrations for existing databases."""
        # Check if sessions table exists first
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        if not cursor.fetchone():
            return  # Table doesn't exist yet, skip migrations

        # Session migrations
        cursor = self._conn.execute("PRAGMA table_info(sessions)")
        session_columns = [row[1] for row in cursor.fetchall()]

        if "name" not in session_columns:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN name TEXT UNIQUE")
            self._conn.execute("ALTER TABLE sessions ADD COLUMN exported_at TIMESTAMP")
            self._conn.execute(
                "UPDATE sessions SET name = 'session-' || id WHERE name IS NULL"
            )
            self._conn.commit()

        if "config_snapshot" not in session_columns:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN config_snapshot TEXT")
            self._conn.commit()

        # Run migrations - add new columns for iteration tracking
        cursor = self._conn.execute("PRAGMA table_info(runs)")
        run_columns = [row[1] for row in cursor.fetchall()]

        new_run_columns = [
            ("change_type", "TEXT"),
            ("change_description", "TEXT"),
            ("change_diff", "TEXT"),
            ("hypothesis", "TEXT"),
            ("metrics_json", "TEXT"),
            ("outcome", "TEXT"),
            ("analysis", "TEXT"),
        ]
        for col_name, col_type in new_run_columns:
            if col_name not in run_columns:
                self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_type}")
        self._conn.commit()

        # Add traces table if it doesn't exist
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(id),
                timestamp TEXT DEFAULT (datetime('now')),
                event_type TEXT NOT NULL,
                data_json TEXT NOT NULL
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_run ON traces(run_id)")

        # Add suggestions table if it doesn't exist
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                run_id TEXT REFERENCES runs(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                suggestion_type TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                handed_off_to TEXT
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggestions_session ON suggestions(session_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggestions_run ON suggestions(run_id)"
        )
        self.conn.commit()

    # Session management

    def create_session(
        self,
        name: str,
        branch: str,
        base_sha: str,
        budget: Budget,
        baseline_run_id: str | None = None,
    ) -> str:
        session_id = str(uuid.uuid4())[:8]
        self.conn.execute(
            """
            INSERT INTO sessions (
                id, name, branch, base_sha, baseline_run_id,
                budget_type, budget_value, retry_budget, pid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                name,
                branch,
                base_sha,
                baseline_run_id,
                budget.type,
                budget.value,
                3,  # Default retry budget
                os.getpid(),
            ),
        )
        self.conn.commit()
        return session_id

    def end_session(
        self,
        session_id: str,
        reason: TerminationReason,
        pr_url: str | None,
    ) -> None:
        if reason == TerminationReason.TARGET_ACHIEVED:
            status = "completed"
        elif reason == TerminationReason.ERROR:
            status = "failed"
        else:
            status = "stopped"
        self.conn.execute(
            """
            UPDATE sessions
            SET status = ?, termination_reason = ?, pr_url = ?, ended_at = ?
            WHERE id = ?
            """,
            (status, reason.value, pr_url, datetime.now().isoformat(), session_id),
        )
        self.conn.commit()

    def get_session(self, session_id: str) -> Session | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def get_running_session(self) -> Session | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE status = 'running' ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def get_session_by_name(self, name: str) -> Session | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def list_sessions(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Session]:
        query = "SELECT * FROM sessions"
        params: list = []

        if status:
            query += " WHERE status = ?"
            params.append(status)

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_session(row) for row in rows]

    def delete_session(self, session_id: str, force: bool = False) -> bool:
        session = self.get_session(session_id)
        if session is None:
            return False
        if session.status == "running" and not force:
            raise ValueError("Cannot delete running session (use --force)")

        runs = self.query_runs(session_id=session_id, limit=1000)
        run_ids = [r.id for r in runs]

        for run_id in run_ids:
            self.conn.execute("DELETE FROM traces WHERE run_id = ?", (run_id,))
            self.conn.execute("DELETE FROM decisions WHERE run_id = ?", (run_id,))
            self.conn.execute("DELETE FROM artifacts WHERE run_id = ?", (run_id,))
            self.conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
            self.conn.execute("DELETE FROM params WHERE run_id = ?", (run_id,))

        self.conn.execute("DELETE FROM runs WHERE session_id = ?", (session_id,))
        self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.conn.commit()
        return True

    def mark_session_exported(self, session_id: str, pr_url: str | None) -> None:
        self.conn.execute(
            "UPDATE sessions SET exported_at = ?, pr_url = ? WHERE id = ?",
            (datetime.now().isoformat(), pr_url, session_id),
        )
        self.conn.commit()

    def session_name_exists(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sessions WHERE name = ? LIMIT 1", (name,)
        ).fetchone()
        return row is not None

    def get_orphaned_sessions(self) -> list[Session]:
        """Get sessions marked as running but whose process is dead."""
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE status = 'running'"
        ).fetchall()

        orphaned = []
        for row in rows:
            pid = row["pid"]
            if pid and not self._process_exists(pid):
                orphaned.append(self._row_to_session(row))
        return orphaned

    def _process_exists(self, pid: int) -> bool:
        """Check if a process exists."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def update_session_budget(self, session_id: str, budget_used: int) -> None:
        self.conn.execute(
            "UPDATE sessions SET budget_used = ? WHERE id = ?",
            (budget_used, session_id),
        )
        self.conn.commit()

    def update_session_cost(self, session_id: str, cost_usd: float) -> None:
        self.conn.execute(
            "UPDATE sessions SET llm_cost_usd = ? WHERE id = ?",
            (cost_usd, session_id),
        )
        self.conn.commit()

    def update_session_retry_budget(self, session_id: str, retry_budget: int) -> None:
        self.conn.execute(
            "UPDATE sessions SET retry_budget = ? WHERE id = ?",
            (retry_budget, session_id),
        )
        self.conn.commit()

    def increment_iteration(self, session_id: str) -> int:
        self.conn.execute(
            "UPDATE sessions SET iteration_count = iteration_count + 1 WHERE id = ?",
            (session_id,),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT iteration_count FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row["iteration_count"]

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            name=row["name"],
            branch=row["branch"],
            base_sha=row["base_sha"],
            baseline_run_id=row["baseline_run_id"],
            status=row["status"],
            termination_reason=(
                TerminationReason(row["termination_reason"])
                if row["termination_reason"]
                else None
            ),
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=(
                datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None
            ),
            budget=Budget(
                type=row["budget_type"],
                value=row["budget_value"],
                used=row["budget_used"],
            ),
            iteration_count=row["iteration_count"],
            pr_url=row["pr_url"],
            llm_cost_usd=row["llm_cost_usd"],
            retry_budget=row["retry_budget"],
            exported_at=(
                datetime.fromisoformat(row["exported_at"]) if row["exported_at"] else None
            ),
        )

    # Run management

    def create_run(
        self,
        session_id: str,
        config_json: str,
        iteration: int,
    ) -> str:
        run_id = str(uuid.uuid4())[:8]
        self.conn.execute(
            """
            INSERT INTO runs (id, session_id, iteration_number, config_json, status, started_at)
            VALUES (?, ?, ?, ?, 'running', ?)
            """,
            (run_id, session_id, iteration, config_json, datetime.now().isoformat()),
        )
        self.conn.commit()
        return run_id

    def set_run_status(self, run_id: str, status: str) -> None:
        updates = {"status": status}
        if status in ("completed", "failed"):
            updates["ended_at"] = datetime.now().isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        self.conn.execute(
            f"UPDATE runs SET {set_clause} WHERE id = ?",
            (*updates.values(), run_id),
        )
        self.conn.commit()

    def set_run_commit(self, run_id: str, sha: str) -> None:
        self.conn.execute(
            "UPDATE runs SET git_sha = ? WHERE id = ?",
            (sha, run_id),
        )
        self.conn.commit()

    def set_run_exit_code(self, run_id: str, exit_code: int) -> None:
        self.conn.execute(
            "UPDATE runs SET exit_code = ? WHERE id = ?",
            (exit_code, run_id),
        )
        self.conn.commit()

    def log_params(self, run_id: str, params: dict) -> None:
        for key, value in params.items():
            self.conn.execute(
                "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)",
                (run_id, key, json.dumps(value)),
            )
        self.conn.commit()

    def log_metrics(self, run_id: str, metrics: dict, step: int | None = None) -> None:
        for name, value in metrics.items():
            self.conn.execute(
                "INSERT INTO metrics (run_id, name, value, step) VALUES (?, ?, ?, ?)",
                (run_id, name, value, step),
            )
        self.conn.commit()

    def log_artifact(self, run_id: str, kind: str, path: str, size_bytes: int | None = None) -> str:
        artifact_id = str(uuid.uuid4())[:8]
        self.conn.execute(
            "INSERT INTO artifacts (id, run_id, kind, path, size_bytes) VALUES (?, ?, ?, ?, ?)",
            (artifact_id, run_id, kind, path, size_bytes),
        )
        self.conn.commit()
        return artifact_id

    def query_runs(
        self,
        session_id: str | None = None,
        branch: str | None = None,
        limit: int = 10,
    ) -> list[Run]:
        query = "SELECT r.* FROM runs r"
        params: list = []

        if branch:
            query += " JOIN sessions s ON r.session_id = s.id WHERE s.branch = ?"
            params.append(branch)
        elif session_id:
            query += " WHERE r.session_id = ?"
            params.append(session_id)

        query += " ORDER BY r.started_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_run(row) for row in rows]

    def get_run(self, run_id: str) -> Run | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def get_baseline_run(self, session_id: str) -> Run | None:
        session = self.get_session(session_id)
        if session is None or session.baseline_run_id is None:
            return None
        return self.get_run(session.baseline_run_id)

    def get_run_metrics(self, run_id: str) -> list[Metric]:
        rows = self.conn.execute(
            "SELECT * FROM metrics WHERE run_id = ? ORDER BY logged_at",
            (run_id,),
        ).fetchall()
        return [
            Metric(
                name=row["name"],
                value=row["value"],
                step=row["step"],
                logged_at=datetime.fromisoformat(row["logged_at"]),
            )
            for row in rows
        ]

    def get_run_artifacts(self, run_id: str) -> list[Artifact]:
        rows = self.conn.execute(
            "SELECT * FROM artifacts WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return [
            Artifact(
                id=row["id"],
                run_id=row["run_id"],
                kind=row["kind"],
                path=row["path"],
                size_bytes=row["size_bytes"],
                uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
            )
            for row in rows
        ]

    def _row_to_run(self, row: sqlite3.Row) -> Run:
        # Handle optional new columns that may not exist in older DBs
        row_dict = dict(row)
        return Run(
            id=row["id"],
            session_id=row["session_id"],
            iteration_number=row["iteration_number"],
            config_json=row["config_json"],
            git_sha=row["git_sha"],
            status=row["status"],
            started_at=(
                datetime.fromisoformat(row["started_at"]) if row["started_at"] else None
            ),
            ended_at=(
                datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None
            ),
            exit_code=row["exit_code"],
            change_type=row_dict.get("change_type"),
            change_description=row_dict.get("change_description"),
            change_diff=row_dict.get("change_diff"),
            hypothesis=row_dict.get("hypothesis"),
            metrics_json=row_dict.get("metrics_json"),
            outcome=row_dict.get("outcome"),
            analysis=row_dict.get("analysis"),
        )

    # Decision tracking

    def attach_decision(self, run_id: str, decision: Decision) -> str:
        decision_id = str(uuid.uuid4())[:8]
        self.conn.execute(
            """
            INSERT INTO decisions (id, run_id, action_type, rationale, commit_sha)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                run_id,
                decision.action_type,
                decision.rationale,
                decision.commit_sha,
            ),
        )
        self.conn.commit()
        return decision_id

    def update_decision_commit(self, decision_id: str, commit_sha: str) -> None:
        self.conn.execute(
            "UPDATE decisions SET commit_sha = ? WHERE id = ?",
            (commit_sha, decision_id),
        )
        self.conn.commit()

    def get_decisions(self, run_id: str) -> list[Decision]:
        rows = self.conn.execute(
            "SELECT * FROM decisions WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return [
            Decision(
                action_type=row["action_type"],
                rationale=row["rationale"],
                commit_sha=row["commit_sha"],
            )
            for row in rows
        ]

    # Trace logging

    def log_trace(self, run_id: str, event_type: str, data: dict) -> None:
        trace_id = str(uuid.uuid4())[:8]
        self.conn.execute(
            "INSERT INTO traces (id, run_id, event_type, data_json) VALUES (?, ?, ?, ?)",
            (trace_id, run_id, event_type, json.dumps(data)),
        )
        self.conn.commit()

    def get_traces(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT timestamp, event_type, data_json FROM traces WHERE run_id = ? ORDER BY timestamp",
            (run_id,),
        ).fetchall()
        return [
            {"timestamp": row[0], "event_type": row[1], "data": json.loads(row[2])}
            for row in rows
        ]

    # Iteration tracking (new fields on runs)

    def update_run_change(
        self,
        run_id: str,
        change_type: str,
        change_description: str,
        change_diff: str | None = None,
        hypothesis: str | None = None,
    ) -> None:
        """Update a run with change tracking info."""
        self.conn.execute(
            """
            UPDATE runs SET change_type = ?, change_description = ?,
                           change_diff = ?, hypothesis = ?
            WHERE id = ?
            """,
            (change_type, change_description, change_diff, hypothesis, run_id),
        )
        self.conn.commit()

    def update_run_results(
        self,
        run_id: str,
        metrics_json: str,
        outcome: str,
        analysis: str | None = None,
    ) -> None:
        """Update a run with results info."""
        self.conn.execute(
            """
            UPDATE runs SET metrics_json = ?, outcome = ?, analysis = ?
            WHERE id = ?
            """,
            (metrics_json, outcome, analysis, run_id),
        )
        self.conn.commit()

    # Suggestion management

    def create_suggestion(
        self,
        session_id: str,
        suggestion_type: str,
        content: str,
        run_id: str | None = None,
    ) -> int:
        """Create a new suggestion."""
        cursor = self.conn.execute(
            """
            INSERT INTO suggestions (session_id, run_id, suggestion_type, content)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, run_id, suggestion_type, content),
        )
        self.conn.commit()
        return cursor.lastrowid

    def update_suggestion_status(
        self,
        suggestion_id: int,
        status: str,
        handed_off_to: str | None = None,
    ) -> None:
        """Update suggestion status."""
        self.conn.execute(
            "UPDATE suggestions SET status = ?, handed_off_to = ? WHERE id = ?",
            (status, handed_off_to, suggestion_id),
        )
        self.conn.commit()

    def get_suggestions(
        self,
        session_id: str,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Suggestion]:
        """Get suggestions for a session."""
        query = "SELECT * FROM suggestions WHERE session_id = ?"
        params: list = [session_id]

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_suggestion(row) for row in rows]

    def get_pending_suggestions(self, session_id: str) -> list[Suggestion]:
        """Get pending suggestions for a session."""
        return self.get_suggestions(session_id, status="pending")

    def _row_to_suggestion(self, row: sqlite3.Row) -> Suggestion:
        return Suggestion(
            id=row["id"],
            session_id=row["session_id"],
            run_id=row["run_id"],
            created_at=(
                datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
            ),
            suggestion_type=row["suggestion_type"],
            content=row["content"],
            status=row["status"],
            handed_off_to=row["handed_off_to"],
        )
