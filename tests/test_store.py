"""Tests for RunStore."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from revis.store.sqlite import SQLiteRunStore
from revis.types import Budget, Decision, TerminationReason


@pytest.fixture
def store():
    """Create a temporary SQLite store."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    store = SQLiteRunStore(db_path)
    store.initialize()
    yield store
    store.close()
    db_path.unlink()


class TestSessionManagement:
    def test_create_session(self, store):
        budget = Budget(type="time", value=3600)
        session_id = store.create_session(
            branch="revis/test-session",
            base_sha="abc123",
            budget=budget,
        )

        assert session_id is not None
        assert len(session_id) == 8

    def test_get_session(self, store):
        budget = Budget(type="runs", value=10)
        session_id = store.create_session(
            branch="revis/test",
            base_sha="def456",
            budget=budget,
        )

        session = store.get_session(session_id)

        assert session is not None
        assert session.id == session_id
        assert session.branch == "revis/test"
        assert session.base_sha == "def456"
        assert session.budget.type == "runs"
        assert session.budget.value == 10
        assert session.status == "running"

    def test_get_running_session(self, store):
        budget = Budget(type="time", value=7200)
        session_id = store.create_session(
            branch="revis/running",
            base_sha="123abc",
            budget=budget,
        )

        running = store.get_running_session()

        assert running is not None
        assert running.id == session_id

    def test_no_running_session(self, store):
        running = store.get_running_session()
        assert running is None

    def test_end_session(self, store):
        budget = Budget(type="time", value=3600)
        session_id = store.create_session(
            branch="revis/ending",
            base_sha="xyz789",
            budget=budget,
        )

        store.end_session(
            session_id,
            reason=TerminationReason.TARGET_ACHIEVED,
            pr_url="https://github.com/test/repo/pull/1",
        )

        session = store.get_session(session_id)
        assert session.status == "completed"
        assert session.termination_reason == TerminationReason.TARGET_ACHIEVED
        assert session.pr_url == "https://github.com/test/repo/pull/1"

    def test_update_budget(self, store):
        budget = Budget(type="time", value=3600)
        session_id = store.create_session(
            branch="revis/budget",
            base_sha="aaa111",
            budget=budget,
        )

        store.update_session_budget(session_id, 1800)

        session = store.get_session(session_id)
        assert session.budget.used == 1800

    def test_increment_iteration(self, store):
        budget = Budget(type="runs", value=10)
        session_id = store.create_session(
            branch="revis/iter",
            base_sha="bbb222",
            budget=budget,
        )

        count1 = store.increment_iteration(session_id)
        count2 = store.increment_iteration(session_id)

        assert count1 == 1
        assert count2 == 2


class TestRunManagement:
    def test_create_run(self, store):
        budget = Budget(type="runs", value=5)
        session_id = store.create_session(
            branch="revis/runs",
            base_sha="ccc333",
            budget=budget,
        )

        run_id = store.create_run(
            session_id=session_id,
            config_json='{"lr": 0.001}',
            iteration=1,
        )

        assert run_id is not None
        assert len(run_id) == 8

    def test_get_run(self, store):
        budget = Budget(type="runs", value=5)
        session_id = store.create_session(
            branch="revis/getrun",
            base_sha="ddd444",
            budget=budget,
        )

        run_id = store.create_run(
            session_id=session_id,
            config_json='{"batch_size": 32}',
            iteration=1,
        )

        run = store.get_run(run_id)

        assert run is not None
        assert run.id == run_id
        assert run.session_id == session_id
        assert run.iteration_number == 1
        assert run.config_json == '{"batch_size": 32}'
        assert run.status == "running"

    def test_set_run_status(self, store):
        budget = Budget(type="runs", value=5)
        session_id = store.create_session(
            branch="revis/status",
            base_sha="eee555",
            budget=budget,
        )

        run_id = store.create_run(
            session_id=session_id,
            config_json="{}",
            iteration=1,
        )

        store.set_run_status(run_id, "completed")

        run = store.get_run(run_id)
        assert run.status == "completed"
        assert run.ended_at is not None

    def test_query_runs(self, store):
        budget = Budget(type="runs", value=10)
        session_id = store.create_session(
            branch="revis/query",
            base_sha="fff666",
            budget=budget,
        )

        for i in range(5):
            store.create_run(
                session_id=session_id,
                config_json=f'{{"iteration": {i}}}',
                iteration=i + 1,
            )

        runs = store.query_runs(session_id=session_id, limit=3)

        assert len(runs) == 3

    def test_log_metrics(self, store):
        budget = Budget(type="runs", value=5)
        session_id = store.create_session(
            branch="revis/metrics",
            base_sha="ggg777",
            budget=budget,
        )

        run_id = store.create_run(
            session_id=session_id,
            config_json="{}",
            iteration=1,
        )

        store.log_metrics(run_id, {"loss": 0.5, "accuracy": 0.95}, step=100)

        metrics = store.get_run_metrics(run_id)
        assert len(metrics) == 2
        assert any(m.name == "loss" and m.value == 0.5 for m in metrics)


class TestDecisionTracking:
    def test_attach_decision(self, store):
        budget = Budget(type="runs", value=5)
        session_id = store.create_session(
            branch="revis/decisions",
            base_sha="hhh888",
            budget=budget,
        )

        run_id = store.create_run(
            session_id=session_id,
            config_json="{}",
            iteration=1,
        )

        decision = Decision(
            action_type="code_patch",
            rationale="Increased LR due to slow convergence",
        )

        decision_id = store.attach_decision(run_id, decision)

        assert decision_id is not None

        decisions = store.get_decisions(run_id)
        assert len(decisions) == 1
        assert decisions[0].action_type == "code_patch"
        assert "LR" in decisions[0].rationale

    def test_update_decision_commit(self, store):
        budget = Budget(type="runs", value=5)
        session_id = store.create_session(
            branch="revis/commit",
            base_sha="iii999",
            budget=budget,
        )

        run_id = store.create_run(
            session_id=session_id,
            config_json="{}",
            iteration=1,
        )

        decision = Decision(
            action_type="code_patch",
            rationale="Test commit tracking",
        )

        decision_id = store.attach_decision(run_id, decision)
        store.update_decision_commit(decision_id, "abc123def456")

        decisions = store.get_decisions(run_id)
        assert decisions[0].commit_sha == "abc123def456"
