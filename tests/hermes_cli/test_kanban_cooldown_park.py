import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


def test_cooldown_exit_is_parked_until_reset(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cooldown", assignee="a")
        host = kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, tid, claimer=f"{host}:w123")
        conn.execute("UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?", (123, int(time.time()) - 120, tid))
        conn.commit()
        log = kb.worker_log_path(tid)
        log.parent.mkdir(parents=True, exist_ok=True)
        reset = time.time() + 600
        log.write_text(f"{KANBAN_WORKER_EXIT_TRAILER}{kb.KANBAN_RATE_LIMIT_EXIT_CODE} reset_after={reset} cooldown_kind=model_cooldown\n")

        assert kbd.detect_crashed_workers(conn) == []
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        run = conn.execute("SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
        assert run["outcome"] == "cooldown_parked"
        metadata = kb._json_dict(run["metadata"])
        assert metadata["cooldown_kind"] == "model_cooldown"
        assert metadata["reset_after"] == pytest.approx(reset, abs=1e-3)
        assert kbd._protocol_violation_streak(conn, tid) == 0
        assert kbd.check_respawn_guard(conn, tid) == "cooldown_parked"
