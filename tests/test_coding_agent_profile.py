import json
import os
import sys
from pathlib import Path

import pytest

from hermes_cli import coding_agent_profile as cap


def test_codex_create_env_and_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(cap.ROOT_ENV, str(tmp_path))

    assert cap.main(["codex", "create", "shortfe", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    expected = tmp_path / "codex" / "shortfe"
    assert payload["provider"] == "codex"
    assert payload["label"] == "shortfe"
    assert payload["path"] == str(expected)
    assert payload["env"] == {"CODEX_HOME": str(expected)}
    assert payload["auth_status"] == "not_authenticated"
    assert expected.is_dir()

    state = json.loads((tmp_path / "state.json").read_text())
    assert state["profiles"]["codex/shortfe"]["env"] == {"CODEX_HOME": str(expected)}


def test_claude_create_uses_isolated_home(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(cap.ROOT_ENV, str(tmp_path))

    assert cap.main(["claude", "create", "hrm"]) == 0
    out = capsys.readouterr().out

    expected = tmp_path / "claude" / "hrm" / "home"
    assert f"claude/hrm\tnot_authenticated\tpath={expected}" in out
    assert expected.is_dir()
    assert (expected / ".claude").is_dir()


def test_env_and_login_commands_are_secret_free(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(cap.ROOT_ENV, str(tmp_path))

    assert cap.main(["codex", "env", "main"]) == 0
    env_out = capsys.readouterr().out.strip()
    assert env_out == f"export CODEX_HOME={tmp_path / 'codex' / 'main'}"

    assert cap.main(["claude", "login-command", "main"]) == 0
    login_out = capsys.readouterr().out.strip()
    assert login_out == f"HOME={tmp_path / 'claude' / 'main' / 'home'} claude auth login"
    assert "token" not in login_out.lower()


def test_run_sets_profile_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(cap.ROOT_ENV, str(tmp_path))
    script = (
        "import os, pathlib; "
        "pathlib.Path(os.environ['OUT']).write_text(os.environ['CODEX_HOME'])"
    )
    out_file = tmp_path / "out.txt"
    monkeypatch.setenv("OUT", str(out_file))

    assert cap.main(["codex", "run", "main", "--", sys.executable, "-c", script]) == 0
    assert out_file.read_text() == str(tmp_path / "codex" / "main")


def test_rotate_retries_only_retryable_failures(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(cap.ROOT_ENV, str(tmp_path))
    marker = tmp_path / "marker"
    script = """
import os, pathlib, sys
marker = pathlib.Path(os.environ['MARKER'])
if 'sub1' in os.environ.get('CODEX_HOME', ''):
    print('success from sub1')
    sys.exit(0)
print('usage limit reached', file=sys.stderr)
marker.write_text('main failed')
sys.exit(7)
""".strip()
    monkeypatch.setenv("MARKER", str(marker))

    code = cap.main([
        "codex",
        "rotate",
        "--profiles",
        "main,sub1",
        "--",
        sys.executable,
        "-c",
        script,
    ])
    captured = capsys.readouterr()

    assert code == 0
    assert marker.read_text() == "main failed"
    assert "success from sub1" in captured.out
    assert "retrying with codex/sub1" in captured.err


def test_init_pool_creates_default_four_accounts(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(cap.ROOT_ENV, str(tmp_path))

    assert cap.main(["init-pool", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    refs = [item["profile"] for item in payload]

    assert refs == ["codex/main", "codex/sub1", "claude/main", "claude/sub1"]
    assert (tmp_path / "codex" / "main").is_dir()
    assert (tmp_path / "codex" / "sub1").is_dir()
    assert (tmp_path / "claude" / "main" / "home").is_dir()
    assert (tmp_path / "claude" / "sub1" / "home").is_dir()
    assert {item["auth_status"] for item in payload} == {"not_authenticated"}


def test_recommend_changes_order_by_task_type(capsys):
    assert cap.main(["recommend", "--task", "coding"]) == 0
    assert capsys.readouterr().out.strip() == "codex/main -> codex/sub1 -> claude/main -> claude/sub1"

    assert cap.main(["recommend", "--task", "review"]) == 0
    assert capsys.readouterr().out.strip() == "claude/main -> claude/sub1 -> codex/main -> codex/sub1"


def test_run_policy_falls_back_across_codex_and_claude(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(cap.ROOT_ENV, str(tmp_path))
    log_file = tmp_path / "log.jsonl"
    script = """
import json, os, pathlib, sys
log = pathlib.Path(os.environ['LOG'])
profile = os.environ.get('CODEX_HOME') or os.environ.get('HOME')
with log.open('a') as f:
    f.write(json.dumps({'profile': profile}) + '\\n')
if os.environ.get('CODEX_HOME'):
    print('usage limit', file=sys.stderr)
    sys.exit(7)
print('claude ok')
sys.exit(0)
""".strip()
    monkeypatch.setenv("LOG", str(log_file))
    code = cap.main([
        "run-policy",
        "--profiles",
        "codex/main,claude/main",
        "--prompt",
        "ignored prompt",
        "--codex-command",
        f"{sys.executable} -c {cap.shlex.quote(script)}",
        "--claude-command",
        f"{sys.executable} -c {cap.shlex.quote(script)}",
    ])
    captured = capsys.readouterr()

    assert code == 0
    assert "retrying with claude/main" in captured.err
    assert "claude ok" in captured.out
    rows = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert rows[0]["profile"] == str(tmp_path / "codex" / "main")
    assert rows[1]["profile"] == str(tmp_path / "claude" / "main" / "home")


def test_run_policy_dry_run_is_secret_free(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(cap.ROOT_ENV, str(tmp_path))
    assert cap.main(["run-policy", "--task", "review", "--prompt", "hello", "--dry-run"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["profile"] for line in lines] == ["claude/main", "claude/sub1", "codex/main", "codex/sub1"]
    assert all("auth" not in " ".join(line["command"]).lower() for line in lines)


def test_invalid_label_rejected():
    with pytest.raises(SystemExit):
        cap.validate_label("../secret")
