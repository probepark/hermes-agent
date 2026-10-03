"""Regression coverage for cooldown errors routing through the real fallback chain."""

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import classify_api_error
from agent.turn_recovery import route_classified_error
from agent.turn_retry_state import TurnRetryState
from run_agent import AIAgent


class _Response:
    def __init__(self, headers=None):
        self.headers = headers or {}


class _ApiError(Exception):
    def __init__(self, message, *, body):
        super().__init__(message)
        self.status_code = 429
        self.body = body
        self.response = _Response()


def _client(base_url="http://127.0.0.1:1/v1"):
    client = MagicMock()
    client.api_key = "test-key"
    client.base_url = base_url
    client._custom_headers = {}
    return client


def _make_agent(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text(
        "custom_providers:\n"
        "  - name: cpa\n"
        "    base_url: http://127.0.0.1:1/v1\n"
        "    api_key: test-key\n",
        encoding="utf-8",
    )
    return AIAgent(
        api_key="test-key",
        base_url="http://127.0.0.1:1/v1",
        provider="cpa",
        model="claude-opus-5",
        fallback_model=[{"provider": "cpa", "model": "gpt-6-astra"}],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
    )


def _route(agent, classified):
    return route_classified_error(
        agent,
        RuntimeError(classified.message),
        classified,
        TurnRetryState(),
        error_msg=classified.message,
        error_context=classified.error_context,
        recovered_with_pool=False,
        base_url=agent.base_url,
        model=agent.model,
        messages=[],
        api_messages=[],
        system_message="",
        active_system_prompt="",
        conversation_history=[],
        retry_count=0,
        max_retries=3,
        compression_attempts=0,
        max_compression_attempts=1,
        api_call_count=1,
        effective_task_id=None,
    )


@pytest.mark.parametrize("code", ["model_cooldown", "credential_quota"])
def test_cooldown_errors_route_to_configured_fallback(code, tmp_path, monkeypatch):
    reset_at = time.time() + 90
    error = _ApiError(
        f"{code} response",
        body={"error": {"code": code, "reset_at": reset_at}},
    )
    with (
        patch("agent.process_bootstrap.OpenAI", return_value=_client()),
        patch("agent.auxiliary_client.OpenAI", return_value=_client()),
        patch("agent.turn_recovery.time.sleep") as sleep,
    ):
        agent = _make_agent(tmp_path, monkeypatch)
        classified = classify_api_error(error, provider=agent.provider, model=agent.model)
        verdict = _route(agent, classified)

    assert verdict.action == "break"
    assert agent.model == "gpt-6-astra"
    assert agent.provider == "cpa"
    assert classified.error_context["reset_at"] == pytest.approx(reset_at, abs=1)
    remaining = agent._rate_limited_until - time.monotonic()
    assert remaining == pytest.approx(reset_at - time.time(), abs=2)
    assert all(call.args[0] < 600 for call in sleep.call_args_list)
