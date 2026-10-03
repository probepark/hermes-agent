from types import SimpleNamespace
from unittest.mock import patch

from agent.error_classifier import FailoverReason
from agent.turn_recovery import route_classified_error
from agent.turn_retry_state import TurnRetryState


class _Agent:
    compression_enabled = True
    provider = "openai"
    _fallback_index = 0
    _fallback_chain = [object()]
    _credential_pool = object()

    def __init__(self):
        self.statuses = []
        self.fallback_calls = []

    def _buffer_diagnostic_status(self, text):
        self.statuses.append(text)

    def _try_activate_fallback(self, **kwargs):
        self.fallback_calls.append(kwargs)
        return True


def _classified(*, cooldown=False):
    return SimpleNamespace(
        reason=FailoverReason.rate_limit,
        error_context={"cooldown": cooldown, "reset_at": 1234} if cooldown else {},
        billing_unverified=False,
        is_auth=False,
    )


def _route(agent, classified):
    return route_classified_error(
        agent,
        RuntimeError("rate limited"),
        classified,
        TurnRetryState(),
        error_msg="rate limited",
        error_context=classified.error_context,
        recovered_with_pool=False,
        base_url="https://example.test/v1",
        model="model",
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


def test_cooldown_bypasses_recoverable_pool_and_passes_reset_at():
    agent = _Agent()
    with (
        patch("run_agent._pool_may_recover_from_rate_limit", return_value=True) as pool_recover,
        patch("agent.conversation_loop._arm_fallback_restart", return_value="prompt"),
    ):
        verdict = _route(agent, _classified(cooldown=True))

    pool_recover.assert_not_called()
    assert verdict.action == "break"
    assert agent.fallback_calls == [{"reason": FailoverReason.rate_limit, "reset_at": 1234}]
    assert "cooldown" in agent.statuses[0].lower()
    assert not any(char.isdigit() for char in agent.statuses[0])


def test_cooldown_does_not_enter_long_backoff_when_fallback_exists():
    agent = _Agent()
    with (
        patch("run_agent._pool_may_recover_from_rate_limit", return_value=True),
        patch("agent.conversation_loop._arm_fallback_restart", return_value="prompt"),
        patch("agent.turn_recovery.time.sleep") as sleep,
    ):
        verdict = _route(agent, _classified(cooldown=True))

    assert verdict.action == "break"
    assert agent.fallback_calls
    assert all(call.args[0] < 600 for call in sleep.call_args_list)


def test_normal_rate_limit_retains_pool_recovery_behavior():
    agent = _Agent()
    with patch("run_agent._pool_may_recover_from_rate_limit", return_value=True) as pool_recover:
        verdict = _route(agent, _classified(cooldown=False))

    pool_recover.assert_called_once_with(agent._credential_pool)
    assert verdict.action == "fallthrough"
    assert agent.fallback_calls == []


def test_rate_limit_error_cooldown_fallback_records_safe_metadata():
    agent = _Agent()
    classified = _classified(cooldown=True)
    classified.error_context.update({"cooldown_kind": "rate_limit_error"})

    with patch("run_agent._pool_may_recover_from_rate_limit", return_value=True):
        verdict = _route(agent, classified)

    assert verdict.action == "break"
    assert agent.fallback_calls == [{"reason": FailoverReason.rate_limit, "reset_at": 1234}]
    assert agent._cooldown_recovery_metadata == {
        "cooldown": True,
        "reset_at": 1234,
        "cooldown_kind": "rate_limit_error",
        "model": "model",
        "caller": "conversation",
        "fallback_result": True,
    }
