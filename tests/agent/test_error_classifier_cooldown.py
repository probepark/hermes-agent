import time

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class _Response:
    def __init__(self, headers):
        self.headers = headers


class _ApiError(Exception):
    def __init__(self, message, *, body=None, headers=None):
        super().__init__(message)
        self.status_code = 429
        self.body = body or {}
        self.response = _Response(headers or {})


def test_model_cooldown_body_reset_at_sets_cooldown_and_reset_metadata():
    reset_at = time.time() + 90
    error = _ApiError(
        "model unavailable",
        body={"error": {"code": "model_cooldown", "reset_at": reset_at}},
    )

    result = classify_api_error(error, provider="openai", model="gpt-test")

    assert result.reason is FailoverReason.rate_limit
    assert result.should_fallback is True
    assert result.error_context["cooldown"] is True
    assert result.error_context["cooldown_kind"] == "model_cooldown"
    assert result.error_context["reset_seconds"] == pytest.approx(90, abs=1)
    assert result.error_context["reset_at"] == pytest.approx(reset_at, abs=1)


def test_credential_quota_retry_after_sets_cooldown_and_reset_metadata():
    error = _ApiError(
        "credential quota reached",
        body={"error": {"code": "credential_quota"}},
        headers={"Retry-After": "45"},
    )

    result = classify_api_error(error, provider="openai", model="gpt-test")

    assert result.reason is FailoverReason.rate_limit
    assert result.should_fallback is True
    assert result.error_context["cooldown"] is True
    assert result.error_context["cooldown_kind"] == "credential_quota"
    assert result.error_context["reset_seconds"] == 45
    assert result.error_context["reset_at"] == pytest.approx(time.time() + 45, abs=1)


def test_plain_429_remains_ordinary_rate_limit():
    result = classify_api_error(_ApiError("rate limit exceeded"), provider="openai")

    assert result.reason is FailoverReason.rate_limit
    assert result.should_fallback is True
    assert "cooldown" not in result.error_context
    assert "reset_seconds" not in result.error_context
    assert "reset_at" not in result.error_context
