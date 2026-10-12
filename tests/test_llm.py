import litellm
import pytest

from syllabus_copilot import llm


def rate_limit(message="Please try again in 15.01s."):
    return litellm.RateLimitError(message=message, llm_provider="groq", model="m")


def test_waits_as_long_as_the_provider_asks_then_retries(monkeypatch):
    calls, waits = [], []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise rate_limit()
        return "response"

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)

    assert llm.chat_json("system", "user", model="m", sleep=waits.append) == "response"
    assert len(calls) == 2
    assert waits == [pytest.approx(16.01)]


def test_gives_up_after_too_many_rate_limits(monkeypatch):
    def always_limited(**kwargs):
        raise rate_limit("no hint here")

    monkeypatch.setattr(llm.litellm, "completion", always_limited)
    waits = []

    with pytest.raises(litellm.RateLimitError):
        llm.chat_json("system", "user", model="m", sleep=waits.append)
    assert waits == [llm.DEFAULT_RATE_LIMIT_WAIT_S] * llm.RATE_LIMIT_RETRIES
