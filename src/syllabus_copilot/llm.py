import re
import time

import litellm

from .config import settings

# Keep errors to one line; LiteLLM otherwise prints help banners around every failure.
litellm.suppress_debug_info = True

# Free tiers cap tokens per minute, and a long syllabus can use half the budget in one call.
RATE_LIMIT_RETRIES = 5
DEFAULT_RATE_LIMIT_WAIT_S = 20.0


def _suggested_wait(error: Exception) -> float | None:
    """Seconds to wait before retrying, or None if a retry can't succeed any time soon."""
    text = str(error)
    if re.search(r"per day|retry in \d+h", text, re.IGNORECASE):
        return None  # A daily quota: waiting hours in a retry loop would only look like a hang.
    match = re.search(r"(?:try again|retry) in ([\d.]+)s", text, re.IGNORECASE)
    return float(match.group(1)) + 1 if match else DEFAULT_RATE_LIMIT_WAIT_S


def chat_json(
    system: str, user: str, model: str | None = None, sleep=time.sleep
) -> litellm.ModelResponse:
    """Send a prompt to any LiteLLM-supported model and return the full response.

    Use this when you need more than the text, such as token usage or cost in evals.
    Rate limits are retried after the wait the provider asks for, so evals on free tiers
    run slower instead of scoring a busy provider as a model failure.
    """
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return litellm.completion(
                model=model or settings.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                # Retries temporary outages such as 503 "high demand".
                num_retries=3,
            )
        except litellm.RateLimitError as error:
            wait = _suggested_wait(error)
            if wait is None or attempt == RATE_LIMIT_RETRIES:
                raise
            sleep(wait)


def complete_json(system: str, user: str, model: str | None = None) -> str:
    """Send a prompt to any LiteLLM-supported model and return its raw JSON text."""
    return chat_json(system, user, model).choices[0].message.content
