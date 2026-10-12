import litellm

from .config import settings

# Keep errors to one line; LiteLLM otherwise prints help banners around every failure.
litellm.suppress_debug_info = True


def chat_json(system: str, user: str, model: str | None = None) -> litellm.ModelResponse:
    """Send a prompt to any LiteLLM-supported model and return the full response.

    Use this when you need more than the text, such as token usage or cost in evals.
    """
    return litellm.completion(
        model=model or settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        # Retries with backoff on rate limits and temporary outages, which free tiers hit
        # often. Without this, evals would score a busy provider as a model failure.
        num_retries=3,
    )


def complete_json(system: str, user: str, model: str | None = None) -> str:
    """Send a prompt to any LiteLLM-supported model and return its raw JSON text."""
    return chat_json(system, user, model).choices[0].message.content
