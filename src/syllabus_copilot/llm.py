import litellm

from .config import settings


def complete_json(system: str, user: str, model: str | None = None) -> str:
    """Send a prompt to any LiteLLM-supported model and return its raw JSON text."""
    response = litellm.completion(
        model=model or settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return response.choices[0].message.content
