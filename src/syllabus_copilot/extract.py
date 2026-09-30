import json
from collections.abc import Callable

from .llm import complete_json
from .models import Syllabus

SYSTEM_PROMPT = """You extract graded deliverables from university course syllabi.

Return only JSON matching this schema:
{schema}

Rules:
- Include every graded item: assignments, quizzes, exams, projects, and graded labs.
- due_date must be YYYY-MM-DD. If the syllabus omits the year, assume {year}.
  If no date is given, use null.
- weight is the percent of the final grade as a number (25% -> 25). Use null if not stated.
- If one weight covers a group (e.g. "4 assignments, 20% total"), list each item
  and split the weight evenly.
- Never invent items that are not in the text."""

CompleteFn = Callable[[str, str], str]


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def extract_syllabus(text: str, year: int, complete: CompleteFn = complete_json) -> Syllabus:
    """Turn raw syllabus text into validated, structured deliverables.

    `complete` is injectable so tests (and evals comparing models) can swap the LLM.
    """
    system = SYSTEM_PROMPT.format(schema=json.dumps(Syllabus.model_json_schema()), year=year)
    raw = complete(system, text)
    return Syllabus.model_validate_json(_strip_code_fences(raw))
