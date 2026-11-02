import json
from collections.abc import Callable

from .llm import complete_json
from .models import Syllabus

SYSTEM_PROMPT = """You extract graded deliverables from university course syllabi.

Return only JSON matching this schema:
{schema}

Rules:
- Include every item that counts toward the final grade: assignments, quizzes, exams, projects,
  graded labs, and ongoing marks such as participation or attendance (type "other").
- Leave out ungraded work and extra credit.
- due_date must be YYYY-MM-DD. If the syllabus omits the year, assume {year}.
  Use null if no exact date is given, including "TBA", a date range, or only a week or
  class number. If an item has a draft and a final version, use the final version's date.
- weight is the percent of the final grade as a number (25% -> 25). Use null if not stated.
  If grades are in points, convert them to percent of the total points.
- If one weight covers a group with a known number of items (e.g. "4 assignments, 20% total"),
  list each item and split the weight evenly. Do the same for "best 7 of 8": list all 8.
- If the number of items isn't stated (e.g. "weekly quizzes, 15%"), or one line combines
  different activities, use a single entry with the whole weight.
- Keep titles short, with no description: "Assignment 1", "Midterm Exam".
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
