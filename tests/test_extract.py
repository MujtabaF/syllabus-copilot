import json
from datetime import date

import pytest
from pydantic import ValidationError

from syllabus_copilot.extract import extract_syllabus
from syllabus_copilot.models import DeliverableType

SAMPLE = {
    "course_code": "CMPT 276",
    "course_name": "Intro to Software Engineering",
    "deliverables": [
        {"title": "Midterm", "type": "exam", "due_date": "2026-10-21", "weight": 25},
        {"title": "Assignment 1", "type": "assignment", "due_date": None, "weight": 5},
    ],
}


def fake_llm(response: str):
    """A stand-in for the real LLM that records the prompt and returns a canned reply."""
    calls = []

    def complete(system: str, user: str) -> str:
        calls.append((system, user))
        return response

    complete.calls = calls
    return complete


def test_parses_valid_json():
    syllabus = extract_syllabus("text", year=2026, complete=fake_llm(json.dumps(SAMPLE)))

    assert syllabus.course_code == "CMPT 276"
    midterm = syllabus.deliverables[0]
    assert midterm.type is DeliverableType.EXAM
    assert midterm.due_date == date(2026, 10, 21)
    assert syllabus.deliverables[1].due_date is None


def test_strips_markdown_code_fences():
    fenced = f"```json\n{json.dumps(SAMPLE)}\n```"
    syllabus = extract_syllabus("text", year=2026, complete=fake_llm(fenced))
    assert len(syllabus.deliverables) == 2


def test_rejects_impossible_weight():
    bad = {**SAMPLE, "deliverables": [{"title": "Final", "type": "exam", "weight": 150}]}
    with pytest.raises(ValidationError):
        extract_syllabus("text", year=2026, complete=fake_llm(json.dumps(bad)))


def test_prompt_includes_year_and_syllabus_text():
    llm = fake_llm(json.dumps(SAMPLE))
    extract_syllabus("Midterm on Oct 21", year=2027, complete=llm)

    system, user = llm.calls[0]
    assert "2027" in system
    assert user == "Midterm on Oct 21"
