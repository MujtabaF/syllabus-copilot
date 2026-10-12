import json
from datetime import date
from pathlib import Path

import pytest

from syllabus_copilot.evals import (
    load_golden,
    match_deliverables,
    run_eval,
    score,
    title_similarity,
    unlabeled,
)
from syllabus_copilot.models import Deliverable, Syllabus

GOLDEN_DIR = Path(__file__).parent.parent / "evals" / "golden"


def item(title, type="assignment", due=None, weight=None):
    return Deliverable(title=title, type=type, due_date=due, weight=weight)


def test_numbered_titles_never_match_different_numbers():
    assert title_similarity("Assignment 1", "Assignment 2") == 0.0
    assert title_similarity("Assignment 1", "assignment #1") > 0.9


def test_matches_reworded_titles_one_to_one():
    predicted = [item("Midterm"), item("Final Exam")]
    expected = [item("Midterm Exam"), item("Final Exam")]

    pairs = match_deliverables(predicted, expected)

    assert {(p.title, e.title) for p, e in pairs} == {
        ("Midterm", "Midterm Exam"),
        ("Final Exam", "Final Exam"),
    }


def test_score_counts_misses_extras_and_field_errors():
    expected = Syllabus(
        deliverables=[
            item("Quiz 1", "quiz", date(2026, 9, 25), 5),
            item("Quiz 2", "quiz", date(2026, 10, 16), 5),
            item("Midterm Exam", "exam", date(2026, 10, 28), 25),
        ]
    )
    predicted = Syllabus(
        deliverables=[
            item("Quiz 1", "quiz", date(2026, 9, 25), 5),
            item("Midterm Exam", "exam", date(2026, 10, 29), 25.3),  # wrong date
            item("Problem Set 1", "assignment"),  # ungraded, should not be there
        ]
    )

    m = score(predicted, expected).metrics()

    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["date_accuracy"] == pytest.approx(1 / 2)
    assert m["weight_accuracy"] == 1.0  # 25.3 is within tolerance
    assert m["type_accuracy"] == 1.0


def test_golden_labels_are_valid():
    examples = load_golden(GOLDEN_DIR)
    assert examples, "no golden syllabi found"
    for example in examples:
        assert example.text.strip(), f"{example.id} has no syllabus text"
        assert example.expected.deliverables, f"{example.id} has no labeled deliverables"


def test_todo_labels_are_skipped(tmp_path):
    (tmp_path / "done.txt").write_text("Midterm Oct 21")
    (tmp_path / "done.json").write_text(
        json.dumps(
            {"year": 2026, "expected": {"deliverables": [{"title": "Midterm", "type": "exam"}]}}
        )
    )
    (tmp_path / "pending.txt").write_text("Final TBA")
    (tmp_path / "pending.json").write_text(
        json.dumps({"todo": True, "year": 2026, "expected": {"deliverables": []}})
    )

    assert [ex.id for ex in load_golden(tmp_path)] == ["done"]
    assert unlabeled(tmp_path) == ["pending"]


def test_perfect_model_scores_100_percent():
    examples = load_golden(GOLDEN_DIR)
    answers = {ex.text: ex.expected.model_dump_json() for ex in examples}

    result = run_eval(examples, "fake", complete=lambda system, user: answers[user])

    summary = result.summary()
    assert summary["errors"] == 0
    for metric in ("precision", "recall", "date_accuracy", "weight_accuracy", "type_accuracy"):
        assert summary[metric] == 1.0, metric


def test_failed_extraction_counts_as_nothing_found():
    examples = load_golden(GOLDEN_DIR)[:1]

    result = run_eval(examples, "fake", complete=lambda system, user: "not json")

    summary = result.summary()
    assert summary["errors"] == 1
    assert summary["recall"] == 0.0
    assert json.dumps(result.to_json(), default=str)  # results stay serializable
