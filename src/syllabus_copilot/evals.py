"""Score deadline extraction against a hand-labeled golden dataset.

Each golden example is a pair of files in the golden directory:
  <id>.txt or <id>.pdf   the syllabus
  <id>.json              {"year": 2026, "expected": <Syllabus JSON>}

Predicted deliverables are matched one-to-one to expected ones by title. Precision and
recall measure whether the right items were found; date, weight and type accuracy are
measured on the matched pairs only, so a missed item is not counted twice.
"""

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel

from .extract import SYSTEM_PROMPT, CompleteFn, extract_syllabus
from .models import Deliverable, Syllabus
from .pdf import pdf_to_text

PROMPT_VERSION = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8]

# Titles below this similarity are never matched, e.g. "Midterm" vs "Final Exam".
MIN_TITLE_SIMILARITY = 0.6
# A weight within this many percentage points counts as correct (allows 33.33 vs 33.3).
WEIGHT_TOLERANCE = 0.5


class GoldenExample(BaseModel):
    id: str
    labeled_by: str = "human"
    year: int
    text: str
    expected: Syllabus


def load_golden(directory: Path, real_only: bool = False) -> list[GoldenExample]:
    """Load every labeled example. Label files marked `"todo": true` are skipped."""
    examples = []
    for label_path in sorted(directory.glob("*.json")):
        label = json.loads(label_path.read_text())
        if label.get("todo") or (real_only and label_path.stem.startswith("synthetic-")):
            continue
        pdf_path = label_path.with_suffix(".pdf")
        if pdf_path.exists():
            text = pdf_to_text(pdf_path)
        else:
            text = label_path.with_suffix(".txt").read_text()
        examples.append(
            GoldenExample(
                id=label_path.stem,
                labeled_by=label.get("labeled_by", "human"),
                year=label["year"],
                text=text,
                expected=label["expected"],
            )
        )
    return examples


def unlabeled(directory: Path) -> list[str]:
    """IDs of label files still marked `"todo": true`."""
    return [
        p.stem for p in sorted(directory.glob("*.json")) if json.loads(p.read_text()).get("todo")
    ]


# Spellings that mean the same thing, applied word by word before titles are compared.
_SYNONYMS = {"hw": "homework", "pset": "problem set", "i": "1", "ii": "2", "iii": "3", "iv": "4"}


def _normalize(title: str) -> str:
    text = re.sub(r"[^a-z0-9]", " ", title.lower())
    text = re.sub(r"(?<=[a-z])(?=\d)", " ", text)  # "hw01" -> "hw 01"
    words = [_SYNONYMS.get(word, word) for word in text.split()]
    return re.sub(r"\b0+(?=\d)", "", " ".join(words))  # "01" -> "1"


def title_similarity(a: str, b: str) -> float:
    """How alike two titles are, from 0 to 1.

    Titles with different numbers ("Assignment 1" vs "Assignment 2") never match, because
    plain string similarity would rate them as near-identical.
    """
    a, b = _normalize(a), _normalize(b)
    if re.findall(r"\d+", a) != re.findall(r"\d+", b):
        return 0.0
    # A title plus a description is the same item: "Project 3" and "Project 3: Final Report".
    shorter, longer = sorted((a, b), key=len)
    if len(shorter.split()) >= 2 and longer.startswith(f"{shorter} "):
        return 0.95
    return SequenceMatcher(None, a, b).ratio()


def match_deliverables(
    predicted: list[Deliverable], expected: list[Deliverable]
) -> list[tuple[Deliverable, Deliverable]]:
    """Pair predicted with expected items one-to-one, best title matches first.

    A correct due date breaks ties between similar titles.
    """
    candidates = []
    for i, p in enumerate(predicted):
        for j, e in enumerate(expected):
            sim = title_similarity(p.title, e.title)
            if sim >= MIN_TITLE_SIMILARITY:
                bonus = 0.1 if p.due_date == e.due_date else 0.0
                candidates.append((sim + bonus, i, j))

    pairs, used_p, used_e = [], set(), set()
    for _, i, j in sorted(candidates, reverse=True):
        if i not in used_p and j not in used_e:
            used_p.add(i)
            used_e.add(j)
            pairs.append((predicted[i], expected[j]))
    return pairs


def _weight_correct(p: float | None, e: float | None) -> bool:
    if p is None or e is None:
        return p is e
    return abs(p - e) <= WEIGHT_TOLERANCE


@dataclass
class Counts:
    """Raw tallies, so per-syllabus results can be summed into a micro-average."""

    predicted: int = 0
    expected: int = 0
    matched: int = 0
    date_correct: int = 0
    weight_correct: int = 0
    type_correct: int = 0
    dated: int = 0  # matched items whose label has a real date
    dated_correct: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(**{k: v + getattr(other, k) for k, v in asdict(self).items()})

    def metrics(self) -> dict[str, float]:
        def ratio(n: int, d: int) -> float:
            return n / d if d else 0.0

        precision = ratio(self.matched, self.predicted)
        recall = ratio(self.matched, self.expected)
        f1 = ratio(2 * precision * recall, precision + recall) if precision + recall else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "date_accuracy": ratio(self.date_correct, self.matched),
            # date_accuracy also counts "no date" correctly left empty; this one doesn't.
            "dated_item_accuracy": ratio(self.dated_correct, self.dated),
            "weight_accuracy": ratio(self.weight_correct, self.matched),
            "type_accuracy": ratio(self.type_correct, self.matched),
        }


def score(predicted: Syllabus, expected: Syllabus) -> Counts:
    pairs = match_deliverables(predicted.deliverables, expected.deliverables)
    return Counts(
        predicted=len(predicted.deliverables),
        expected=len(expected.deliverables),
        matched=len(pairs),
        date_correct=sum(p.due_date == e.due_date for p, e in pairs),
        weight_correct=sum(_weight_correct(p.weight, e.weight) for p, e in pairs),
        type_correct=sum(p.type == e.type for p, e in pairs),
        dated=sum(e.due_date is not None for _, e in pairs),
        dated_correct=sum(e.due_date is not None and p.due_date == e.due_date for p, e in pairs),
    )


def differences(predicted: Syllabus, expected: Syllabus) -> tuple[list[str], list[str], list[str]]:
    """What went wrong, for reading results: (missed, extra, wrong fields on matched items)."""
    pairs = match_deliverables(predicted.deliverables, expected.deliverables)
    matched_p = {id(p) for p, _ in pairs}
    matched_e = {id(e) for _, e in pairs}
    wrong = []
    for p, e in pairs:
        if p.due_date != e.due_date:
            wrong.append(f"{e.title}: date {p.due_date}, expected {e.due_date}")
        if not _weight_correct(p.weight, e.weight):
            wrong.append(f"{e.title}: weight {p.weight}, expected {e.weight}")
        if p.type != e.type:
            wrong.append(f"{e.title}: type {p.type.value}, expected {e.type.value}")
    return (
        [e.title for e in expected.deliverables if id(e) not in matched_e],
        [p.title for p in predicted.deliverables if id(p) not in matched_p],
        wrong,
    )


@dataclass
class ExampleResult:
    id: str
    counts: Counts
    latency_s: float
    error: str | None = None
    missed: list[str] = field(default_factory=list)  # expected, but not found
    extra: list[str] = field(default_factory=list)  # found, but not expected
    wrong_fields: list[str] = field(default_factory=list)  # matched, with a wrong field


@dataclass
class RunResult:
    model: str
    prompt_version: str
    examples: list[ExampleResult] = field(default_factory=list)
    cost_usd: float = 0.0

    @property
    def totals(self) -> Counts:
        return sum((r.counts for r in self.examples), Counts())

    def summary(self) -> dict:
        latencies = [r.latency_s for r in self.examples]
        return {
            "model": self.model,
            "prompt_version": self.prompt_version,
            "examples": len(self.examples),
            "errors": sum(r.error is not None for r in self.examples),
            **self.totals.metrics(),
            "mean_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
            "cost_usd": self.cost_usd,
        }

    def to_json(self) -> dict:
        return {"summary": self.summary(), "examples": [asdict(r) for r in self.examples]}


def run_eval(
    examples: list[GoldenExample],
    model: str,
    complete: CompleteFn | None = None,
    on_example: Callable[[ExampleResult], None] | None = None,
) -> RunResult:
    """Run extraction over every example and score it.

    By default this calls `model` through LiteLLM and records cost. Pass `complete` to
    swap in a fake LLM, as the tests do.
    """
    result = RunResult(model=model, prompt_version=PROMPT_VERSION)
    waited = 0.0

    if complete is None:
        from .llm import chat_json

        def wait(seconds: float) -> None:  # Rate-limit waits don't count as model latency.
            nonlocal waited
            waited += seconds
            time.sleep(seconds)

        def complete(system: str, user: str) -> str:
            response = chat_json(system, user, model=model, sleep=wait)
            result.cost_usd += _cost(response, model)
            return response.choices[0].message.content

    for example in examples:
        waited = 0.0
        start = time.perf_counter()
        try:
            predicted = extract_syllabus(example.text, year=example.year, complete=complete)
            counts, error = score(predicted, example.expected), None
            missed, extra, wrong = differences(predicted, example.expected)
        except Exception as exc:  # A failed extraction scores as zero items found.
            empty = Syllabus(deliverables=[])
            counts, error = score(empty, example.expected), f"{type(exc).__name__}: {exc}"
            missed, extra, wrong = [e.title for e in example.expected.deliverables], [], []
        example_result = ExampleResult(
            id=example.id,
            counts=counts,
            latency_s=time.perf_counter() - start - waited,
            error=error,
            missed=missed,
            extra=extra,
            wrong_fields=wrong,
        )
        result.examples.append(example_result)
        if on_example:
            on_example(example_result)
    return result


def _cost(response, model: str) -> float:
    """Price a call by the model we asked for.

    The response's own model name drops the provider (Groq returns "openai/gpt-oss-120b"),
    which LiteLLM would then price as an OpenAI model, or fail to price at all.
    """
    import litellm

    usage = response.usage
    try:
        prompt, completion = litellm.cost_per_token(
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
    except Exception:  # Local models, and some free ones, have no price listed.
        return 0.0
    return prompt + completion


def format_table(summaries: list[dict]) -> str:
    """Render run summaries as a Markdown table for the terminal or the README."""
    header = (
        "| Model | Prompt | Precision | Recall | F1 | Date acc | Real-date acc | Weight acc "
        "| Type acc | Errors | Mean latency | Cost |"
    )
    lines = [header, "|---" * 12 + "|"]
    for s in summaries:
        lines.append(
            f"| {s['model']} | {s['prompt_version']} | {s['precision']:.1%} | {s['recall']:.1%} "
            f"| {s['f1']:.1%} | {s['date_accuracy']:.1%} | {s['dated_item_accuracy']:.1%} "
            f"| {s['weight_accuracy']:.1%} "
            f"| {s['type_accuracy']:.1%} | {s['errors']}/{s['examples']} "
            f"| {s['mean_latency_s']:.1f}s | ${s['cost_usd']:.4f} |"
        )
    return "\n".join(lines)
