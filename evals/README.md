# Evals

Measures how well deadline extraction works, so prompt and model changes are judged by
numbers instead of by eye.

```bash
syllabus-copilot eval                                   # the model in LLM_MODEL
syllabus-copilot eval --model gemini/gemini-3.8-flash \
                      --model groq/openai/gpt-oss-120b
```

Each run prints a comparison table and saves per-syllabus results to
`evals/results/<time>_<model>_<prompt version>.json`. The prompt version is a hash of the
system prompt, so results from different prompt versions are never mixed up.

## Metrics

Predicted deliverables are matched one-to-one to labeled ones by title (numbers must
agree, so "Assignment 1" never matches "Assignment 2").

| Metric | Meaning |
|---|---|
| Precision | Of the items the model returned, the share that are real graded items |
| Recall | Of the real graded items, the share the model found |
| Date / weight / type accuracy | On matched items only: due date exact, weight within 0.5 points, type exact |
| Errors | Syllabi where extraction failed outright (bad JSON, schema violation, API error); these count as zero items found |

Scores are micro-averaged: every deliverable counts equally, across all syllabi.

## Adding a golden syllabus

1. Put the syllabus in `golden/` as `<id>.pdf`, or as `<id>.txt` if you've pasted the text.
   Remove names, emails and other personal details first.
2. Write `golden/<id>.json` **by hand from the syllabus, not from model output**:
   ```json
   {"year": 2026, "expected": {"course_code": "...", "course_name": "...", "deliverables": [...]}}
   ```
3. Run `pytest`: it checks every label file against the schema.

### Labeling rules

Labels must follow the same rules as the prompt, or the scores measure disagreements
between labels instead of model errors.

- One entry per graded item. A group weight is split evenly ("4 assignments, 20%" gives
  four entries at 5% each), named `<Kind> <n>` (`Assignment 1`, `Lab 3`).
- Leave out ungraded items (practice sets, optional readings, ungraded check-ins).
- Participation and similar ongoing marks are type `other` with a `null` date.
- `due_date` is `null` when the syllabus says TBA, "during the exam period", or gives no date.
- `weight` is `null` when the syllabus gives no weight.

## Golden set

| ID | Source | What it tests |
|---|---|---|
| `synthetic-cs101` | Synthetic | Grading table, group weights, dates without a year, TBA final, an ungraded item |
| `synthetic-stat200` | Synthetic | Weights and dates in prose, a multi-part project, an undated presentation |
| `synthetic-hist210` | Synthetic | Dates only in the weekly schedule, an ungraded milestone |

The synthetic syllabi check that the harness works. Report metrics only from a set that
includes real syllabi.
