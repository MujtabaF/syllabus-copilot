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
| Real-date accuracy | Date accuracy on matched items whose label has a real date. Plain date accuracy also credits correctly leaving a date empty, which can hide wrong dates |
| Errors | Syllabi where extraction failed outright (bad JSON, schema violation, API error); these count as zero items found |

Scores are micro-averaged: every deliverable counts equally, across all syllabi.

## Adding a golden syllabus

1. Put the syllabus in `golden/` as `<id>.pdf`, or as `<id>.txt` if you've pasted the text.
   Remove names, emails and other personal details first. Use only syllabi you're allowed
   to share (your own courses, or openly licensed ones), and credit them in
   [`golden/SOURCES.md`](golden/SOURCES.md).
2. Write `golden/<id>.json` **by hand from the syllabus, not from model output**:
   ```json
   {"year": 2026, "expected": {"course_code": "...", "course_name": "...", "deliverables": [...]}}
   ```
3. Run `pytest`: it checks every label file against the schema.

Files marked `"todo": true` are skipped by the eval.

`"labeled_by": "claude-draft"` means the label was written by an AI reading the syllabus and
no person has checked it yet. After checking a file against its syllabus, delete that line.
Until then, describe any metrics as measured against AI-drafted labels, not hand labels.

### Labeling rules

Labels must follow the same rules as the prompt, or the scores measure disagreements
between labels instead of model errors.

- One entry per graded item. A group weight is split evenly ("4 assignments, 20%" gives
  four entries at 5% each), named `<Kind> <n>` (`Assignment 1`, `Lab 3`).
- Leave out ungraded items (practice sets, optional readings, ungraded check-ins) and
  extra credit.
- Participation and similar ongoing marks are type `other` with a `null` date.
- `due_date` is `null` when the syllabus says TBA, "during the exam period", or gives no date.
- `weight` is `null` when the syllabus gives no weight.
- Recurring work with no stated count ("reading questions", "weekly quizzes" with no
  number) is one entry carrying the whole group weight, with a `null` date.
- "Drop the lowest" or "best 7 of 9" rules don't change the count: list all 9 and split
  the group weight across all of them.
- A date given only as a class session or week number ("Session 12", "Week 7") is `null`.
- `year` is the year the term starts in. For dates in a term that crosses New Year, use
  the right year for each date.

## Golden set

| ID | Source | What it tests |
|---|---|---|
| `synthetic-cs101` | Synthetic | Grading table, group weights, dates without a year, TBA final, an ungraded item |
| `synthetic-stat200` | Synthetic | Weights and dates in prose, a multi-part project, an undated presentation |
| `synthetic-hist210` | Synthetic | Dates only in the weekly schedule, an ungraded milestone |
| `ocw-*` (5, AI-drafted labels) | MIT OpenCourseWare | Real syllabi in economics, statistics, theory and CS; weights in prose, drop-lowest rules, no calendar dates |
| `njit-*` (6, AI-drafted labels) | NJIT Digital Commons | Real CS and art syllabi; grading tables, midterm dates, drop-lowest rules, weights that don't add up |
| `cuny-*` (3, AI-drafted labels) | CUNY Academic Works | Real history, writing and education syllabi; due dates in weekly schedules, points instead of percentages, extra credit, two-column layouts |

Syllabus files without a `.json` label are skipped until they're labeled. Sources and
licenses are in [`golden/SOURCES.md`](golden/SOURCES.md).

The synthetic syllabi check that the harness works. Report metrics only from a set that
includes real syllabi.

## Results

Measured on 2026-11-02 over the 14 real syllabi (132 labeled items). Raw results are in
[`results/`](results/).

| Model | Prompt | Precision | Recall | F1 | Dates right on dated items |
|---|---|---|---|---|---|
| `groq/openai/gpt-oss-120b` | `3ebbb164` (original) | 70.0% | 95.5% | 80.8% | about 9 of 12 |
| `groq/openai/gpt-oss-20b` | `3ebbb164` (original) | 71.8% | 88.6% | 79.3% | about 8 of 12 |
| `groq/openai/gpt-oss-120b` | `d7f51cb0` (improved) | 92.1% | 87.9% | 89.9% | 8 of 12 |

The improved prompt spells out the labeling rules: leave out extra credit, convert points to
percent, and keep combined or uncounted items as one entry. Models then added far fewer wrong
items, at the cost of some recall.

Read these numbers with care:

- **The labels are AI drafts** that no person has checked yet (see above).
- **There's no held-out test set.** The prompt was improved by reading errors on these same
  syllabi, so its score is probably somewhat optimistic.
- **Runs vary.** The same prompt and model can score differently between runs even at
  temperature 0, so small differences don't mean much.
- **Only 12 items have real dates,** so date results are rough.
- **Some misses are naming, not errors,** such as "Participation" for "Classroom Attendance
  and Participation", or one entry for "Five to six assignments". These lower recall.
- **Not yet compared:** Gemini and the improved prompt on `gpt-oss-20b`, because free-tier daily
  limits ran out. To finish, run this on a new day:
  `syllabus-copilot eval --real-only --model gemini/gemini-3.8-flash --model groq/openai/gpt-oss-20b`
