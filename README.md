# Syllabus Copilot

An AI agent that reads your course syllabi, pulls out every deadline and exam, and builds a
personalized study schedule in your calendar.

Tools are exposed through [MCP](https://modelcontextprotocol.io), and the agent is
model-agnostic via [LiteLLM](https://github.com/BerriAI/litellm), so it runs on Gemini, Groq,
local Ollama models, and others. Every change is measured against a hand-labeled eval set.

> 🚧 Early development. See the roadmap below.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # add a free Gemini key from https://aistudio.google.com
docker compose up -d          # Postgres + pgvector (used from week 3)

syllabus-copilot extract path/to/syllabus.pdf --year 2026
```

Example output:

```json
{
  "course_code": "CMPT 276",
  "course_name": "Introduction to Software Engineering",
  "deliverables": [
    {"title": "Midterm Exam", "type": "exam", "due_date": "2026-10-21", "weight": 25.0}
  ]
}
```

## Project layout

```
src/syllabus_copilot/
  models.py    Pydantic schemas for syllabi and deliverables
  pdf.py       PDF → text
  llm.py       Provider-agnostic LLM calls (LiteLLM)
  extract.py   Syllabus text → structured deliverables
  cli.py       Command-line entry point
evals/         Hand-labeled golden dataset and eval harness
tests/         Unit tests (no network / API keys needed)
```

## Roadmap

- [x] **Week 1:** PDF → structured deadlines extraction
- [ ] **Week 2:** Eval harness + golden dataset, baseline scores across models
- [ ] **Week 3:** `course-docs` MCP server with RAG search (pgvector)
- [ ] **Week 4:** `calendar` MCP server (Google Calendar OAuth, free/busy)
- [ ] **Week 5:** Agent loop with human-in-the-loop confirmation
- [ ] **Week 6:** Next.js frontend
- [ ] **Week 7:** Deploy + CI-gated evals
- [ ] **Week 8:** Launch to real users
