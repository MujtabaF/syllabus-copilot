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
pip install -e ".[dev,embeddings]"
cp .env.example .env          # add a free Gemini key from https://aistudio.google.com
docker compose up -d          # Postgres + pgvector, on port 5433

syllabus-copilot extract path/to/syllabus.pdf --year 2026
syllabus-copilot eval --model gemini/gemini-3.8-flash --model groq/openai/gpt-oss-120b
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

## MCP servers

Two servers expose the project's tools to any MCP client, such as Claude Desktop or Cursor.

| Server | Command | Tools |
|---|---|---|
| Course docs | `course-docs-mcp` | `docs_ingest_syllabus`, `docs_search`, `docs_list_courses` |
| Calendar | `calendar-mcp` | `calendar_get_busy`, `calendar_find_free_slots`, `calendar_list_study_blocks`, `calendar_create_study_block`, `calendar_move_study_block` |

Example client config (use the full path to your virtual environment):

```json
{
  "mcpServers": {
    "course-docs": {
      "command": "/path/to/syllabus-copilot/.venv/bin/course-docs-mcp",
      "env": {
        "COURSE_DOCS_DIR": "/path/to/your/syllabi",
        "DATABASE_URL": "postgresql://copilot:copilot@localhost:5433/copilot"
      }
    },
    "calendar": {
      "command": "/path/to/syllabus-copilot/.venv/bin/calendar-mcp",
      "env": { "GOOGLE_SECRETS_DIR": "/path/to/syllabus-copilot/.secrets" }
    }
  }
}
```

**Course docs.** Syllabi are split into chunks and embedded on your machine, so their text
isn't sent anywhere. The first run downloads the embedding model (about 90 MB). The server
only reads `.pdf` and `.txt` files inside `COURSE_DOCS_DIR`.

**Calendar setup.**

1. In the [Google Cloud console](https://console.cloud.google.com/), create a project and
   enable the Google Calendar API.
2. Configure the OAuth consent screen and add your own Google account as a test user.
3. Create an OAuth client ID of type **Desktop app** and save its JSON file as
   `.secrets/google_client.json`. `.secrets/` is ignored by git.
4. Run `syllabus-copilot calendar-login` and approve access in your browser.

**Calendar safety.**

- The server sees only *when* you're busy, not what your events are.
- It can only list or move study blocks it created, which carry a private marker.
- Creating or moving a block returns a preview, including any conflicts, unless the call
  passes `confirm=true`. A client should show that preview to you before confirming.

## Project layout

```
src/syllabus_copilot/
  models.py    Pydantic schemas for syllabi and deliverables
  pdf.py       PDF → text
  llm.py       Provider-agnostic LLM calls (LiteLLM)
  extract.py   Syllabus text → structured deliverables
  evals.py     Scores extraction against the golden dataset
  docs.py      Syllabus chunks, local embeddings, and vector search
  gcal.py      Google sign-in, free/busy lookup, and study blocks
  servers/     MCP servers: course_docs.py and google_calendar.py
  cli.py       Command-line entry point
evals/         Hand-labeled golden dataset and results (see evals/README.md)
tests/         Unit tests (no network / API keys needed)
```

## Roadmap

- [x] **Week 1:** PDF → structured deadlines extraction
- [x] **Week 2:** Eval harness + golden dataset, baseline scores across models
- [x] **Week 3:** `course-docs` MCP server with RAG search (pgvector)
- [x] **Week 4:** `calendar` MCP server (Google Calendar OAuth, free/busy). Tested against
  Google's client library with simulated responses, not yet against a live account.
- [ ] **Week 5:** Agent loop with human-in-the-loop confirmation
- [ ] **Week 6:** Next.js frontend
- [ ] **Week 7:** Deploy + CI-gated evals
- [ ] **Week 8:** Launch to real users
