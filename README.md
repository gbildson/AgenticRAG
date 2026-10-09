# Agentic RAG

A small, configurable agentic retrieval framework. A bounded tool-calling agent gathers evidence from local documents and, when configured, a read-only SQLite database. The included help desk example is deliberately small: three Markdown guides, a four-row asset inventory, three tickets, and four evaluation questions.

The framework provides:

- `POST /api/chat` with `answer`, `evidence`, and `trace`
- a browser UI at `/` and a one-shot or interactive CLI
- recursive ingestion of Markdown, text, and PDF documents
- SQLite FTS5 search, plus optional cosine vector search when an embedding model is configured
- discovered SQLite schema with an enforced table/column allowlist
- a bounded tool loop requiring evidence before answering

## Quick start

Requirements: Python 3.11+, [`uv`](https://docs.astral.sh/uv/), and a chat-completions endpoint whose model supports tool calling. The adapter uses the OpenAI-compatible API shape. The default endpoint is a local server at `http://localhost:11434/v1`; set `LLM_BASE_URL` for a different compatible endpoint. No embedding model or external data service is required for the default full-text demo.

```bash
./setup.sh
cp .env.example .env
# Edit CHAT_MODEL in .env to name a tool-capable model available at LLM_BASE_URL.
./run.sh
```

Open `http://localhost:8000`, or ask from the terminal:

```bash
.venv/bin/python -m app.cli --show-evidence "Ticket 101 is about a broken laptop. Is a replacement automatic, and which laptops are available?"
```

Example output from a tool-capable model (wording and tool calls may vary):

```text
A replacement for a broken laptop is not automatic. Employees may request a replacement when a device cannot be repaired within two business days or when the service desk marks it as end of life. The technician first checks whether a repair is practical; for example, a cracked screen alone does not automatically qualify for replacement. The process involves opening a help desk ticket with the device asset ID and describing the problem. The service desk confirms eligibility and then assigns an available device from inventory. The old device must be returned before the replacement is closed out. Approved replacements are normally ready for pickup within three business days after an available device is assigned. If no suitable device is available, the service desk provides an updated estimate in the ticket (source: device-replacement.md, sections Eligibility, Process, Timing).

Currently available laptops are:

- CedarBook 14 (available, located in North office)
- CedarBook 16 (available, located in South office)

One laptop, Maple Pro 13, is assigned and not available (source: assets table).

--- Evidence ---

SQL: SELECT status FROM tickets WHERE id = 101
[
  {
    "status": "open"
  }
]

SQL: SELECT id, model, status, location FROM assets WHERE device_type = 'laptop'
[
  {
    "id": 1,
    "model": "CedarBook 14",
    "status": "available",
    "location": "North office"
  },
  {
    "id": 2,
    "model": "CedarBook 16",
    "status": "available",
    "location": "South office"
  },
  {
    "id": 3,
    "model": "Maple Pro 13",
    "status": "assigned",
    "location": "North office"
  }
]

[device-replacement.md, Timing]
Timing
Approved replacements are normally ready for pickup within three business days after an available device is assigned. If inventory has no suitable device, the service desk provides an updated estimate in the ticket.

[device-replacement.md, Process]
Process
Open a help desk ticket with the device asset ID and describe the problem. The service desk confirms eligibility, then assigns an available device from inventory. The old device must be returned before the replacement is closed out.

[device-replacement.md, Eligibility]
Eligibility
Employees may request a replacement when a device cannot be repaired within two business days, or when the service desk marks it as end of life. A cracked screen alone does not automatically qualify; the technician first checks whether a repair is practical.

Tools: query_database -> query_database -> search_documents
```

The first run builds a cached index under `data/rag_index/`. `./setup.sh` creates the tiny example SQLite database if it is missing. To rebuild the index, run `.venv/bin/python -m app.ingest --force`.

## Configure another corpus

Set `DOCUMENTS_DIR` to a directory containing `.md`, `.txt`, or `.pdf` files. Each file's relative path becomes its citation source; PDF pages and Markdown headings are retained as evidence locations. Set `SQL_DATABASE_PATH` to a SQLite database, or `none` for documents only. `INDEX_DIR` sets the index location.

`SQL_ALLOWED_SCHEMA` can restrict both the schema shown to the model and what SQLite permits it to read. For example:

```bash
export SQL_DATABASE_PATH=/path/to/support.sqlite
export SQL_ALLOWED_SCHEMA='{"tickets":["id","issue","status"],"assets":["id","model","status"]}'
```

If the allowlist is omitted, all ordinary and virtual tables and their visible columns are available. Unknown names fail during agent initialization. SQL queries remain read-only, limited to one `SELECT` or CTE, 100 rows, and two seconds. Treat the allowlist as a data-access control, not just a prompt hint.

`EMBEDDING_MODEL` is optional. When set, ingestion creates normalized vectors and retrieval fuses vector and FTS5 ranks. The configured endpoint must also implement the embeddings API. Changing documents or embedding models invalidates the cached index. `RETRIEVAL_TOP_K` defaults to 6; `MAX_AGENT_STEPS` defaults to 6. `PORT` defaults to 8000.

## API

```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which laptops are available?"}'
```

The response contains an `answer`, an `evidence` list of document passages or SQL rows, and a `trace` of tool calls. `GET /api/health` reports asset, index, and chat-model configuration status. Each request is independent; conversation history is not persisted.

## Evaluate and test

```bash
.venv/bin/python scripts/evaluate.py
.venv/bin/python -m pytest -q
```

The evaluator loads `.env` for local model runs, including `CHAT_MODEL`, `LLM_BASE_URL`, and `LLM_API_KEY`. Exported environment variables take precedence. Pass `--questions-file` for another JSON array or `--base-url` to evaluate a running API. The evaluator checks required tool routing and expected answer terms for `examples/helpdesk/questions.json`, and saves each question's tool trace in `eval_results.json` for troubleshooting. These are development checks, not a complete measure of answer quality. Tests use fake model responses and do not require model credentials.

## Layout

- `app/agent.py`: bounded tool loop and evidence trace
- `app/models.py`: chat and embedding protocols plus OpenAI-compatible adapter
- `app/ingest.py`: document extraction, chunking, and cached indexing
- `app/retrieval.py`: FTS5 and optional vector retrieval
- `app/sql_tool.py`: schema discovery and protected SQLite queries
- `app/api.py`, `app/cli.py`, `app/static/index.html`: interfaces
- `examples/helpdesk/`: documents, sample database, and evaluation questions
- `scripts/prepare_example.py`: deterministic example database builder

The help desk content and inventory are synthetic demonstration data. Replace them with your own corpus and database for a real deployment.

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).
