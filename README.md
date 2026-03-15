# Axolotl

A multi-agent framework built with LangGraph, FastAPI and NiceGUI. Supports dynamic agent orchestration (supervisor, swarm, single), agentic RAG with corrective retrieval, pluggable tools via MCP, and a web-based admin dashboard for managing the entire system without touching code.

---

## Architecture

```
                    NiceGUI Admin UI (:8080)
                           |
                    FastAPI Backend (:8000)
                           |
               +-----------+-----------+
               |           |           |
          LangGraph    Tool Registry   RAG Engine
          (agents)     (native/API/MCP) (ChromaDB)
               |           |           |
        +------+------+    |    +------+------+
        |      |      |    |    |             |
    Supervisor Swarm  Single|  Retrieve    Grade
     pattern  pattern agent |  + Rewrite   + Check
                            |
               +------------+------------+
               |            |            |
           PostgreSQL     Redis       ChromaDB
           (state)       (tasks)     (vectors)
```

**Backend** (`:8000`) -- FastAPI with LangGraph engine, REST API, SSE streaming, MCP server exposure.

**Admin UI** (`:8080`) -- NiceGUI dashboard for managing agents, skills, documents, tools, and chat.

**Infrastructure** -- PostgreSQL (pgvector) for persistence, Redis for task queue, ChromaDB for vector storage.

---

## Requirements

- Docker and Docker Compose
- At least one LLM API key (OpenAI or Anthropic)
- Python 3.11+ (only for local development without Docker)

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url> && cd axolotl
cp .env.example .env
```

Edit `.env` and set your API keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_MODEL=openai:gpt-4.1-mini
```

### 2. Start everything

```bash
docker compose up
```

That's it. This builds and starts all services (PostgreSQL, Redis, ChromaDB, backend, admin UI) and runs database migrations automatically.

Services:
- Backend API: `http://localhost:8000`
- Admin UI: `http://localhost:8080`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`
- ChromaDB: `localhost:8001`

Verify with:

```bash
curl http://localhost:8000/api/v1/health
```

You can also start individual services:

```bash
docker compose up postgres redis chromadb   # infrastructure only
docker compose up backend                   # backend + dependencies
docker compose up admin                     # admin UI + backend + dependencies
```

Stop everything with `docker compose down`.

### Local development (without Docker for backend)

If you prefer running the backend locally for hot-reload:

```bash
docker compose up -d postgres redis chromadb   # start infra
cd backend && pip install -e ".[dev]"           # install deps
alembic upgrade head                            # run migrations
uvicorn app.main:app --reload --port 8000       # start backend
```

---

## Project Structure

```
axolotl/
|-- backend/
|   |-- app/
|   |   |-- main.py                  # FastAPI application factory
|   |   |-- config.py                # Environment settings (Pydantic)
|   |   |-- api/v1/
|   |   |   |-- router.py            # Route aggregation
|   |   |   |-- health.py            # GET /health
|   |   |   |-- chat.py              # POST /chat, /chat/stream (SSE)
|   |   |   |-- agents.py            # Agent CRUD
|   |   |   |-- skills.py            # Skill CRUD
|   |   |   |-- documents.py         # Document upload and ingestion
|   |   |   +-- tools.py             # Tool and MCP server management
|   |   |-- core/
|   |   |   |-- langgraph/
|   |   |   |   |-- state.py         # AgentState schema
|   |   |   |   |-- factory.py       # GraphFactory (builds agents from config)
|   |   |   |   |-- graphs/
|   |   |   |   |   |-- simple_agent.py    # Single ReAct agent
|   |   |   |   |   |-- supervisor.py      # Supervisor + workers pattern
|   |   |   |   |   |-- swarm.py           # Decentralized swarm pattern
|   |   |   |   |   +-- rag_agent.py       # Corrective/Adaptive RAG
|   |   |   |   |-- tools/
|   |   |   |   |   |-- registry.py        # Tool registry
|   |   |   |   |   |-- api_tool.py        # Dynamic HTTP tool builder
|   |   |   |   |   +-- mcp_manager.py     # MCP client manager
|   |   |   |   +-- subgraphs/
|   |   |   |       +-- registry.py        # Reusable subgraph registry
|   |   |   |-- llm/
|   |   |   |   +-- provider.py      # Multi-provider LLM init
|   |   |   |-- vector_store/
|   |   |   |   +-- client.py        # ChromaDB collection manager
|   |   |   +-- mcp_server.py        # Axolotl as MCP server (FastMCP)
|   |   |-- models/                  # SQLAlchemy ORM models
|   |   +-- services/
|   |       +-- document_service.py  # Parse, chunk, embed, index pipeline
|   |-- tests/
|   |-- alembic/                     # Database migrations
|   +-- pyproject.toml
|
|-- admin_ui/
|   |-- main.py                      # NiceGUI entrypoint
|   +-- pages/
|       |-- layout.py                # Shared navigation
|       |-- chat_page.py             # Chat interface with streaming
|       |-- agents_page.py           # Agent management
|       |-- skills.py                # Skill management
|       |-- documents.py             # Document upload
|       |-- tools_page.py            # Tool and MCP management
|       +-- evals_page.py            # Evaluation dashboard
|
|-- evals/                           # Evaluation suite (DeepEval, RAGAS)
|   |-- test_agent_quality.py
|   |-- test_rag_quality.py
|   +-- datasets/                    # Test datasets (JSON)
|
|-- docker/
|   |-- Dockerfile.backend
|   |-- Dockerfile.admin
|   +-- entrypoint.sh          # Auto-runs migrations on startup
|
|-- docker-compose.yml               # Full production stack
|-- docker-compose.dev.yml           # Infrastructure only (dev)
|-- Makefile
|-- .env.example
+-- ruff.toml
```

---

## API Endpoints

All endpoints are prefixed with `/api/v1`.

### Chat

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send message, get response |
| POST | `/chat/stream` | Send message, get SSE stream |

### Agents

| Method | Path | Description |
|--------|------|-------------|
| POST | `/agents` | Create agent (single, supervisor, swarm) |
| GET | `/agents` | List all agents |
| GET | `/agents/{id}` | Get agent details |
| DELETE | `/agents/{id}` | Delete agent |

### Skills

| Method | Path | Description |
|--------|------|-------------|
| POST | `/skills` | Create skill |
| GET | `/skills` | List skills |
| GET | `/skills/{id}` | Get skill |
| PUT | `/skills/{id}` | Update skill |
| DELETE | `/skills/{id}` | Delete skill |
| POST | `/skills/{id}/activate` | Toggle active state |

### Documents

| Method | Path | Description |
|--------|------|-------------|
| POST | `/documents` | Upload and ingest document (PDF, TXT, MD, DOCX) |
| GET | `/documents` | List documents (optional `?collection=` filter) |
| GET | `/documents/{id}` | Get document details |
| DELETE | `/documents/{id}` | Delete document and its chunks |

### Tools

| Method | Path | Description |
|--------|------|-------------|
| POST | `/tools` | Register tool (native or API) |
| GET | `/tools` | List tools |
| GET | `/tools/{name}` | Get tool details |
| DELETE | `/tools/{name}` | Remove tool |
| POST | `/tools/{name}/test` | Test tool with sample input |
| POST | `/tools/mcp-servers` | Add MCP server |
| GET | `/tools/mcp-servers` | List MCP servers |
| DELETE | `/tools/mcp-servers/{name}` | Remove MCP server |
| POST | `/tools/mcp-servers/{name}/refresh` | Reconnect and reload tools |

---

## Agent Patterns

### Single Agent

A standalone ReAct agent with optional tools and a custom system prompt. Good for simple use cases.

### Supervisor

Hierarchical orchestration. A supervisor agent delegates tasks to specialized worker agents. Each worker has its own tools and system prompt. The supervisor decides which worker handles each request.

### Swarm

Decentralized peer-to-peer. Each agent can hand off to any other agent in the group. No central coordinator. Agents decide amongst themselves who should handle the current task.

---

## RAG Pipeline

The RAG engine implements corrective and adaptive retrieval:

1. **Classify** -- Determine if the query needs retrieval, web search, or a direct answer.
2. **Retrieve** -- Search relevant ChromaDB collections.
3. **Grade** -- Evaluate document relevance (LLM-as-judge).
4. **Decide** -- If documents are relevant, generate. Otherwise, rewrite query and re-retrieve.
5. **Generate** -- Produce answer grounded in retrieved context.
6. **Hallucination check** -- Validate the response against the source documents.

Documents are ingested through the `/documents` endpoint. Supported formats: PDF, DOCX, TXT, Markdown. Files are parsed, split into chunks, embedded, and indexed in ChromaDB.

---

## MCP Integration

Axolotl works with MCP in two directions:

**As client** -- Connect to external MCP servers (Streamable HTTP or stdio transport). Tools from those servers are loaded into the tool registry and become available to agents.

**As server** -- Axolotl exposes its own knowledge base and capabilities as an MCP server at `/mcp`, allowing other MCP-compatible clients to query it.

---

## LLM Providers

The framework supports runtime provider switching via `init_chat_model()`. Set the model using the `provider:model_name` format:

```
openai:gpt-4.1-mini
anthropic:claude-sonnet-4-6
ollama:llama3
```

Change the default model in `.env` (`DEFAULT_MODEL`) or per-agent when creating agents through the API or admin UI.

---

## Docker Compose Commands

```
docker compose up                              # Start everything (builds if needed)
docker compose up -d                           # Start in background
docker compose up backend                      # Start backend + infra
docker compose up postgres redis chromadb      # Start infra only
docker compose down                            # Stop all containers
docker compose logs -f                         # Follow logs
docker compose build                           # Rebuild images
```

### Makefile (shortcuts)

```
make lint         Run ruff and mypy
make test         Run unit tests
make evals        Run evaluation pipeline
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | -- | OpenAI API key |
| `ANTHROPIC_API_KEY` | Yes* | -- | Anthropic API key |
| `DEFAULT_MODEL` | No | `openai:gpt-4.1-mini` | Default LLM model |
| `DEFAULT_TEMPERATURE` | No | `0.0` | Default temperature |
| `DATABASE_URL` | No | auto-set by Docker | PostgreSQL connection string |
| `REDIS_URL` | No | auto-set by Docker | Redis connection string |
| `CHROMA_HOST` | No | auto-set by Docker | ChromaDB host |
| `CHROMA_PORT` | No | auto-set by Docker | ChromaDB port |
| `LANGSMITH_TRACING` | No | `false` | Enable LangSmith tracing |
| `LANGSMITH_API_KEY` | No | -- | LangSmith API key |
| `APP_ENV` | No | `development` | Environment (development/production) |

*At least one LLM provider key is required.

---

## Running Tests

```bash
make test          # unit tests
make evals         # evaluation pipeline (requires API keys)
```

---

## License

MIT
