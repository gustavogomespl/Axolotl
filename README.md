# Axolotl

A multi-agent framework built with LangGraph and FastAPI. Each project has a planner agent that orchestrates worker agents, each with their own tools, skills, and MCP servers. Conversations are persisted by phone number (30 days) with Redis-backed session memory (3h window).

---

## Architecture

```
          React + shadcn/ui (:8080)
                  |
           nginx reverse proxy
                  |
           FastAPI Backend (:8000)
                  |
     +------------+------------+
     |            |            |
  Orchestrator  Tool Registry  RAG Engine
  (planner →    (API / MCP)   (ChromaDB)
   workers)          |
     |               |
     +-------+-------+-------+
             |       |       |
         PostgreSQL Redis  ChromaDB
         (data)    (sessions) (vectors)
```

**Backend** (`:8000`) -- FastAPI with LangGraph supervisor pattern, project-scoped REST API, SSE streaming, MCP server.

**Admin UI** (`:8080`) -- React + shadcn/ui + Tailwind CSS. Manages projects, agents, skills, tools, MCP servers, documents, chat, and conversation history.

**Infrastructure** -- PostgreSQL (pgvector) for data, Redis for LangGraph checkpointing (3h sessions), ChromaDB for vector storage.

---

## How It Works

1. **Projects** are the top-level container. Each project has a planner prompt and a default model.
2. **Agents** belong to a project. One agent is marked as the **planner** (`is_planner=true`), the rest are **workers**.
3. Each agent can be linked to **tools** (API or MCP), **skills** (RAG, prompt), and **MCP servers**.
4. When a message is sent to chat, the **orchestrator** loads the project's agents and their resources from the database, builds a LangGraph supervisor graph, and invokes it.
5. **Conversations** are persisted at the phone number level for 30 days. Active sessions are managed via Redis with a 3-hour window.

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

Services:
- Admin UI: `http://localhost:8080`
- Backend API: `http://localhost:8000`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`
- ChromaDB: `localhost:8001`

Verify:

```bash
curl http://localhost:8000/api/v1/health
```

### Local development

```bash
# Infrastructure
docker compose -f docker-compose.dev.yml up -d

# Backend (hot-reload)
cd backend && pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000

# Frontend (hot-reload)
cd admin_ui && npm install && npm run dev
```

---

## Project Structure

```
axolotl/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI app factory + lifespan
│   │   ├── config.py                   # Pydantic settings
│   │   ├── api/v1/
│   │   │   ├── projects.py             # Project CRUD
│   │   │   ├── agents.py               # Agent CRUD (with tool/skill/mcp linking)
│   │   │   ├── chat.py                 # Chat + conversations + streaming
│   │   │   ├── skills.py               # Skill CRUD + toggle
│   │   │   ├── tools.py                # Tool CRUD + test
│   │   │   ├── documents.py            # Document upload + ingestion
│   │   │   └── mcp_servers.py          # MCP server CRUD + refresh
│   │   ├── core/
│   │   │   ├── langgraph/
│   │   │   │   ├── graphs/
│   │   │   │   │   ├── simple_agent.py # Single ReAct agent (fallback)
│   │   │   │   │   └── supervisor.py   # Supervisor pattern (planner → workers)
│   │   │   │   └── tools/
│   │   │   │       ├── registry.py     # In-memory tool registry
│   │   │   │       ├── api_tool.py     # Dynamic HTTP tool builder
│   │   │   │       └── mcp_manager.py  # MCP client manager
│   │   │   ├── llm/provider.py         # Multi-provider LLM init
│   │   │   ├── vector_store/client.py  # ChromaDB collection manager
│   │   │   ├── redis.py                # Redis manager (LangGraph checkpointer)
│   │   │   └── mcp_server.py           # Axolotl as MCP server
│   │   ├── models/                     # SQLAlchemy ORM (project, agent, skill, tool, etc.)
│   │   └── services/
│   │       ├── orchestrator.py         # Planner/worker orchestration
│   │       ├── agent_resolver.py       # Resolves agent DB relations → LangChain tools
│   │       └── document_service.py     # Parse, chunk, embed, index pipeline
│   ├── tests/                          # Unit tests (232 tests, 90%+ coverage)
│   └── pyproject.toml
│
├── admin_ui/                           # React + Vite + shadcn/ui + Tailwind
│   ├── src/
│   │   ├── pages/                      # Projects, Agents, Skills, Tools, etc.
│   │   ├── api/                        # API client per resource
│   │   ├── components/                 # Layout + shared + shadcn/ui
│   │   └── router.tsx                  # Project-scoped routing
│   ├── nginx.conf                      # Production nginx (SPA + API proxy)
│   └── package.json
│
├── evals/                              # Evaluation suite (DeepEval, RAGAS)
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.admin                # Multi-stage: Node build → nginx serve
│   └── entrypoint.sh
├── docker-compose.yml                  # Full stack
├── docker-compose.dev.yml              # Infrastructure only
├── Makefile
└── .github/workflows/ci.yml           # Lint + tests (90% coverage gate)
```

---

## API Endpoints

All endpoints prefixed with `/api/v1`. Resources are scoped to projects.

### Projects

| Method | Path | Description |
|--------|------|-------------|
| POST | `/projects` | Create project |
| GET | `/projects` | List projects |
| GET | `/projects/{id}` | Get project |
| PUT | `/projects/{id}` | Update project |
| DELETE | `/projects/{id}` | Delete project |

### Agents

| Method | Path | Description |
|--------|------|-------------|
| POST | `/projects/{id}/agents` | Create agent (with tool_ids, skill_ids, mcp_server_ids) |
| GET | `/projects/{id}/agents` | List agents |
| GET | `/projects/{id}/agents/{aid}` | Get agent |
| PUT | `/projects/{id}/agents/{aid}` | Update agent |
| DELETE | `/projects/{id}/agents/{aid}` | Delete agent |

### Chat

| Method | Path | Description |
|--------|------|-------------|
| POST | `/projects/{id}/chat` | Send message (supports phone_number for persistence) |
| POST | `/projects/{id}/chat/stream` | SSE streaming |
| GET | `/projects/{id}/chat/conversations` | List conversations (filter by phone_number) |
| GET | `/projects/{id}/chat/conversations/{cid}/messages` | Get messages |

### Skills

| Method | Path | Description |
|--------|------|-------------|
| POST | `/projects/{id}/skills` | Create skill (rag, tool, prompt) |
| GET | `/projects/{id}/skills` | List skills |
| PUT | `/projects/{id}/skills/{sid}` | Update skill |
| DELETE | `/projects/{id}/skills/{sid}` | Delete skill |
| POST | `/projects/{id}/skills/{sid}/activate` | Toggle active |

### Tools

| Method | Path | Description |
|--------|------|-------------|
| POST | `/projects/{id}/tools` | Create API tool |
| GET | `/projects/{id}/tools` | List tools |
| DELETE | `/projects/{id}/tools/{tid}` | Delete tool |
| POST | `/projects/{id}/tools/{tid}/test` | Test tool |

### MCP Servers

| Method | Path | Description |
|--------|------|-------------|
| POST | `/projects/{id}/mcp-servers` | Add MCP server |
| GET | `/projects/{id}/mcp-servers` | List servers |
| DELETE | `/projects/{id}/mcp-servers/{mid}` | Remove server |
| POST | `/projects/{id}/mcp-servers/{mid}/refresh` | Reconnect + load tools |

### Documents

| Method | Path | Description |
|--------|------|-------------|
| POST | `/projects/{id}/documents` | Upload document (PDF, TXT, MD, DOCX) |
| GET | `/projects/{id}/documents` | List documents |
| DELETE | `/projects/{id}/documents/{did}` | Delete document |

---

## LLM Providers

Runtime provider switching via `provider:model_name` format:

```
openai:gpt-4.1-mini
openai:gpt-4.1
anthropic:claude-sonnet-4-6
ollama:llama3
```

Set the default in `.env` (`DEFAULT_MODEL`) or per-project/per-agent.

---

## MCP Integration

**As client** -- Connect to external MCP servers (HTTP or stdio). Tools are loaded into the registry and linked to agents.

**As server** -- Axolotl exposes its knowledge base at `/mcp` for MCP-compatible clients.

---

## Testing

```bash
make test         # Run unit tests
make test-cov     # Run with coverage (fails below 90%)
make lint         # Ruff + format check
make evals        # Evaluation pipeline (requires API keys)
```

CI enforces 90% code coverage on every push/PR.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | -- | OpenAI API key |
| `ANTHROPIC_API_KEY` | Yes* | -- | Anthropic API key |
| `DEFAULT_MODEL` | No | `openai:gpt-4.1-mini` | Default LLM |
| `DEFAULT_TEMPERATURE` | No | `0.0` | Default temperature |
| `DATABASE_URL` | No | auto (Docker) | PostgreSQL connection |
| `REDIS_URL` | No | auto (Docker) | Redis connection |
| `CHROMA_HOST` | No | auto (Docker) | ChromaDB host |
| `CHROMA_PORT` | No | auto (Docker) | ChromaDB port |
| `LANGSMITH_TRACING` | No | `false` | Enable LangSmith |
| `LANGSMITH_API_KEY` | No | -- | LangSmith key |

*At least one LLM provider key is required.

---

## License

MIT
