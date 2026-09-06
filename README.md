# poc-agentic-lims

Agentic SOP and AI assistant for a Laboratory Information Management System (LIMS) — a proof of concept.

## What this is

Lab staff need fast, trustworthy answers to questions like *"what's the storage temperature for SOP-02?"* or *"what's the pH range for injectables?"* without digging through PDF manuals, and they need to look up sample data (batch, type, status) by sample ID. This project is a chat API that answers both kinds of questions:

- **SOP / regulatory questions** are answered via retrieval-augmented generation (RAG) over the lab's own SOP and equipment-manual documents — the model never answers from its own memory, only from retrieved document chunks.
- **Sample lookups** are answered via a direct tool call against sample data.
- A guardrail layer flags and suppresses answers that aren't actually grounded in what the tools returned, since a wrong SOP limit in a regulated lab is a real-world risk, not just an annoying bug.

It's built as a POC: uses Groq API for LLM inference and Ollama locally for embeddings, but structured so the LLM provider, vector store, and retrieval tuning are all swappable via configuration rather than code changes.

## Use Cases

### Real-World Scenarios

**1. Lab Technician During Sample Processing**
- *Scenario:* Mid-procedure, needs to verify pH limits for injectable solution (SOP-02)
- *Query:* "What to do if pH is too low in SOP-02?"
- *Traditional approach:* Stop work, open PDF manual, search through 20+ pages, find section
- *With this app:* Type question, get instant answer with exact page reference, continue work
- *Value:* Reduces 5-10 minute manual lookup to <10 seconds, maintains sterile field/workflow continuity

**2. Quality Control Auditor**
- *Scenario:* Reviewing batch compliance, needs to cross-reference multiple SOPs
- *Query:* "What PPE is required for handling hazardous chemicals?"
- *Traditional approach:* Search through multiple PDF files across different directories
- *With this app:* Single query retrieves relevant sections from all SOPs and equipment manuals
- *Value:* Faster audits, reduced risk of missing critical safety requirements

**3. New Employee Onboarding**
- *Scenario:* Training on equipment calibration procedures
- *Query:* "How do I calibrate the spectrophotometer?"
- *Traditional approach:* Read entire equipment manual, ask supervisor for clarification
- *With this app:* Get step-by-step instructions with context from both SOP and equipment manual
- *Value:* Accelerates training, reduces supervisor interruptions, provides consistent answers

**4. Incident Response**
- *Scenario:* Equipment malfunction during production run
- *Query:* "HFA filter clogged after F2 cleaning procedure"
- *Traditional approach:* Emergency stop, call supervisor, wait for manual lookup
- *With this app:* Immediate troubleshooting guidance from maintenance manuals
- *Value:* Reduces downtime, enables faster problem resolution

**5. Regulatory Compliance Queries**
- *Scenario:* Inspector asks about storage temperature validation
- *Query:* "What are the temperature requirements for SOP-02?"
- *Traditional approach:* Search through validation documents, risk providing incorrect information
- *With this app:* Verified answer with exact source citation, hallucination guardrails prevent speculation
- *Value:* Confident compliance responses, audit-trail ready citations

### Why This Matters

**Laboratory Context:**
- **Time-sensitive operations:** Delays during procedures can compromise sample integrity
- **Regulatory environment:** Wrong information can lead to batch rejection, failed audits, or safety incidents
- **Knowledge accessibility:** 50+ SOPs and equipment manuals across multiple systems
- **Staff turnover:** New employees need quick, reliable access to institutional knowledge
- **24/7 operations:** Instant access to SOPs needed during night shifts when supervisors unavailable

**Business Impact:**
- **Efficiency:** Reduce lookup time from minutes to seconds
- **Quality:** Consistent, verified answers reduce human error
- **Compliance:** Audit-ready citations and hallucination guardrails
- **Training:** Accelerate onboarding, reduce supervisor burden
- **Cost savings:** Minimize production delays, prevent compliance failures

## Architecture

```
Client
  │  POST /api/v1/chat/completions  {message, thread_id}
  ▼
FastAPI route (api/routes/chat.py)
  ▼
ValidationMiddleware.validated_invoke   (middleware/validation.py)
  │  wraps the graph call, checks the final answer against tool output afterward
  ▼
LangGraph agent  (agent/graph.py)
  │
  │   ┌─────────────┐   tool_calls?   ┌───────────┐
  └──▶│  agent node │ ───────────────▶│ tools node│
      │ (call_model)│◀─────────────── │ (ToolNode)│
      └─────────────┘   tool result   └───────────┘
        │                                  │
        │ trims history,                   ├─ get_sop_rules  → EnsembleRetriever (BM25 + Chroma)
        │ calls LLM                        └─ get_sample_data → in-memory sample lookup
        ▼
   ChatGroq / ChatAnthropic (swappable via LLM_PROVIDER)
```

Conversation state (`MessagesState`) is persisted per `thread_id` in SQLite (`AsyncSqliteSaver`) so a thread survives across requests and process restarts. Every LLM call first passes through a trimmer that bounds the prompt to a fixed token budget and prunes the same messages out of the persisted checkpoint, so state doesn't grow unbounded across a long-running conversation.

## What it uses, and why

| Piece | Why |
|---|---|
| **FastAPI** | Async HTTP layer with auto-generated OpenAPI docs (`/docs`) for a small POC API. |
| **LangGraph** (`StateGraph` + `MessagesState`) | Models the agent as an explicit graph (agent ⇄ tools loop) instead of a hand-rolled `while` loop, and gets checkpointing, thread-scoped state, and conditional tool-routing for free. |
| **AsyncSqliteSaver** | Durable, per-`thread_id` conversation memory — a user's chat history survives a server restart without needing a separate database service. |
| **ChatGroq** (default) / **ChatAnthropic** (optional) | Groq provides fast, cost-effective inference via API. A small provider layer (`llm/`) makes swapping to Claude or other providers a config change, not a code change. |
| **Chroma + OllamaEmbeddings** (`nomic-embed-text`) | A file-backed vector store with Ollama for local embeddings — no external vector DB or embedding service needed. |
| **BM25Retriever + EnsembleRetriever** (hybrid retrieval) | Vector search alone misses exact keyword/ID matches (e.g. a user typing "SOP-02"); BM25 alone misses paraphrased questions. Combining both with weighted RRF (`EnsembleRetriever`) covers both cases. |
| **RecursiveCharacterTextSplitter** | Splits SOP/equipment-manual PDFs and text files into small (~500 char) overlapping chunks so retrieval returns focused, relevant passages instead of whole documents. |
| **`smart_trim_message_history`** (custom, on top of LangChain's `trim_messages`) | Keeps the prompt sent to the LLM bounded to a fixed token budget regardless of how long the conversation gets. |
| **`ResponseValidator` + `ValidationMiddleware`** | A cheap, deterministic hallucination guardrail: flags speculative language ("probably", "I think") and numbers in the answer that don't appear in the tool output it was supposed to be grounded in, and replaces unverified answers with a safe refusal instead of surfacing a possibly-wrong number. |
| **Pydantic Settings** | Typed, `.env`-driven configuration for every tunable (chunk size, retriever `k`/score threshold, model choice, context window, etc.) in one place. |
| **LangSmith** | Tracing/observability for agent runs, and a channel for recording hallucination-check feedback (`hallucination_check`) against each traced run. |
| **pytest / ruff / mypy** | Test, lint, and type-check tooling used in CI-style `make` targets. |
| **Docker Compose** | Bundles the API for quick local deployment. |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
make install    # uv sync --group groq — creates .venv/ and installs Groq provider deps
make ingest     # build the Chroma index from data/sops/ and data/equipment_manuals/
make dev        # run the API on :8000
```

## Docker Compose

Run the full stack with Docker Compose:

```bash
make compose-build  # Build images
make compose-up     # Start all services (API + Ollama for embeddings)
make compose-logs   # Follow logs

# Pull required model on startup (if not already pulled)
docker compose exec ollama ollama pull nomic-embed-text
```

Access the API at http://localhost:8000/docs

## Configuration

See `.env` for configurable settings (backed by `src/lims_agent/core/config.py`). Key settings:

- `DATABASE_URL`: SQLite database path for the conversation checkpointer (default: `sqlite:///./data/lims_agent.db`)
- `GROQ_API_KEY`: Required for Groq API access
- `GROQ_MODEL`: Groq model ID (default: `openai/gpt-oss-120b`)
- `OLLAMA_BASE_URL`: URL of the Ollama server for embeddings (default `http://localhost:11434`; Docker Compose overrides this to `http://ollama:11434` so the API container can reach the `ollama` service)
- `SOPS_SOURCE_DIRS`: directories scanned for SOP/manual source documents (default `["./data/sops", "./data/equipment_manuals"]`)
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: text-splitter settings for ingestion (default `500` / `50`)
- `RETRIEVER_K`: chunks retrieved per sub-retriever (BM25 and vector each contribute up to this many)
- `RETRIEVER_SCORE_THRESHOLD`: minimum similarity score for the vector retriever to surface a chunk (default `0.5`)
- `LANGSMITH_API_KEY`: required for tracing
- `CHROMA_PERSIST_DIR`: vector store location

### Query Rewriting & Re-ranking

**Query Rewriting** expands user queries into semantic variants before retrieval to improve recall:

- `QUERY_REWRITING_ENABLED`: Enable/disable query rewriting (default: `true`)
- `QUERY_REWRITING_STRATEGY`: Strategy to use:
  - `multi_query`: LLM generates 2-3 semantic paraphrases (recommended)
  - `rule_based`: Pattern matching for SOP IDs + parameter synonyms (faster, deterministic)
- `QUERY_REWRITING_NUM_VARIANTS`: Number of query variants to generate (default: `3`)
- `QUERY_REWRITING_TEMPERATURE`: LLM temperature for query generation (default: `0.3`)

**Re-ranking** scores retrieved documents by relevance and reorders results:

- `RERANKING_ENABLED`: Enable/disable re-ranking (default: `true`)
- `RERANKER_MODEL`: Model name for cross-encoder strategy (default: `cross-encoder/ms-marco-MiniLM-L-12-v2`)
- `RERANKER_TOP_K`: Number of documents to return after re-ranking (default: `5`)

**Performance Notes:**
- Query rewriting adds ~200-500ms per query (multiple retrieval calls)
- Cross-encoder re-ranking adds ~50-100ms for 10 documents
- Combined latency: ~300-600ms (acceptable for complex queries)
- To optimize: reduce `QUERY_REWRITING_NUM_VARIANTS` to `2` or disable for simple queries

**Trade-offs:**
- `multi_query` + `cross_encoder`: Best quality, moderate latency
- `rule_based` + `cross_encoder`: Good quality, lower latency  
- Multi-query only: Better recall, no re-ranking cost
- Re-ranking only: Better precision, no query expansion cost

### Switching LLM Providers

The application uses the **Strategy pattern** for LLM provider abstraction.

**Production (Groq - Fast & Free, default):**
```bash
# Configure provider
export GROQ_API_KEY=gsk_...
export GROQ_MODEL=openai/gpt-oss-120b  # or any accessible Groq model

make dev
```

**Production (Anthropic Claude):**
```bash
# Install Anthropic dependencies
uv sync --group anthropic

# Configure provider
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

make dev
```

The provider is selected at runtime based on configuration—no code changes needed.

### LLM Provider Comparison

| Provider | Speed | Cost | Free Tier | Best For |
|----------|-------|------|-----------|----------|
| **Groq** | **Very Fast** | Free | 14,400 req/day | Production POC, demos (default) |
| **Anthropic** | Fast | ~$0.003/query | None | High quality needs |

**Recommendation:** **Groq** is the default and recommended provider—it's fast, cost-effective, and has a generous free tier.

## API Endpoints

- `POST /api/v1/chat/completions` - Chat with the agent
- `GET /health` - Health check

## Development

```bash
make test       # Run pytest
make lint       # Check code style
make typecheck  # Run mypy
```
