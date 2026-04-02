# ADR-001: Handler Architecture - Pydantic Models and Fail-Fast SK-MCP Integration

**Date:** 2026-04-01
**Status:** Proposed
**Decision Area:** Core Architecture, Error Handling, Type Safety
**Affected Components:** `handlers.py`, `integrations.py`, `main.py`, `data_models.py`

---

## Context

The semantic-forge codebase had several critical blockers preventing the `build_dataset` pipeline from functioning:

1. **C1**: `generate_contrastive_pair` returned missing fields (`chosen_trajectory`, `rejected_trajectory`, `embedding_distance_chosen_rejected`)
2. **C2**: Fake hash-based embeddings produced invalid cosine distances (zero variance)
3. **C3**: Handlers returned dicts instead of Pydantic models, causing type mismatches in `build_dataset()`
4. **C4**: `SemanticKinematicsClient` had hardcoded endpoint, ignoring the `endpoint` parameter
5. **C5**: `main.py` accepted `--host`/`--port` arguments that were never used (stdio-only transport)

Additionally, the codebase lacked defensive testing for integration points (SK-MCP availability, LLM failures).

---

## Decision

### C1: Fail-Fast for SK-MCP Dependency

**Decision:** `generate_contrastive_pair` fails fast when SK-MCP is unavailable.

**Implementation:**
```python
class SemanticKinematicsRequiredError(Exception):
    """Raised when semantic-kinematics-mcp is required but unavailable."""
    pass

# In handle_generate_contrastive_pair:
sk_endpoint = get_semantic_kinematics_endpoint()
if not sk_endpoint:
    raise SemanticKinematicsRequiredError(
        "semantic-kinematics-mcp is required for generate_contrastive_pair but is not configured."
    )

# Verify connectivity
sk_client = SemanticKinematicsClient(sk_endpoint)
await sk_client.initialize()
await sk_client.model_status()  # Verify responsive
```

**Rationale:**
- SK-MCP is required for trajectory analysis and embedding distance
- Failing fast provides clear error messages rather than cryptic downstream failures
- Two-level check (config + connectivity) ensures the server is actually available
- Aligns with MCP-first design where containers are expected dependencies

---

### C2: Stub Local Embedding Fallback

**Decision:** Remove fake embeddings; defer local sentence-transformers fallback to GitHub issue #1.

**Implementation:**
```python
# Removed:
# hash_val = int(hashlib.md5(r.encode()).hexdigest(), 16) % 1000
# embedding = [hash_val / 1000.0] * 384  # Fake embedding - INVALID!

# Now: compute pairwise drift directly via SK-MCP
for i, text_a in enumerate(rephrasings):
    for j, text_b in enumerate(rephrasings):
        if j > i:
            drift_result = await sk_client.calculate_drift(text_a, text_b)
            drift = drift_result.get("drift", 0.0)
```

**Rationale:**
- Hash-based embeddings produced identical dimensions (zero variance)
- SK-MCP is the primary path and is running in production
- Local fallback is a nice-to-have enhancement for offline/CI scenarios
- Defer to issue #1 to keep scope manageable

---

### C3: Handlers Return Pydantic Models

**Decision:** All core handlers return Pydantic models; `register_handlers()` wraps with JSON serialization.

**Implementation:**
```python
# Handler signatures:
async def handle_permutate_phrasing(params) -> PermutatePhrasingResult
async def handle_generate_scenario(params) -> list[Scenario]
async def handle_generate_contrastive_pair(params) -> ContrastivePair
async def handle_build_dataset(params) -> BuildDatasetResult

# Serialization layer in register_handlers:
def wrap_model_handler(handler_func):
    async def wrapper(params):
        result = await handler_func(params)
        if isinstance(result, list):
            return handlers._make_result({"items": [r.model_dump() for r in result]})
        return handlers._make_result(result.model_dump())
    return wrapper

server.register_tool("generate_contrastive_pair")(
    wrap_model_handler(handlers.handle_generate_contrastive_pair)
)
```

**Rationale:**
- Full type safety from handler to `build_dataset()` pipeline
- Catches errors at compile time rather than runtime
- Handlers work with typed models internally; MCP boundary handles JSON
- Cleaner architecture than Option A (convert dicts at pipeline boundary)

---

### C4: Flexible Endpoint Parsing

**Decision:** `SemanticKinematicsClient` parses endpoint string to support multiple formats.

**Implementation:**
```python
def _parse_endpoint(self, endpoint: str) -> tuple[str, list[str], dict | None]:
    if endpoint.startswith("docker:"):
        docker_args = endpoint[7:].split(",")
        return "docker", ["run", "-i", "--rm"] + docker_args, None
    elif "," in endpoint:
        parts = endpoint.split(",")
        return parts[0], parts[1:], None
    else:
        return endpoint, [], None
```

**Supported formats:**
- Simple: `"semantic-kinematics-mcp"`
- Docker: `"docker:semantic-kinematics-mcp"`
- With args: `"command,arg1,arg2"`
- Full docker: `"docker:run,-i,--rm,network=host,image-name"`

**Rationale:**
- Supports both local binaries and Docker containers
- Matches MCP stdio transport design
- Backward compatible with simple command names

---

### C5: Stdio-Only Transport

**Decision:** Remove `--host`/`--port` arguments; document stdio-only usage.

**Implementation:**
```python
# Removed:
# parser.add_argument("--host", default="localhost", ...)
# parser.add_argument("--port", type=int, default=8080, ...)

# Updated:
async def run_server() -> None:  # No host/port parameters
    print("Starting semantic-forge MCP server (stdio transport)")
```

**Rationale:**
- Arguments were misleading (accepted but ignored)
- MCP clients (Docker containers) use stdio JSON-RPC
- No need for HTTP/SSE transport in current design
- Cleaner interface without unused parameters

---

## Consequences

### Positive

1. **Type Safety:** Full type checking from handler to dataset pipeline
2. **Clear Errors:** Fail-fast provides actionable error messages
3. **Test Coverage:** 52 tests (up from 28), including defensive integration tests
4. **Clean Interface:** Stdio-only matches MCP design philosophy
5. **Maintainable:** Pydantic models as single source of truth for output schemas

### Negative

1. **Breaking Change:** Handler return types changed (requires test updates)
2. **SK-MCP Dependency:** Pipeline requires SK-MCP to be running (no graceful degradation)
3. **Serialization Overhead:** `wrap_model_handler()` adds indirection layer

### Neutral

1. **Local Embeddings:** Deferred to issue #1 (enhancement, not blocker)

---

## Alternatives Considered

### Option A: Convert Dicts at Pipeline Boundary (Rejected)

```python
# In handle_build_dataset:
pair_model = ContrastivePair(**pair_data)  # Convert dict to model here
```

**Why rejected:** Type safety only at pipeline boundary, not at handler level. Errors caught later in development cycle.

### Option B: Keep Dicts, Add Runtime Validation (Rejected)

```python
# Validate dict keys at runtime
assert "chosen_trajectory" in result
```

**Why rejected:** No compile-time safety; validates too late in the process.

### Option C: Implement Local Embeddings Immediately (Deferred)

Add `sentence-transformers` fallback when SK-MCP unavailable.

**Why deferred:** SK-MCP is running in production; local fallback is nice-to-have for offline/CI scenarios.

### Option D: Add HTTP Transport (Rejected)

Support both stdio and HTTP/SSE transports.

**Why rejected:** MCP clients use stdio; adds unnecessary complexity. Can be added later if needed.

---

## Implementation Checklist

- [x] C1: Add `SemanticKinematicsRequiredError` and fail-fast checks
- [x] C2: Remove fake embeddings, compute pairwise drift via SK-MCP
- [x] C3: Refactor handlers to return Pydantic models
- [x] C4: Implement `_parse_endpoint()` in `SemanticKinematicsClient`
- [x] C5: Remove `--host`/`--port` arguments
- [x] Tests: Update existing tests, add defensive integration tests
- [x] Docs: Update ROADMAP.md with completion status

---

## References

- GitHub Issue #1: Implement local sentence-transformers fallback
- MCP Specification: https://modelcontextprotocol.io/
- Test results: 52 passing (up from 28)

---

## Future Considerations

1. **Local Embedding Fallback:** Implement when offline/CI support needed (issue #1)
2. **HTTP Transport:** Add if MCP clients require networked access
3. **Weighted CogSec Scoring:** Differentiate mechanism importance (M1 in ROADMAP)
4. **Parquet Export:** Add if downstream tools require it (H4 in ROADMAP)
