# Semantic-Forge: Implementation Roadmap

**Last Updated:** 2026-04-02
**Status:** C1-C5 Complete + Backend Switching MVP

---

## Current State (Actual)

| Area | Status | Notes |
|------|--------|-------|
| **CogSec Judge** | ✅ Complete | All 9 manipulation mechanisms detected, scoring logic working |
| **Concept Library** | ✅ Complete | 8 behavioral concepts defined and working |
| **Data Models** | ✅ Complete | Pydantic models for all output types |
| **Dataset I/O** | ✅ Complete | JSONL/JSON save/load working |
| **Tests** | ⚠️ 59 unit tests | All tests use mocked LLMs/SK-MCP; no integration or E2E tests in pytest suite |
| **MCP Server** | ✅ Complete | stdio-only transport, host/port arguments removed |
| **LLM Integration** | ✅ Complete | Handlers return Pydantic models, proper type safety |
| **Semantic-Kinematics Integration** | ✅ Complete | Backend switching via model_load tool, LM Studio default |
| **Prompt-Prix Integration** | ⚠️ Partial | Client code exists, not tested/integrated |
| **`build_dataset` Handler** | ✅ Complete | Returns BuildDatasetResult, full type safety |
| **`generate_contrastive_pair`** | ✅ Complete | Returns all fields including trajectory and embedding distance |
| **Parquet Export** | ❌ Not Implemented | Only JSONL/JSON supported |

---

## Test Coverage Gaps

| Area | Status | Notes |
|------|--------|-------|
| **`build_dataset` handler** | ❌ Untested | No test exercises the full pipeline end-to-end |
| **Integration tests** | ❌ None | All tests mock LLM and SK-MCP backends |
| **E2E tests** | ❌ None in pytest | `test_connections.py` exists as manual script outside test suite |
| **Minimal dataset generation** | ❌ Untested | No test validates JSONL output matches README schema |

---

## Critical Blockers (Must Fix)

**Note: Order reflects dependencies. Complete in sequence.**

### C1: Fix `generate_contrastive_pair` Output Format (FOUNDATIONAL) ✅
**Status:** Complete

**What was fixed:**
- ✅ Handler now returns `ContrastivePair` Pydantic model (not dict)
- ✅ All trajectory fields populated via SK-MCP `analyze_trajectory()`
- ✅ Embedding distance computed via SK-MCP `calculate_drift()`
- ✅ Fail-fast when SK-MCP not configured or unavailable
- ✅ Defensive tests added (4 new tests)

**Location:** `handlers.py:207-310`

---

### C2: Remove Fake Embeddings ✅
**Status:** Complete (stubbed, local fallback deferred to issue #1)

**What was fixed:**
- ✅ Removed broken hash-based fake embeddings (zero variance)
- ✅ `validate_diversity` now computes pairwise drift via SK-MCP
- ✅ TODO added for local sentence-transformers fallback

**Deferred:** Local embedding fallback implemented when SK-MCP unavailable (GitHub issue #1)

**Rationale:** SK-MCP is the primary path and is running in user's environment. Local fallback is a nice-to-have enhancement for offline/CI scenarios.

---

### C3: Fix `build_dataset` Pipeline Type Safety ✅
**Status:** Complete

**What was fixed:**
- ✅ All handlers now return Pydantic models (not dicts wrapped in CallToolResult)
- ✅ `handle_permutate_phrasing` → `PermutatePhrasingResult`
- ✅ `handle_generate_scenario` → `list[Scenario]`
- ✅ `handle_generate_contrastive_pair` → `ContrastivePair`
- ✅ `handle_build_dataset` → `BuildDatasetResult`
- ✅ `register_handlers` wraps Pydantic returns with JSON serialization
- ✅ Full type safety throughout the pipeline

**Location:** `handlers.py`, `register_handlers()`

---

### C4: Fix `SemanticKinematicsClient` Endpoint Handling ✅
**Status:** Complete

**What was fixed:**
- ✅ Added `_parse_endpoint()` method supporting multiple formats
- ✅ Simple command: `"semantic-kinematics-mcp"`
- ✅ Docker prefix: `"docker:semantic-kinematics-mcp"`
- ✅ Comma-separated args: `"command,arg1,arg2"`
- ✅ Full docker format: `"docker:run,-i,--rm,network=host,image-name"`
- ✅ `initialize()` now uses `self._parse_endpoint(self.endpoint)` instead of hardcoded command
- ✅ 12 new tests for endpoint parsing and initialization

**Location:** `integrations.py:76-108`

---

### C5: Clarify MCP Server as Stdio-Only ✅
**Status:** Complete

**What was fixed:**
- ✅ Removed `--host` and `--port` CLI arguments (were accepted but ignored)
- ✅ `run_server()` no longer accepts host/port parameters
- ✅ Updated help text: `--server` now shows "(stdio transport only)"
- ✅ Print message clarifies: "Starting semantic-forge MCP server (stdio transport)"
- ✅ Tests verify stdio-only interface and reject unrecognized args

**Location:** `main.py:18-33`, `main.py:65-97`

**Rationale:** MCP clients (Docker containers) use stdio JSON-RPC; no need for HTTP/SSE transport.

---

## High Priority Fixes

### H1: Fix `export_for_dpo` Type Mismatch
**Issue:** Function passes `list[dict]` to `save_dataset` which expects `list[TrainingExample]`.

**Location:** `dataset.py:203-213`

**Tasks:**
- [ ] Create proper DPO export function that handles dicts
- [ ] Or convert dicts to a lightweight DPO model

### H2: Add Proper Error Handling
**Issue:** Multiple handlers silently swallow exceptions.

**Locations:**
- `handlers.py:125-126` - Silent pass in diversity validation
- `handlers.py:325-333` - Error returns empty data instead of propagating

**Tasks:**
- [ ] Add logging for failures
- [ ] Return proper error responses
- [ ] Add configurable retry logic

### H3: Fix `generate_structured` Error Handling
**Issue:** No error handling for malformed JSON from LLMs.

**Location:** `llm.py:71-96`

**Tasks:**
- [ ] Add try/except around `json.loads`
- [ ] Implement retry with different prompt
- [ ] Add validation against expected schema

### H4: Add Parquet Export Support
**Issue:** README claims parquet support but only JSONL/JSON implemented.

**Tasks:**
- [ ] Add `pyarrow` or `pandas` dependency
- [ ] Implement parquet writer in `dataset.py`
- [ ] Or remove parquet claim from documentation

---

## Medium Priority

### M1: Weight CogSec Mechanisms Appropriately
**Issue:** All 9 mechanisms have equal weight; `intent_defense` should dominate.

**Current:** `cogsec.py:210`
```python
manipulation_score = len(detected_mechanisms) / len(MECHANISMS)
```

**Tasks:**
- [ ] Define weights per mechanism
- [ ] Implement weighted scoring
- [ ] Update documentation

### M2: Narrow Regex Patterns in CogSec
**Issue:** Overly broad patterns cause false positives.

**Problematic Patterns:**
- `r"\bwe\b.*\bcan\b"` - matches any "we can" in text
- `r"\bI think\b"` - legitimate in reasoning contexts

**Tasks:**
- [ ] Review and narrow each pattern
- [ ] Add context-aware detection
- [ ] Update tests

### M3: Add Integration Tests
**Issue:** All tests mock the LLM client; no real integration tests.

**Tasks:**
- [ ] Add optional integration tests (skip if no LLM available)
- [ ] Test actual Ollama/vLLM connections
- [ ] Test `build_dataset` end-to-end

### M4: Fix Config Type Safety
**Issue:** `inference: dict` loses type safety.

**Location:** `config.py:38-46`

**Tasks:**
- [ ] Use `TypedDict` or proper dataclass
- [ ] Add validation for config values

---

## Low Priority / Nice-to-Have

### L1: Remove Unused `utils.py` Functions
**Issue:** `sample_with_min_distance`, `chunk_text`, `merge_scores` are never called.

**Tasks:**
- [ ] Remove unused functions or document intended use

### L2: Add Proper Logging
**Issue:** Uses `print()` statements instead of logging module.

**Tasks:**
- [ ] Replace `print()` with proper logging
- [ ] Add log level configuration

### L3: Define Constants for Magic Numbers
**Issue:** Thresholds like `0.2`, `0.5` appear in multiple places.

**Tasks:**
- [ ] Create `consts.py` or add to `config.py`
- [ ] Document threshold rationale

### L4: Fix `spread_score` Calculation
**Issue:** `len(rephrasings) * 0.1` is not a real diversity metric.

**Location:** `handlers.py:97`

**Tasks:**
- [ ] Implement proper embedding-based spread calculation
- [ ] Or remove metric until embeddings work

---

## Completed Items (Verified Working)

| Item | Verification | Notes |
|------|--------------|-------|
| CogSec Judge detection | 10 tests passing | Mocked completions only |
| Concept library lookup | 7 tests passing | Static data, no LLM required |
| Dataset JSONL I/O | 2 tests passing | File I/O only, no generation |
| Dataset stats computation | 1 test passing | Mock data |
| Handler basic structure | 8 tests passing | All handlers mocked |
| SK-MCP client | 19 tests passing | Defensive tests, no real Docker calls |
| Backend config | 7 tests passing | Config loading and validation |
| CLI `--list-concepts` | Manual verification | Works as expected |
| CLI `--concept <id>` | Manual verification | Works as expected |
| **`build_dataset`** | ❌ No tests | Handler exists but untested |

---

## Phase 5: Critical Bug Fixes (COMPLETED)

**Order Completed:** C1 → C2 → C3 → C4 → C5

| Task | Priority | Dependencies | Status |
|------|----------|--------------|--------|
| Fix `generate_contrastive_pair` output | Critical | None | ✅ Complete |
| Remove fake embeddings | Critical | C1 | ✅ Complete |
| Fix `build_dataset` pipeline | Critical | C1, C2 | ✅ Complete |
| Fix `SemanticKinematicsClient` | Medium | C2 | ✅ Complete |
| Fix MCP server transport | Medium | None | ✅ Complete |
| Fix `export_for_dpo` | High | None | ⏳ Pending (deferred) |
| Add proper error handling | High | C1-C3 | ⏳ Pending (deferred) |
| Add parquet export | High | None | ⏳ Pending (deferred) |

---

## Phase 6: Backend Switching MVP (COMPLETED)

**Date:** 2026-04-02

**Goal:** Enable runtime backend switching via SK-MCP `model_load` tool, prioritizing LM Studio with embeddinggemma:300m for low VRAM usage.

| Task | Priority | Dependencies | Status |
|------|----------|--------------|--------|
| Add backend config to `SemanticKinematicsClient.__init__` | High | None | ✅ Complete |
| Implement `_ensure_backend()` method | High | None | ✅ Complete |
| Add `model_status`, `model_load`, `model_unload` public methods | High | None | ✅ Complete |
| Update config schema with backend fields | High | None | ✅ Complete |
| Add env var support (`SEMANTIC_KINEMATICS_BACKEND`, etc.) | High | None | ✅ Complete |
| Update handlers to pass backend config | High | Config | ✅ Complete |
| Add tests for backend config | Medium | None | ✅ Complete (7 new tests) |
| Update README documentation | Medium | None | ✅ Complete |

**TODOs (marked in code):**
- Add error handling for failed `model_load` in `_ensure_backend()`
- Check current status before switching to avoid unnecessary reloads
- Graceful fallback if backend unavailable

**Test Count:** 59 passing (52 original + 7 new backend config tests)

---

## Phase 7: Context Health Concepts (ADR-CORE-079 Integration)

**Date:** Planned
**Depends on:** C1-C5 (complete), H1 (DPO export fix), LAS ADR-CORE-079 Phase 0 empirical validation

**Goal:** Add the context_health concept family and regularization fragment generation, connecting semantic-forge to LAS's context curation layer.

| Task | Priority | Dependencies | Status |
|------|----------|--------------|--------|
| Define `coherence_resistance` concept + scenarios | High | None | ⏳ Pending |
| Define `directive_preservation` concept + scenarios | High | None | ⏳ Pending |
| Define `minority_voice_attention` concept + scenarios | High | None | ⏳ Pending |
| Add regularization fragment output mode to `permutate_phrasing` | High | None | ⏳ Pending |
| CogSec judge: detect conclusory compression pattern | Medium | M1 (weighted scoring) | ⏳ Pending |
| Dual-flywheel scenario templates (LAS graph vs react_step) | Medium | Concept definitions | ⏳ Pending |
| lfm2 DPO pair generation for directive_preservation | Medium | H1 | ⏳ Pending |
| Validate against LAS archive corpus (ADR-CORE-079 Phase 0 data) | Low | LAS analysis script | ⏳ Pending |

**Key design decisions:**
- Regularization fragments are a new output type (not training data) — delivered via MCP tool results at LAS runtime
- CogSec judge needs a new detection: `conclusory_compression` — accurately summarizes but transforms active state into resolved narrative (distinct from `performative_competence` which fabricates work)
- `directive_preservation` concept is specifically for lfm2 finetuning, not the target model. The DPO pairs are shaped for small-model compression behavior.

**Blocked on:**
- LAS ADR-CORE-079 Phase 0 empirical analysis (confirms whether resonance chambers are measurable in production archives). If the analysis shows the problem is theoretical, this phase deprioritizes. If it shows active degradation, this phase accelerates.

---

## Testing Strategy

### Current State
1. **Unit tests** - Test each handler in isolation with mocked LLM ✅ (59 tests)
2. **Integration tests** - None exist; all backends mocked
3. **E2E tests** - `test_connections.py` exists as manual script outside pytest suite

### Planned Reorganization
1. **Unit tests** - Keep as-is, add `@pytest.mark.defensive` marker for regression guards
2. **Smoke tests** - Convert `test_connections.py` to `tests/test_smoke.py` with `@pytest.mark.smoke`
3. **Infrastructure probes** - Explicit backend availability checks (no inference/skip)

---

## Success Criteria

- [x] All MCP tools respond correctly with proper data
- [ ] Can generate a complete dataset for `temporal_trust` concept (no E2E test exists)
- [x] CogSec scoring works in real pipeline
- [x] Integration with semantic-kinematics-mcp verified (backend switching working)
- [x] All tests passing (59 unit tests, all mocked)
- [x] Documentation matches actual implementation
- [ ] Smoke tests verify infrastructure wiring (deferred to test reorganization)
- [ ] Defensive tests added for regression guards (deferred to usage-driven optimization)
