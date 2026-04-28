<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.

## Coding Philosophy & Execution Rules

Code must be simple, efficient, scalable, and production-ready.

### Core Principles

- Prioritize readability over cleverness.
- Keep logic explicit, flat, use early returns.
- Avoid duplicated logic and unnecessary abstractions. Split files only if it improves clarity.

### Performance & Scalability

- Optimize from the start (CPU, GPU, memory, I/O).
- Use parallelism, batching, vectorization when applicable.
- Consider large-scale data (lazy-loading).
- Balance scalability with low overhead for small inputs.

### Logging & Observability

- Use structured logging (e.g., self.logger).
- Logs must indicate: what, where, why it failed.
- Do NOT over-log.

### Error Handling

- FAIL FAST for internal errors (bad logic, invalid inputs).
- Do NOT hide bugs with fallbacks.
- try/except only when necessary (~1 per function).
- Fallbacks/retries ONLY for external failures (network, APIs, infra).
- Retries must be explicit, limited, logged.

### Modularity & OOP

- Use OOP when it improves structure. Each class/function must have a clear responsibility. Avoid monolithic "god objects".

### Production Readiness

- Code must be robust in real environments.
- Ensure debuggable through logs.
- Handle edge cases that occur in production.

### Full Stack Changes

A configuration or feature change must propagate through ALL affected layers:

- Identify every component the change touches
- Update each layer in the correct order
- Missing a layer = broken feature

Always map the full chain before implementing.

### Tech Stack

- **Local Dev**: conda env `ml-platform` (Python libs pre-installed)
- **Infra**: Kubernetes (kubeadm, Docker Desktop, WSL2)
- **Stack**: Frontend → Backend → Airflow → SkyPilot

### Final Rule

If code is hard to read, inefficient, not scalable, or hard to debug → it is WRONG.