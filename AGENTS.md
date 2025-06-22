# AGENTS.md — Contributor Guide  
*(Place this file at the root of any folder where you want the agent to pick up these rules.)*

---

## 1  Project Layout
| Path | Purpose |
|------|---------|
| **main.py** | Single, authoritative entry-point. Keep it up to date with every feature or refactor. |
| **UnitTests/** | Mirrors the source tree; holds **all** unit tests. |
| **Logs/** | Rotating runtime logs (directory is Git-ignored). |

---

## 2  Naming & Style Conventions
- **PascalCase** for *every* new **file**, **variable**, **function**, and **class** name.  
  - *Exception*: keep Python built-ins and third-party identifiers in their original casing.
- Follow PEP 8 for formatting; run&nbsp;`ruff check .` and&nbsp;`mypy .` before committing.

---

## 3  Development Workflow
1. **Implement** the feature/fix, adhering to the naming rules above.  
2. **Add/Update tests** in *UnitTests/*** so coverage matches new logic.  
3. **Run tests**: `pytest UnitTests/` — must be green.  
4. **Lint & type-check**: `ruff` and `mypy` must pass.  
5. **Update main.py** to wire up any new modules or dependencies.  
6. **Run end-to-end**: `python main.py`; fix issues until it exits cleanly.

---

## 4  Logging
- Use the standard `logging` library with a rotating file handler that writes to *Logs/*.  
- Avoid `print` in production paths.

---

## 5  Debugging & Troubleshooting
- On any error, **search the web** (docs, Stack Overflow, etc.) and document non-obvious fixes in code comments.

---

## 6  Secrets Management
- Azure OpenAI credentials are provided via environment variables.  
- **Never** hard-code secrets or commit them to the repo.

---

## 7  Pull-Request Checklist
- ✅ All unit tests pass.  
- ✅ `main.py` runs without error.  
- ✅ No plaintext secrets or log files committed.  
- ✅ PR title format: `[project_name] <concise description>`.

---

*Keep this guide concise; update sections rather than scattering new rules across the codebase.*
