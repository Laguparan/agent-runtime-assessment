# Part B: Framework Evaluation Report (`FRAMEWORK.md`)

## 1. Three Things the Framework Did Better Than Part A
1. **Schema Validation & Parsing:** Pydantic AI uses native Python type hints and Pydantic models to automatically validate tool inputs, completely removing the manual JSON decoding and trailing comma error-handling code required in Part A (Scenario S2).
2. **Context Management & Prompt Flow:** The framework manages message turns and history natively without requiring a manually maintained procedural loop.
3. **Ergonomic Tool Decorators:** Turning any standard Python function into an LLM-callable tool is reduced to a single `@agent.tool` decorator.

## 2. Three Places the Abstraction Leaked (And What It Cost)
1. **Transport Layer Override:** Pointing Pydantic AI to a local mock server (`http://localhost:8000/v1`) required instantiating a custom `AsyncOpenAI` client override, exposing underlying transport settings.
2. **Telemetry Black-Boxing:** Framework-level retries and error swallowing happened inside internal client layers, making it opaque to capture precise retry counts and exact error traces without deep instrumentation.
3. **Custom Compocation Friction:** Forcing our custom "Anchor-Window" token compaction strategy (R3) required fighting the framework's internal message history store.

## 3. One Thing the Framework Makes Impossible or Unreasonably Expensive
* **Granular Micro-Step Intercepts:** Inserting precise transactional checkpoints and custom database hooks between every single micro-step for the SQLite intent ledger required extensive boilerplate wrapper logic, proving frameworks optimize for speed rather than strict compliance auditing.

## 4. Exit Cost
If we drop Pydantic AI in six months, removal cost is moderate because business logic (tools and validation schemas) is decoupled into standard Python functions. Rewriting the core loop back to a pure native client would take approximately 3–4 hours.

## 5. Final Recommendation
**Verdict for the System in Part A: Do not use a framework.** 
* **Defense:** While Pydantic AI provides excellent type safety, safety-critical agent systems requiring strict deterministic trust boundaries (like exact-once idempotency ledgers and granular step tracking) suffer when upstream abstractions hide low-level control flow. Hand-rolling the runtime gives absolute determinism.