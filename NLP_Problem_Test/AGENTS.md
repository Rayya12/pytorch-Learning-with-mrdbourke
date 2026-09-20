# AGENTS.md

This file defines how any AI coding agent (Claude Code, Cursor, Copilot Workspace, Codex, etc.) must behave when working in this repository. These rules are mandatory and override default agent behavior unless the user explicitly says otherwise in a given message.

---

## 1. Always Clarify Before Building

Never start writing code, creating files, or scaffolding a project the moment a request comes in — even if the request sounds complete.

**Before any implementation work, the agent MUST:**
- Restate its understanding of the goal in 2–4 sentences.
- List the assumptions it is about to make (tech stack, data shape, scope, edge cases, non-functional requirements like auth/performance/deployment target).
- Ask targeted clarifying questions **only** for things that would materially change the design (e.g. "Should this be a REST API or GraphQL?", "Is this single-user or multi-tenant?", "Do we need persistence, or is in-memory fine for now?").
- Wait for user confirmation or answers before proceeding to design or code.

**Exceptions (agent may proceed without asking):**
- Trivial, low-ambiguity tasks (fixing a typo, renaming a variable, adding a comment).
- The user has explicitly said "just build it" / "use your best judgment" / "don't ask, decide and go."
- A prior message in the same task already answered the open questions.

If ambiguity is minor, the agent should state its default assumption inline and proceed, rather than blocking on trivial ambiguity. Blocking questions are reserved for decisions that are expensive to reverse.

---

## 2. Use Sub-Agents for Token Efficiency

The agent should decompose non-trivial tasks into sub-agents / sub-tasks instead of doing everything in one long, monolithic context.

**Guidelines:**
- **Split by concern, not by file.** A sub-agent should own a coherent unit of work (e.g. "database schema + migrations," "API route handlers," "frontend component tree," "test suite"), not an arbitrary file count.
- **Give each sub-agent a scoped brief**, not the full conversation history: goal, relevant interfaces/contracts (types, API shapes, function signatures), constraints, and expected output format. Do not forward unrelated context.
- **Use a lead/orchestrator agent** to:
  - Hold the overall plan and architecture decisions.
  - Dispatch sub-agents with minimal, targeted context.
  - Integrate sub-agent outputs, resolve conflicts, and run final review.
- **Prefer parallelizable sub-agents** for independent modules (e.g. frontend vs backend, or separate services) to reduce sequential context bloat.
- **Summarize, don't dump.** When a sub-agent finishes, its output should be condensed into an interface/contract summary (what it exposes, how to call it) before being handed to the next sub-agent or the orchestrator — not the full raw implementation reasoning.
- **Avoid re-reading full files repeatedly.** Cache/reference summaries of files already inspected instead of re-fetching full content across sub-agent calls when it can be avoided.

Reserve full single-context execution for small, tightly coupled tasks where splitting would add overhead without saving tokens.

---

## 3. Explain the Design Before Building

Once scope is clarified, the agent must present a design before writing implementation code.

**The design explanation should include:**
- **Approach summary** — the overall strategy in plain language.
- **Architecture/structure** — key components, modules, or files that will be created/changed, and how they relate (a short diagram or bullet tree is fine).
- **Key decisions and trade-offs** — e.g. chosen library vs alternatives, data model choice, why sub-agents are split the way they are.
- **Interfaces/contracts** — function signatures, API endpoints, data schemas — enough that the user could sanity-check correctness without reading code.
- **Risks or open questions**, if any remain.

The agent then pauses for user approval (or proceeds automatically only if the user has pre-approved "explain then go ahead without waiting").

**Only after the design is presented (and approved, where required) does the agent begin implementation.**

---

## Summary Flow

1. **Clarify** → confirm goal, assumptions, and constraints with the user.
2. **Design** → propose architecture, components, and trade-offs; get sign-off.
3. **Delegate** → break the approved design into scoped sub-agent tasks with minimal, targeted context.
4. **Build** → sub-agents implement; orchestrator integrates and reviews.
5. **Report** → summarize what was built, any deviations from the design, and remaining follow-ups.