# YouTube Learning Companion Execution Plan

## Project Boundary

Keep this project standalone in `Downloads/Projects/youtube_learning_companion`. Do not modify the core VIDEX repository. Any optional VIDEX integration must remain explicitly documented and optional. Preserve user files, transcripts, generated notes, and local configuration; do not commit, push, or run destructive cleanup commands.

## Priorities

1. Verify the extracted project and its package boundary.
2. Audit transcript ingestion, completeness reporting, evidence-grounded synthesis, and Markdown export against the implementation.
3. Add only high-value regression tests and evaluation scaffolding; do not invent ground truth or quality scores.
4. Run single-lecture acceptance only with a complete transcript from the user or another authorized source. Never use condensed fixtures as proof of full-lecture behavior.
5. Defer playlist processing until single-lecture acceptance passes.

## Engineering and Safety Rules

- Do not expose secrets or copy `.env`, credentials, caches, virtual environments, or model weights into deliverables.
- Distinguish real provider output, mock output, deterministic fallback, and cached output.
- Do not fabricate transcript content, evidence, citations, timestamps, completeness, or test results.
- Preserve source text and timestamps during parsing and report uncertainty honestly.
- Respect platform terms, copyright, authentication, and access controls.
- Prefer focused changes and validate each change with focused tests before broader checks.

## Phase Status

- Phase 0, standalone extraction verification: completed 2026-10-21. Editable install, package import origin, test suite, Ruff, Mypy, and fixture-backed CLI smoke checks were exercised. Results and blockers are recorded in the Copilot session report.
- Phase 1, source-grounded workflow audit: completed for the local single-transcript flow; unimplemented capabilities and failure behavior are summarized in `docs/architecture.md`.
- Phase 2, transcript honesty: partially completed. JSON, VTT, and SRT validate timestamps/order and preserve JSON source text/IDs; declared coverage is explicit. Timestamped TXT still uses synthetic intervals for untimestamped lines and needs a separate review.
- Phase 3-4, evidence and synthesis: partially completed. LLM evidence links now resolve to segment IDs, timestamps, and source snippets; model timestamps outside a chunk are rejected. Consolidation and semantic claim support still need deeper audit.
- Phase 5, human evaluation scaffolding: completed as instructions/schema only; scoring is blocked pending authorized transcripts and human annotations.
- Phase 6, complete-transcript acceptance: blocked because no complete authorized transcript is available. Fixture-backed checks are not full-lecture acceptance.
- Phase 7, Obsidian export: partial. Frontmatter, provenance, coverage warning, timestamps, segment IDs, and explicit overwrite policy are implemented.
- Phase 8, playlists: deferred until single-lecture acceptance passes.
- Phase 9, packaging/docs: partial. Local installation, tests, Ruff, and Mypy were verified; repository setup/publishing remains user-controlled.

## Phase 0 Caveats

The existing 12-segment fixture has 351 words over 22:11 and full temporal span, but coverage is unknown absent source attestation. It is not suitable for full-lecture acceptance. VIDEX is not installed in the standalone environment; its adapters remain optional. The README Python range now matches `pyproject.toml` (`>=3.11`).
