# YouTube Learning Companion Architecture

This repository is an independent Python package. The architecture below describes implemented local paths; optional integrations and unimplemented behavior are called out explicitly. No core VIDEX code belongs in this repository.

## Single-Transcript Path

| Stage | Implementation | Input and output | Failure behavior and limits |
|---|---|---|---|
| Metadata | `schemas.YouTubeVideoMetadata`; CLI constructs it | User-supplied video ID, title, URL, optional duration | No automatic metadata lookup is performed by `run_companion.py`. |
| Transcript selection | `transcript.FileTranscriptProvider`; `CompositeTranscriptProvider` | A video ID maps to a local JSON, VTT, SRT, or timestamped TXT file | Missing or invalid file returns `None`; provider exceptions are logged by composite provider. YouTube caption acquisition currently returns `None` and does not download captions. |
| Parsing | `transcript.parse_timestamp_seconds` and file-specific parsers | UTF-8 transcript file to `TranscriptDocument` and ordered `TranscriptSegment` records | JSON rejects missing/invalid timing, empty text, wrong video IDs, and out-of-order segments. VTT/SRT support is a focused parser, not a full standards implementation. TXT supports timestamped lines and a legacy untimestamped-line convention; confirm coverage manually. |
| Coverage audit | `transcript_audit.audit_transcript` | Transcript and optional declared video duration to counts, span/density diagnostics, gaps, overlaps, duplicates, out-of-range count, supportability | Speech rate and span are diagnostics only. Full-note eligibility requires source-declared complete coverage plus a rationale, sufficient temporal span, limited gaps, and no out-of-range cues. |
| Windowing | `StudyNoteSynthesizer._partition_into_windows` | Ordered segments to deterministic time windows | Segment IDs and original timestamps are retained as source evidence; windows are local to one video synthesis invocation. |
| Section synthesis | `synthesizer.StudyNoteSynthesizer`; prompt builders; `llm` providers | Windowed transcript evidence to structured sections | Structured parsing and note validation run after generation. Provider failure may fall back deterministically if configured; fallback mode and error are recorded. Mock mode is separately identified. |
| Consolidation | `build_consolidation_prompt` and synthesis consolidation path | Section summaries to overview, objectives, takeaways, and questions | Consolidation outputs are not a human correctness check. Evidence for model-produced global items requires scrutiny; do not equate presence of references with semantic support. |
| Evaluation | `evaluation.evaluate_study_note`; synthesizer validation | Study note and optional reference concepts to automated structural metrics | Automated checks validate references, bounds, LaTeX balance, and string-level concept coverage; they do not establish factual correctness. No human-reviewed ground truth is currently included. |
| Export | `markdown.MarkdownExporter` | Study note to Obsidian-compatible plain Markdown | Timestamp links are generated from section/evidence references. Note filename derives from title. Existing same-name files are currently overwritten; use a new output directory to avoid replacement. |
| CLI | `scripts/audit_transcript.py`, `scripts/run_companion.py` | Local transcript file plus CLI metadata/provider options | Mock is the synthesis CLI default. Metadata acquisition, YouTube caption download, playlist execution from the CLI, and acceptance reports are not implemented in these entry points. |

## Data and Provenance

- `TranscriptSegment` carries video ID, segment ID, start/end seconds, source text, and optional speaker/confidence.
- JSON segment IDs are preserved when supplied; otherwise IDs are generated. JSON transcript documents may declare `coverage_status` as `complete` or `partial` only when a non-empty `coverage_rationale` is supplied. Default status is `unknown`.
- LLM-produced evidence intervals must fall within their supplied chunk and overlap a source segment. References resolve to source segment IDs, original source timestamps, and source text; there is no timestamp clamping.
- VTT, SRT, and TXT currently produce unknown coverage because these formats do not carry the project’s coverage attestation contract.
- Notes record completeness classification and rationale, provider/model/mode metadata, and evidence references. An evidence reference is a provenance pointer, not an independent proof that the claim is semantically supported.
- The cache is an in-memory section cache scoped to a `StudyNoteSynthesizer` instance. It is not a persistent cross-process cache.

## Optional VIDEX Interface

`VIDEXASRTranscriptProvider` optionally imports `videx.audio.pipeline.AudioPipeline` to transcribe authorized local media. `TranscriptSegment.to_domain_segment()` optionally converts to a VIDEX domain object. Neither feature is required by standalone parsing, synthesis, export, or tests; the interop method raises an actionable `ImportError` if VIDEX is absent.

## Not Implemented or Not Verified

- Native YouTube caption download or authenticated transcript retrieval.
- Source verification tooling for a user’s complete/partial coverage declaration.
- Persistent cache invalidation across process restarts.
- Independent verification of source coverage declarations.
- Timestamped TXT parsing still has a legacy synthetic-interval behavior for untimestamped lines and requires review before treating it as an evidence-preserving format.
- Human-reviewed quality dataset, accuracy scores, and source-aligned acceptance report.
- Full-lecture single-note acceptance: available fixture data is condensed and must not be used as proof.
- Playlist processing acceptance: playlist code exists, but it must remain deferred until a real single-lecture acceptance passes.
