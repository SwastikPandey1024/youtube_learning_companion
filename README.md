# YouTube Learning Companion

A standalone Python package and CLI for turning user-provided timestamped transcripts into study-note drafts with source-linked evidence, transcript diagnostics, and Markdown export. The core VIDEX project is out of scope; optional VIDEX adapters are isolated integrations only.

---

## 1. Scope & Capabilities

- **Transcript inputs**: User-supplied JSON, WebVTT, SubRip, and timestamped plain text; local ASR is an optional VIDEX adapter.
- **Transcript diagnostics**: Word count, temporal span, density, gaps, overlaps, duplicates, and out-of-range cues. Completeness remains `unknown` unless JSON explicitly declares coverage and gives a rationale; low speech density is only a diagnostic.
- **Synthesis**: Deterministic extraction or configured mock, Ollama, and OpenAI-compatible providers. Provider failure may produce a clearly labelled deterministic fallback when enabled.
- **Evidence and export**: Structured notes retain evidence references and export to Markdown with clickable timestamps and coverage warnings.
- **Cache**: Section cache is in-memory per synthesizer instance; it does not persist between CLI processes.

Automated grounding and format checks are not human verification of factual correctness. No reviewed quality scores are currently available; see [docs/quality_evaluation.md](docs/quality_evaluation.md).

---

## 2. Installation & Setup

### Prerequisites
- Python 3.11 or newer (package metadata currently declares `>=3.11`)
- A clean virtual environment (recommended)

### Installation
From the project root directory:

```bash
# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install package in editable mode with development dependencies:
pip install -e ".[dev]"
```

Alternatively, install runtime dependencies using `requirements.txt` (this does not install the package):
```bash
pip install -r requirements.txt
```

### Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Available configuration settings:
| Setting | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | Provider: `mock`, `ollama`, or `openai` | `openai` in settings; CLI defaults to `mock` |
| `LLM_MODEL` | Model identifier | `gpt-4o` |
| `LLM_API_BASE_URL` | OpenAI-compatible endpoint | `https://api.openai.com/v1` |
| `LLM_API_KEY` | Provider key; never put it in source control | unset |
| `LLM_TIMEOUT_SECONDS` | Request timeout | `45.0` |
| `LLM_MAX_RETRIES` | Retry limit | `2` |
| `YOUTUBE_API_KEY` | Optional YouTube Data API v3 key for playlist/video metadata only | unset |
| `NOTES_EXPORT_DIR` | Default export directory for orchestrator | `data/notes` |

The CLI’s `--provider` and `--model` options override provider selection. No paid provider is required for tests; use `--provider mock` for an offline CLI run.
The YouTube Data API key does not grant caption or media-stream access.

---

## 3. Supported Transcript Input Formats

The companion ingests four timestamped transcript formats:

1. **WebVTT (`.vtt`)**:
   Standard WebVTT subtitle files with cue timestamps and optional speech cues.
2. **SubRip (`.srt`)**:
   Standard SRT subtitle files with `HH:MM:SS,mmm --> HH:MM:SS,mmm` cues.
3. **Structured JSON (`.json`)**:
   List of segments or a dict with a `"segments"` key:
   ```json
   {
     "segments": [
       {
         "segment_id": "optional-stable-id",
         "text": "Welcome to Linear Algebra.",
         "start": 0.0,
         "end": 4.5,
         "start_timestamp_is_source": true,
         "end_timestamp_is_source": true,
         "confidence": 0.95
       }
     ],
     "coverage_status": "unknown",
     "coverage_rationale": null
   }
   ```
   Endpoint provenance flags are optional and default to source-provided for legacy files.
4. **Timestamped plain text (`.txt`)**:
   One cue per line in `[HH:MM:SS.mmm --> HH:MM:SS.mmm] transcript text` form.
   The range endpoints must be present; plain un-timestamped text is rejected.

> [!IMPORTANT]
> **Transcript Coverage Notice:** A transcript spanning the video timeline or having a typical speech rate does not prove completeness. JSON may explicitly set `coverage_status` to `complete` or `partial` only with a non-empty `coverage_rationale`; otherwise coverage is `unknown`. Speech rate and temporal span remain diagnostic heuristics. Current VTT/SRT/TXT parsers do not infer or establish completeness. The repository fixtures are short or condensed test data, not complete lecture transcripts.

Example JSON with an explicit source attestation:

```json
{
   "video_id": "VIDEO_ID",
   "coverage_status": "complete",
   "coverage_rationale": "Compared against the authorized source and verified complete by the operator.",
   "segments": [
      {"segment_id": "seg-1", "start": 0.0, "end": 4.5, "text": "Source transcript text."}
   ]
}
```

The attestation is an operator/source declaration, not independently verified by software.

---

## 4. Running Tests & Quality Checks

Run all unit and integration tests from the package root:

```bash
# Run full test suite
pytest -v

# Run unit tests only
pytest tests/unit/ -v

# Run acceptance integration test
pytest tests/integration/test_companion_acceptance.py -v

# Run code style & lint checks
ruff check src tests

# Run type checks
mypy src
```

---

## 5. Usage & CLI Workflows

### A. Audit a Transcript
Check speech rate, timeline gaps, overlaps, and completeness classification:

```bash
python scripts/audit_transcript.py tests/fixtures/companion/DrCeIbpfuzE.json --video-id DrCeIbpfuzE --duration 1331.0
```

### B. Inventory a Playlist
Fetch ordered playlist items from the YouTube Data API, enrich durations through
`videos.list`, and assess exact-ID local transcript matches:

```bash
python scripts/inventory_playlist.py \
  "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

The JSON report is written to a unique directory under the system temporary
directory by default. `enumeration_status` and `pagination_complete` describe
whether playlist pagination actually finished; `inventory_status` also reflects
metadata-enrichment gaps. The API's declared item count is included only as a
comparison, not as proof. Duplicate playlist entries remain separate rows.
Transient timeouts/rate limits receive bounded retries;
quota and other failures are reported as non-sensitive categories with partial
items retained. Metadata failures do not silently trigger public-page fallback.

An authorized, credential-free metadata export may be used when the API is
unavailable. It must contain `playlist_id`, `pagination_complete`, and `videos`;
each video has a zero-based `position`, optional `video_id`, `title`, and
`duration_seconds`. Optional `declared_item_count` and `source_method` fields
are accepted (`authorized_metadata_export`, `youtube_public_playlist_page`,
or `youtube_data_api_v3`):

```json
{
  "playlist_id": "PLAYLIST_ID",
  "title": "Playlist title",
  "declared_item_count": 1,
  "pagination_complete": true,
  "source_method": "authorized_metadata_export",
  "videos": [
    {
      "position": 0,
      "video_id": "VIDEO_ID",
      "title": "Lecture title",
      "duration_seconds": 1200
    }
  ]
}
```

```bash
python scripts/inventory_playlist.py PLAYLIST_ID \
  --metadata-export path/to/authorized-metadata.json
```

Transcript matching uses exact video-ID filenames in `data/transcripts/` and
`tests/fixtures/companion/` by default; add or replace scan roots with repeated
`--transcript-dir` options. A local file match is not evidence of completeness.
Audio and visual processing remain ineligible unless an authorized local media
source is separately established; the YouTube API key does not provide media
access. Inventory output does not include API credentials or raw request URLs.

### C. Run End-to-End Study Note Synthesis
Generate markdown study notes from any transcript file:

```bash
# Using deterministic offline mock provider (no external service required):
python scripts/run_companion.py tests/fixtures/companion/DrCeIbpfuzE.json \
    --title "1a. Why Study Linear Algebra" \
    --video-id "DrCeIbpfuzE" \
    --provider mock \
    --output-dir outputs/mock-smoke \
    --acceptance-report outputs/mock-smoke/acceptance.json

# Using local Ollama:
python scripts/run_companion.py path/to/full_transcript.vtt \
    --title "Lecture 1: Introduction" \
    --video-id "LEC_01" \
    --provider ollama \
    --model llama3.2 \
   --output-dir data/notes
```

The output filename is derived from the title. Existing files are not replaced unless `--overwrite` is supplied.
An acceptance report records input provenance, timing-provenance counts, audit findings,
provider mode, generated artifact paths, structural evaluation, and explicit gate statuses.
Acceptance-report mode disables deterministic fallback after provider errors and refuses to
overwrite an existing report. `PASS` is not possible without a full-lecture eligible transcript,
real-provider synthesis, all structural gates, and a separately performed human factual review.
The bundled condensed fixture therefore produces a `BLOCKED` acceptance outcome; mock runs are
smoke tests only and do not establish factual accuracy.
To record a later reviewer attestation, pass `--human-review-status pass`,
`--reviewer "name"`, and `--review-notes "review source and scope"` with
`--acceptance-report`. The report marks this as user-attested; the harness does not
independently verify the reviewer or methodology.

### D. Programmatic Python API

```python
from pathlib import Path

from youtube_learning_companion.markdown import MarkdownExporter
from youtube_learning_companion.schemas import YouTubeVideoMetadata
from youtube_learning_companion.synthesizer import StudyNoteSynthesizer
from youtube_learning_companion.transcript import FileTranscriptProvider

video_id = "sample_lecture"
provider = FileTranscriptProvider(
   file_map={video_id: Path("tests/fixtures/companion/sample_lecture.vtt")}
)
transcript = provider.get_transcript(video_id)
assert transcript is not None
metadata = YouTubeVideoMetadata(
   video_id=video_id,
   title="Sample Lecture",
   source_url=f"https://www.youtube.com/watch?v={video_id}",
   duration_seconds=90,
)
note = StudyNoteSynthesizer().synthesize(metadata, transcript)
markdown = MarkdownExporter().export_note_to_markdown(note)
print(markdown)
```

---

## 6. Output Artifacts & Directories

- `data/notes/`: Default output directory for generated study notes in Markdown format.
- `data/transcripts/`: Recommended directory for saving raw `.vtt`, `.srt`, or `.json` transcripts.
- Cache: In-memory section cache scoped to a synthesizer instance; no persistent cache directory is currently implemented.
- `outputs/`: Scratch directory for evaluation reports and benchmark runs.

---

## 7. Optional Core VIDEX Integrations

The YouTube Learning Companion is 100% self-contained and operates completely independently of the VIDEX video-intelligence repository.

However, it provides optional plug-and-play interfaces for users who also have `videx` installed:

1. **Local ASR (`VIDEXASRTranscriptProvider`)**: Requires a separately installed, compatible VIDEX package and authorized local media. Without it, this adapter returns no transcript and logs a warning.
2. **Domain interoperability**: `TranscriptSegment.to_domain_segment()` requires VIDEX and raises an actionable `ImportError` when absent. Core file parsing, tests, synthesis, and export do not require VIDEX.

---

## 8. Standalone Project Readiness

This folder contains the package, tests, fixtures, CLI scripts, and metadata for standalone development. A complete authorized lecture transcript and human-reviewed references are not included, so full-lecture acceptance and factual quality scoring remain unverified.
- Complete source code under `src/youtube_learning_companion/`
- Full test suite with unit and integration tests under `tests/`
- Standard fixtures under `tests/fixtures/companion/`
- Ready-to-use CLI scripts under `scripts/`
- Isolated `pyproject.toml` and `requirements.txt`
