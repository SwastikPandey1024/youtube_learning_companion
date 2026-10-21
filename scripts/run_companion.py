"""CLI script to synthesize a study note from a transcript file or URL.

Usage:
    python scripts/run_companion.py tests/fixtures/companion/DrCeIbpfuzE.json \
        --video-id DrCeIbpfuzE \
        --title "1a. Why Study Linear Algebra" \
        --duration 1331.0 \
        --output-dir data/notes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

from youtube_learning_companion.acceptance import build_acceptance_report
from youtube_learning_companion.evaluation import evaluate_study_note
from youtube_learning_companion.llm import LLMError, create_llm_client
from youtube_learning_companion.markdown import MarkdownExporter, sanitize_filename
from youtube_learning_companion.schemas import YouTubeVideoMetadata
from youtube_learning_companion.synthesizer import StudyNoteSynthesizer
from youtube_learning_companion.transcript import FileTranscriptProvider
from youtube_learning_companion.transcript_audit import audit_transcript


def _human_review_status(value: str) -> Literal["not_run", "pass", "fail"]:
    if value == "pass":
        return "pass"
    if value == "fail":
        return "fail"
    return "not_run"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate evidence-grounded study notes.")
    parser.add_argument(
        "transcript_path",
        type=Path,
        help="Path to transcript .json, .vtt, .srt, or timestamped .txt file",
    )
    parser.add_argument(
        "--video-id", type=str, default="sample_lecture", help="YouTube video identifier"
    )
    parser.add_argument("--title", type=str, default="Sample Lecture", help="Video lecture title")
    parser.add_argument(
        "--duration", type=float, default=1331.0, help="Video duration in seconds"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/notes"), help="Output directory"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        help="LLM provider ('mock', 'ollama', 'openai')",
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model name")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing note with the same title"
    )
    parser.add_argument(
        "--acceptance-report",
        type=Path,
        default=None,
        help="Write a JSON acceptance report; refuses to overwrite an existing report",
    )
    parser.add_argument(
        "--human-review-status",
        choices=("not-run", "pass", "fail"),
        default="not-run",
        help="Record a human factual-review result in the acceptance report",
    )
    parser.add_argument("--reviewer", type=str, default=None, help="Human reviewer identifier")
    parser.add_argument(
        "--review-notes",
        type=str,
        default=None,
        help="Review basis, reference source, and limitations",
    )

    args = parser.parse_args()

    if args.human_review_status != "not-run" and not (
        args.reviewer and args.reviewer.strip() and args.review_notes and args.review_notes.strip()
    ):
        parser.error("--reviewer and --review-notes are required when recording a review result")
    if args.human_review_status != "not-run" and not args.acceptance_report:
        parser.error("--acceptance-report is required when recording a human review result")

    if args.acceptance_report and args.acceptance_report.exists():
        print(
            f"Error: Refusing to overwrite existing acceptance report: {args.acceptance_report}",
            file=sys.stderr,
        )
        return 1
    if args.acceptance_report:
        expected_note = args.output_dir / f"{sanitize_filename(args.title)}.md"
        if args.acceptance_report.resolve() == expected_note.resolve():
            print(
                "Error: Acceptance report path must differ from the Markdown note path.",
                file=sys.stderr,
            )
            return 1

    if not args.transcript_path.exists():
        print(f"Error: Transcript file not found: {args.transcript_path}", file=sys.stderr)
        return 1

    provider = FileTranscriptProvider(file_map={args.video_id: args.transcript_path})
    doc = provider.get_transcript(args.video_id)

    if doc is None:
        print(f"Error: Failed to parse transcript: {args.transcript_path}", file=sys.stderr)
        return 1

    meta = YouTubeVideoMetadata(
        video_id=args.video_id,
        title=args.title,
        source_url=f"https://www.youtube.com/watch?v={args.video_id}",
        duration_seconds=args.duration,
    )
    audit = audit_transcript(doc, video_duration=meta.duration_seconds, video_title=meta.title)
    print(
        f"Transcript audit: {audit.coverage_status.value}; "
        f"{audit.segment_count} segments, {audit.total_words} words; "
        f"full-lecture eligible={audit.can_support_full_lecture_notes}"
    )

    llm_client = create_llm_client(provider_name=args.provider, model=args.model)
    synthesizer = StudyNoteSynthesizer(
        llm_client=llm_client,
        fallback_to_deterministic=args.acceptance_report is None,
    )

    print(f"Synthesizing study note for {meta.title} (provider={args.provider})...")
    try:
        note = synthesizer.synthesize(meta, doc)
    except LLMError as error:
        if args.acceptance_report:
            report = build_acceptance_report(
                transcript=doc,
                video=meta,
                audit=audit,
                provider_name=args.provider,
                model_name=args.model,
                note=None,
                evaluation=None,
                markdown_path=None,
                report_path=args.acceptance_report,
                synthesis_error_type=type(error).__name__,
                human_review_status=_human_review_status(args.human_review_status),
                human_reviewer=args.reviewer,
                human_review_notes=args.review_notes,
            )
            args.acceptance_report.parent.mkdir(parents=True, exist_ok=True)
            args.acceptance_report.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"Acceptance report status: {report['overall_status']}")
            print(f"Acceptance report: {args.acceptance_report}")
        print(f"Error: Synthesis failed ({type(error).__name__}).", file=sys.stderr)
        return 1

    exporter = MarkdownExporter(enable_obsidian=True)
    try:
        out_file = exporter.write_note_file(
            note, output_dir=args.output_dir, overwrite=args.overwrite
        )
    except FileExistsError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Exported note to: {out_file}")

    eval_result = evaluate_study_note(note)
    if args.acceptance_report:
        report = build_acceptance_report(
            transcript=doc,
            video=meta,
            audit=audit,
            provider_name=args.provider,
            model_name=args.model,
            note=note,
            evaluation=eval_result,
            markdown_path=out_file,
            report_path=args.acceptance_report,
            human_review_status=_human_review_status(args.human_review_status),
            human_reviewer=args.reviewer,
            human_review_notes=args.review_notes,
        )
        args.acceptance_report.parent.mkdir(parents=True, exist_ok=True)
        args.acceptance_report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Acceptance report status: {report['overall_status']}")
        print(f"Acceptance report: {args.acceptance_report}")
    print("=" * 60)
    print("Study Note Evaluation Summary:")
    print(f"  Total Claims:            {eval_result.total_claims}")
    print(f"  Grounded Claims:         {eval_result.grounded_claims}")
    print(f"  Unsupported Claims:      {eval_result.unsupported_claims}")
    print(f"  Total Timestamps:        {eval_result.total_timestamps}")
    print(f"  Valid Timestamps:        {eval_result.valid_timestamps}")
    print(f"  Timestamp Accuracy Rate: {eval_result.timestamp_accuracy_rate:.1%}")
    print(f"  LaTeX Equation Fidelity: {eval_result.latex_equation_fidelity:.1%}")
    print(f"  Structural Gate Acceptable: {eval_result.is_acceptable}")
    print("  Note: Structural checks do not establish factual correctness.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
