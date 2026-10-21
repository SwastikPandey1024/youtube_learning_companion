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
import sys
from pathlib import Path

from youtube_learning_companion.evaluation import evaluate_study_note
from youtube_learning_companion.llm import create_llm_client
from youtube_learning_companion.markdown import MarkdownExporter
from youtube_learning_companion.schemas import YouTubeVideoMetadata
from youtube_learning_companion.synthesizer import StudyNoteSynthesizer
from youtube_learning_companion.transcript import FileTranscriptProvider


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate evidence-grounded study notes.")
    parser.add_argument(
        "transcript_path", type=Path, help="Path to transcript .json, .vtt, or .srt file"
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

    args = parser.parse_args()

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

    llm_client = create_llm_client(provider_name=args.provider, model=args.model)
    synthesizer = StudyNoteSynthesizer(llm_client=llm_client)

    print(f"Synthesizing study note for {meta.title} (provider={args.provider})...")
    note = synthesizer.synthesize(meta, doc)

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
    print("=" * 60)
    print("Study Note Evaluation Summary:")
    print(f"  Total Claims:            {eval_result.total_claims}")
    print(f"  Grounded Claims:         {eval_result.grounded_claims}")
    print(f"  Unsupported Claims:      {eval_result.unsupported_claims}")
    print(f"  Total Timestamps:        {eval_result.total_timestamps}")
    print(f"  Valid Timestamps:        {eval_result.valid_timestamps}")
    print(f"  Timestamp Accuracy Rate: {eval_result.timestamp_accuracy_rate:.1%}")
    print(f"  LaTeX Equation Fidelity: {eval_result.latex_equation_fidelity:.1%}")
    print(f"  Quality Gate Acceptable: {eval_result.is_acceptable}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
