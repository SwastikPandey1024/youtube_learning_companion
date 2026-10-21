"""CLI script to audit a transcript file for synthesis readiness.

Usage:
    python scripts/audit_transcript.py tests/fixtures/companion/DrCeIbpfuzE.json --duration 1331.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from youtube_learning_companion.transcript import FileTranscriptProvider
from youtube_learning_companion.transcript_audit import audit_transcript


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a transcript file for completeness and quality."
    )
    parser.add_argument(
        "transcript_path", type=Path, help="Path to .json, .vtt, .srt, or timestamped .txt file"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Video duration in seconds if known"
    )
    parser.add_argument("--video-id", type=str, default="audit_video", help="Video identifier")

    args = parser.parse_args()

    if not args.transcript_path.exists():
        print(f"Error: Transcript file not found: {args.transcript_path}", file=sys.stderr)
        return 1

    provider = FileTranscriptProvider(file_map={args.video_id: args.transcript_path})
    doc = provider.get_transcript(args.video_id)

    if doc is None:
        print(f"Error: Failed to parse transcript from {args.transcript_path}", file=sys.stderr)
        return 1

    report = audit_transcript(doc, video_duration=args.duration)

    span_str = f"{report.transcript_span_seconds:.1f}s / {report.video_duration_seconds:.1f}s"
    overlap_str = f"{report.overlap_count} (max: {report.max_overlap_seconds:.1f}s)"

    print("=" * 60)
    print(f"Transcript Audit Report: {args.transcript_path.name}")
    print("=" * 60)
    print(f"Video ID:                {report.video_id}")
    print(f"Acquisition Method:      {report.acquisition_method}")
    print(f"Language:                {report.language} (auto-gen: {report.is_auto_generated})")
    print(f"Segment Count:           {report.segment_count}")
    print(
        "Timestamp Provenance:    "
        f"start {report.source_start_timestamp_count} source / "
        f"{report.inferred_start_timestamp_count} inferred; "
        f"end {report.source_end_timestamp_count} source / "
        f"{report.inferred_end_timestamp_count} inferred"
    )
    print(f"Inferred-timing Segments:{report.inferred_timing_segment_count:>8}")
    print(f"Word Count:              {report.total_words}")
    print(f"First Timestamp:         {report.start_seconds:.1f}s")
    print(f"Last Timestamp:          {report.end_seconds:.1f}s")
    print(f"Span / Duration:         {span_str}")
    print(f"Temporal Coverage:       {report.temporal_coverage_pct:.1f}%")
    print(f"Speech Rate:             {report.words_per_minute:.1f} WPM")
    print(f"Text Density Ratio:      {report.text_density_ratio:.2f}")
    print(f"Timeline Gaps (>2s):     {report.gap_count} (max: {report.max_gap_seconds:.1f}s)")
    print(f"Timeline Overlaps:       {overlap_str}")
    print(f"Empty Segments:          {report.empty_segment_count}")
    print(f"Duplicate Segments:      {report.duplicate_segment_count}")
    print(f"Out-of-range Segments:   {report.out_of_range_segment_count}")
    print(f"Low-density Diagnostic:  {report.is_abridged}")
    print(f"Coverage Status:         {report.coverage_status}")
    print(f"Coverage Rationale:      {report.coverage_rationale or 'Not provided'}")
    print(f"Eligible for Full Note:  {report.can_support_full_lecture_notes}")
    print(f"Eligible for Synopsis:   {report.can_support_topic_synopsis}")
    print("=" * 60)

    if report.findings:
        print("Findings:")
        for f in report.findings:
            print(f"  - {f}")

    if report.recommendations:
        print("\nRecommendations:")
        for r in report.recommendations:
            print(f"  - {r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
