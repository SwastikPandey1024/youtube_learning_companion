"""Reproducible single-lecture acceptance reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from youtube_learning_companion.evaluation import CompanionEvaluationResult
from youtube_learning_companion.schemas import (
    NoteGenerationStatus,
    StudyNote,
    TranscriptDocument,
    YouTubeVideoMetadata,
)
from youtube_learning_companion.transcript_audit import TranscriptAuditReport

GateStatus = Literal["PASS", "FAIL", "BLOCKED", "NOT RUN"]


def build_acceptance_report(
    *,
    transcript: TranscriptDocument,
    video: YouTubeVideoMetadata,
    audit: TranscriptAuditReport,
    provider_name: str,
    model_name: str,
    note: StudyNote | None,
    evaluation: CompanionEvaluationResult | None,
    markdown_path: Path | None,
    report_path: Path | None = None,
    synthesis_error_type: str | None = None,
    human_review_status: Literal["not_run", "pass", "fail"] = "not_run",
    human_reviewer: str | None = None,
    human_review_notes: str | None = None,
) -> dict[str, Any]:
    """Build an evidence-explicit report without treating structural checks as factual review."""
    if human_review_status != "not_run" and not (
        human_reviewer
        and human_reviewer.strip()
        and human_review_notes
        and human_review_notes.strip()
    ):
        raise ValueError("A reviewer and review notes are required when human review is recorded")

    source_start_count = sum(segment.start_timestamp_is_source for segment in transcript.segments)
    source_end_count = sum(segment.end_timestamp_is_source for segment in transcript.segments)
    inferred_segment_count = sum(
        not (segment.start_timestamp_is_source and segment.end_timestamp_is_source)
        for segment in transcript.segments
    )

    if audit.coverage_status.value == "unknown" or audit.coverage_status.value == "partial":
        transcript_status: GateStatus = "BLOCKED"
    elif audit.can_support_full_lecture_notes:
        transcript_status = "PASS"
    else:
        transcript_status = "FAIL"

    if note is None:
        synthesis_status: GateStatus = "FAIL"
    elif note.status == NoteGenerationStatus.FAILED:
        synthesis_status = "FAIL"
    elif note.status == NoteGenerationStatus.COMPLETE and transcript_status == "PASS":
        synthesis_status = "PASS"
    else:
        synthesis_status = "BLOCKED"

    if evaluation is None or evaluation.total_claims == 0:
        grounding_status: GateStatus = "NOT RUN"
    elif evaluation.unsupported_claims > 0:
        grounding_status = "FAIL"
    else:
        grounding_status = "PASS"

    if evaluation is None or evaluation.total_timestamps == 0:
        timestamp_status: GateStatus = "NOT RUN"
    elif evaluation.valid_timestamps == evaluation.total_timestamps:
        timestamp_status = "PASS"
    else:
        timestamp_status = "FAIL"

    if evaluation is None:
        evaluation_status: GateStatus = "NOT RUN"
    else:
        evaluation_status = "PASS" if evaluation.is_acceptable else "FAIL"

    if markdown_path is None:
        markdown_status: GateStatus = "NOT RUN"
    else:
        markdown_status = "PASS" if markdown_path.is_file() else "FAIL"

    if human_review_status == "not_run":
        review_status: GateStatus = "NOT RUN"
    elif human_review_status == "pass":
        review_status = "PASS"
    else:
        review_status = "FAIL"
    mode = str(note.generation_metadata.get("synthesizer_mode", "")) if note else ""
    is_mock = bool(note.generation_metadata.get("is_mock", False)) if note else False
    actual_provider = (
        str(note.generation_metadata.get("llm_provider", provider_name)) if note else None
    )
    actual_model = (
        str(
            note.generation_metadata.get("model_id")
            or note.generation_metadata.get("llm_model")
            or model_name
        )
        if note
        else None
    )
    if provider_name == "mock" or actual_provider == "mock" or is_mock:
        real_provider_status: GateStatus = "NOT RUN"
    elif mode == "llm":
        real_provider_status = "PASS"
    elif note is None:
        real_provider_status = "FAIL"
    else:
        real_provider_status = "BLOCKED"

    gates: dict[str, GateStatus] = {
        "transcript_audit": transcript_status,
        "note_generation": synthesis_status,
        "structural_grounding": grounding_status,
        "source_timestamps": timestamp_status,
        "automated_evaluation": evaluation_status,
        "markdown_export": markdown_status,
        "real_provider_execution": real_provider_status,
        "human_factual_review": review_status,
    }
    if "FAIL" in gates.values():
        overall_status: GateStatus = "FAIL"
    elif "BLOCKED" in gates.values() or "NOT RUN" in gates.values():
        overall_status = "BLOCKED"
    else:
        overall_status = "PASS"

    first = transcript.segments[0] if transcript.segments else None
    last = transcript.segments[-1] if transcript.segments else None
    return {
        "schema_version": 1,
        "overall_status": overall_status,
        "input": {
            "video_id": video.video_id,
            "video_title": video.title,
            "source_url": video.source_url,
            "transcript_source": transcript.source_path_or_url,
            "acquisition_method": transcript.acquisition_method.value,
            "language": transcript.language,
            "is_auto_generated": transcript.is_auto_generated,
            "duration_seconds": video.duration_seconds,
            "duration_basis": (
                "CLI --duration value or its script default; verify independently before "
                "acceptance"
            ),
            "coverage_status": transcript.coverage_status.value,
            "coverage_rationale": transcript.coverage_rationale,
            "segment_count": len(transcript.segments),
            "word_count": transcript.word_count,
            "time_span_seconds": (
                round(last.end_seconds - first.start_seconds, 3) if first and last else 0.0
            ),
            "time_span_endpoints_source_verified": bool(
                first
                and last
                and first.start_timestamp_is_source
                and last.end_timestamp_is_source
            ),
            "temporal_coverage_fully_source_timed": inferred_segment_count == 0,
            "timestamp_provenance": {
                "source_start_endpoints": source_start_count,
                "inferred_start_endpoints": len(transcript.segments) - source_start_count,
                "source_end_endpoints": source_end_count,
                "inferred_end_endpoints": len(transcript.segments) - source_end_count,
                "segments_with_inferred_timing": inferred_segment_count,
            },
        },
        "audit": audit.model_dump(mode="json"),
        "provider": {
            "requested_provider": provider_name,
            "requested_model": model_name,
            "actual_provider": actual_provider,
            "actual_model": actual_model,
            "synthesis_mode": mode or None,
            "is_mock": is_mock,
        },
        "quality_gates": gates,
        "evaluation": evaluation.model_dump(mode="json") if evaluation else None,
        "artifacts": {
            "markdown_note": str(markdown_path) if markdown_path else None,
            "acceptance_report": str(report_path) if report_path else None,
        },
        "execution_error": (
            {"stage": "synthesis", "type": synthesis_error_type}
            if synthesis_error_type
            else None
        ),
        "human_review": {
            "status": review_status,
            "basis": "user_attested" if human_review_status != "not_run" else "not supplied",
            "reference_annotations": "not supplied",
            "reviewer": human_reviewer,
            "notes": human_review_notes,
        },
        "limitations": [
            (
                "Automated evidence-reference checks establish structural traceability, "
                "not factual support."
            ),
            (
                "Transcript completeness declarations and duration were not independently "
                "authenticated."
            ),
            (
                "Human review status is user-attested; this report does not independently verify "
                "the reviewer's source or methodology."
                if human_review_status != "not_run"
                else (
                    "Human review against a verified source or reviewed reference transcript "
                    "was not run."
                )
            ),
            (
                "Temporal span is a diagnostic and does not establish complete transcript coverage."
            ),
        ],
    }
