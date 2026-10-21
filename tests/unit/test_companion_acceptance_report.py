"""Tests for explicit and reproducible single-lecture acceptance outcomes."""

from pathlib import Path

from youtube_learning_companion.acceptance import build_acceptance_report
from youtube_learning_companion.evaluation import evaluate_study_note
from youtube_learning_companion.markdown import MarkdownExporter
from youtube_learning_companion.schemas import YouTubeVideoMetadata
from youtube_learning_companion.synthesizer import StudyNoteSynthesizer
from youtube_learning_companion.transcript import FileTranscriptProvider
from youtube_learning_companion.transcript_audit import audit_transcript

FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "companion" / "DrCeIbpfuzE.json"
)


def test_condensed_fixture_report_is_blocked_and_discloses_limits(tmp_path: Path) -> None:
    video = YouTubeVideoMetadata(
        video_id="DrCeIbpfuzE",
        title="1a. Why Study Linear Algebra",
        source_url="https://www.youtube.com/watch?v=DrCeIbpfuzE",
        duration_seconds=1331.0,
    )
    transcript = FileTranscriptProvider(
        file_map={video.video_id: FIXTURE}
    ).get_transcript(video.video_id)
    assert transcript is not None

    audit = audit_transcript(transcript, video_duration=video.duration_seconds)
    note = StudyNoteSynthesizer().synthesize(video, transcript)
    markdown_path = tmp_path / "note.md"
    markdown_path.write_text(MarkdownExporter().export_note_to_markdown(note), encoding="utf-8")
    evaluation = evaluate_study_note(note)

    report = build_acceptance_report(
        transcript=transcript,
        video=video,
        audit=audit,
        provider_name="mock",
        model_name="offline-test",
        note=note,
        evaluation=evaluation,
        markdown_path=markdown_path,
        report_path=tmp_path / "acceptance.json",
    )

    assert report["overall_status"] == "BLOCKED"
    assert report["quality_gates"]["transcript_audit"] == "BLOCKED"
    assert report["quality_gates"]["real_provider_execution"] == "NOT RUN"
    assert report["quality_gates"]["human_factual_review"] == "NOT RUN"
    assert report["input"]["video_id"] == "DrCeIbpfuzE"
    assert report["input"]["segment_count"] == 12
    assert report["input"]["word_count"] == 351
    assert report["input"]["coverage_status"] == "unknown"
    assert report["artifacts"]["markdown_note"] == str(markdown_path)
    assert any("not factual support" in item for item in report["limitations"])

    attested_report = build_acceptance_report(
        transcript=transcript,
        video=video,
        audit=audit,
        provider_name="mock",
        model_name="offline-test",
        note=note,
        evaluation=evaluation,
        markdown_path=markdown_path,
        human_review_status="pass",
        human_reviewer="test reviewer",
        human_review_notes="Test-only attestation.",
    )
    assert attested_report["quality_gates"]["human_factual_review"] == "PASS"
    assert attested_report["overall_status"] == "BLOCKED"


def test_synthesis_failure_report_records_type_without_error_text(tmp_path: Path) -> None:
    video = YouTubeVideoMetadata(
        video_id="DrCeIbpfuzE",
        title="1a. Why Study Linear Algebra",
        source_url="https://www.youtube.com/watch?v=DrCeIbpfuzE",
        duration_seconds=1331.0,
    )
    transcript = FileTranscriptProvider(
        file_map={video.video_id: FIXTURE}
    ).get_transcript(video.video_id)
    assert transcript is not None
    audit = audit_transcript(transcript, video_duration=video.duration_seconds)

    report = build_acceptance_report(
        transcript=transcript,
        video=video,
        audit=audit,
        provider_name="openai",
        model_name="model-name",
        note=None,
        evaluation=None,
        markdown_path=None,
        synthesis_error_type="LLMAuthenticationError",
    )

    assert report["overall_status"] == "FAIL"
    assert report["quality_gates"]["note_generation"] == "FAIL"
    assert report["execution_error"] == {
        "stage": "synthesis",
        "type": "LLMAuthenticationError",
    }
