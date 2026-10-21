"""Integration acceptance tests for VIDEX YouTube Learning Companion.

Verifies:
- First vertical slice: URL -> transcript -> synthesis -> markdown -> evaluation
- Resumable playlist orchestration and manifest tracking
- Skipping already processed videos and retry capability
"""

from pathlib import Path
from unittest.mock import MagicMock

from youtube_learning_companion.connector import YouTubeConnector
from youtube_learning_companion.evaluation import evaluate_study_note
from youtube_learning_companion.markdown import MarkdownExporter
from youtube_learning_companion.orchestrator import PlaylistJobOrchestrator
from youtube_learning_companion.schemas import (
    NoteGenerationStatus,
    PlaylistAccessStatus,
    VideoAccessStatus,
    YouTubePlaylistMetadata,
    YouTubeVideoMetadata,
)
from youtube_learning_companion.synthesizer import StudyNoteSynthesizer
from youtube_learning_companion.transcript import FileTranscriptProvider

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "companion"


class TestCompanionAcceptance:
    """Acceptance test suite for YouTube Learning Companion vertical slice."""

    def test_end_to_end_single_lecture_vertical_slice(self, tmp_path: Path) -> None:
        """Verify complete flow on benchmark lecture DrCeIbpfuzE."""
        # 1. Setup metadata and file provider
        meta = YouTubeVideoMetadata(
            video_id="DrCeIbpfuzE",
            title=(
                "1a. Why Study Linear Algebra | The Importance of "
                "Linear Algebra in Machine Learning | Sachin Mittal"
            ),
            source_url="https://www.youtube.com/watch?v=DrCeIbpfuzE",
            channel_title="GO Classes for GATE CS",
            duration_seconds=1331.0,
            access_status=VideoAccessStatus.ACCESSIBLE,
        )
        json_file = FIXTURES_DIR / "DrCeIbpfuzE.json"
        provider = FileTranscriptProvider(file_map={"DrCeIbpfuzE": json_file})

        # 2. Synthesize note
        synthesizer = StudyNoteSynthesizer()
        doc = provider.get_transcript("DrCeIbpfuzE")
        assert doc is not None

        note = synthesizer.synthesize(meta, doc)
        assert note.status == NoteGenerationStatus.PARTIAL_SUCCESS
        assert note.temporal_coverage_percentage == 100.0
        assert note.transcript_completeness == "unknown"

        # 3. Export to Markdown
        exporter = MarkdownExporter()
        out_file = exporter.write_note_file(note, tmp_path)
        assert out_file.exists()

        content = out_file.read_text(encoding="utf-8")
        assert "Why Study Linear Algebra" in content
        assert "https://youtube.com/watch?v=DrCeIbpfuzE&t=" in content
        assert "$$" in content
        assert "> [!note] Definition:" in content

        # 4. Evaluate against quality gate
        eval_result = evaluate_study_note(
            note, reference_concepts=["vector", "matrix", "linear transformation"]
        )
        assert eval_result.is_acceptable
        assert eval_result.unsupported_claim_rate == 0.0
        assert eval_result.latex_equation_fidelity == 1.0
        assert eval_result.timestamp_accuracy_rate == 1.0

    def test_resumable_playlist_orchestration(self, tmp_path: Path) -> None:
        """Verify multi-video playlist orchestration, manifest tracking, and resumability."""
        # 1. Create mock playlist with 3 diverse videos
        videos = [
            YouTubeVideoMetadata(
                video_id="DrCeIbpfuzE",
                title="Lecture 1 Complete",
                source_url="https://youtube.com/watch?v=vid_complete",
                position=0,
                duration_seconds=600.0,
                access_status=VideoAccessStatus.ACCESSIBLE,
            ),
            YouTubeVideoMetadata(
                video_id="vid_awaiting",
                title="Lecture 2 Awaiting Transcript",
                source_url="https://youtube.com/watch?v=vid_awaiting",
                position=1,
                duration_seconds=600.0,
                access_status=VideoAccessStatus.ACCESSIBLE,
            ),
            YouTubeVideoMetadata(
                video_id="vid_private",
                title="Private video",
                source_url="https://youtube.com/watch?v=vid_private",
                position=2,
                access_status=VideoAccessStatus.PRIVATE,
                error_message="Video is private",
            ),
        ]
        pl_meta = YouTubePlaylistMetadata(
            playlist_id="PLmock",
            title="GATE Linear Algebra Mock Course",
            source_url="https://youtube.com/playlist?list=PLmock",
            video_count=3,
            access_status=PlaylistAccessStatus.ACCESSIBLE,
            videos=tuple(videos),
        )

        mock_connector = MagicMock(spec=YouTubeConnector)
        mock_connector.get_playlist.return_value = pl_meta

        # Provide transcript only for vid_complete
        json_file = FIXTURES_DIR / "DrCeIbpfuzE.json"
        provider = FileTranscriptProvider(file_map={"DrCeIbpfuzE": json_file})

        orchestrator = PlaylistJobOrchestrator(
            connector=mock_connector,
            transcript_provider=provider,
            output_dir=tmp_path,
        )

        # 2. First run
        manifest1 = orchestrator.process_playlist("PLmock")

        assert manifest1.total_videos_discovered == 3
        assert manifest1.processed_count == 1
        assert manifest1.awaiting_transcript_count == 1
        assert manifest1.skipped_count == 1

        # Check files created
        assert (tmp_path / "00_Playlist_Index.md").exists()
        assert (tmp_path / "00_Topic_Index.md").exists()
        assert (tmp_path / "processing_manifest.json").exists()

        # 3. Resumability: second run without retry skips completed video
        manifest2 = orchestrator.process_playlist("PLmock", retry_failed=False)
        assert manifest2.processed_count == 1
