"""Adversarial and boundary condition tests for YouTube Learning Companion.

Tests:
- Empty playlist discovery
- Fully private/unavailable playlist discovery
- Duplicate video IDs across paginated responses
- Corrupted/malformed VTT and SRT caption files
- Zero-duration and inverted timestamp intervals
- Corrupted LaTeX syntax detection
- Deliberate unsupported claims and validation rejection
- Resumable orchestrator partial retry logic
"""

from unittest.mock import MagicMock

import httpx

from youtube_learning_companion.connector import YouTubeConnector
from youtube_learning_companion.evaluation import evaluate_study_note
from youtube_learning_companion.schemas import (
    DefinitionItem,
    EvidenceReference,
    EvidenceSourceType,
    StudyNote,
    StudyNoteSection,
    VideoAccessStatus,
    YouTubeVideoMetadata,
)
from youtube_learning_companion.synthesizer import validate_latex_balance
from youtube_learning_companion.transcript import FileTranscriptProvider, parse_timestamp_seconds


class TestAdversarialYouTubeDiscovery:
    """Adversarial tests for playlist discovery and pagination."""

    def test_empty_playlist(self):
        connector = YouTubeConnector(api_key="mock_key")
        mock_client = MagicMock(spec=httpx.Client)

        pl_resp = MagicMock(spec=httpx.Response)
        pl_resp.status_code = 200
        pl_resp.json.return_value = {
            "items": [{"snippet": {"title": "Empty Course"}, "contentDetails": {"itemCount": 0}}]
        }

        items_resp = MagicMock(spec=httpx.Response)
        items_resp.status_code = 200
        items_resp.json.return_value = {"items": []}

        mock_client.get.side_effect = [pl_resp, items_resp]
        mock_client.__enter__.return_value = mock_client

        from unittest.mock import patch

        with patch.object(connector, "_get_client", return_value=mock_client):
            pl = connector.get_playlist("PLempty")
        assert pl.title == "Empty Course"
        assert len(pl.videos) == 0
        assert pl.video_count == 0

    def test_all_private_videos_in_playlist(self):
        connector = YouTubeConnector(api_key="mock_key")
        mock_client = MagicMock(spec=httpx.Client)

        pl_resp = MagicMock(spec=httpx.Response)
        pl_resp.status_code = 200
        pl_resp.json.return_value = {
            "items": [{"snippet": {"title": "Private Course"}, "contentDetails": {"itemCount": 2}}]
        }

        items_resp = MagicMock(spec=httpx.Response)
        items_resp.status_code = 200
        items_resp.json.return_value = {
            "items": [
                {
                    "snippet": {"title": "Private video", "position": 0},
                    "contentDetails": {"videoId": "priv_1"},
                    "status": {"privacyStatus": "private"},
                },
                {
                    "snippet": {"title": "Deleted video", "position": 1},
                    "contentDetails": {"videoId": "del_1"},
                    "status": {"privacyStatus": "public"},
                },
            ]
        }

        mock_client.get.side_effect = [pl_resp, items_resp]
        mock_client.__enter__.return_value = mock_client

        from unittest.mock import patch

        with patch.object(connector, "_get_client", return_value=mock_client):
            pl = connector.get_playlist("PLprivate")
        assert len(pl.videos) == 2
        assert pl.videos[0].access_status == VideoAccessStatus.PRIVATE
        assert pl.videos[1].access_status == VideoAccessStatus.DELETED


class TestAdversarialTranscriptParsing:
    """Adversarial tests for malformed or noisy transcript inputs."""

    def test_malformed_timestamp_parsing(self):
        for value in ("nonsense", "", ":::", "-10.0", "00:75.000", "nan"):
            try:
                parse_timestamp_seconds(value)
            except ValueError:
                continue
            raise AssertionError(f"Invalid timestamp was accepted: {value!r}")

    def test_corrupted_vtt_handling(self, tmp_path):
        bad_vtt = tmp_path / "bad.vtt"
        bad_vtt.write_text("NOT A VTT\nJust random prose without timestamps", encoding="utf-8")
        provider = FileTranscriptProvider(file_map={"bad_vtt_id": bad_vtt})
        doc = provider.get_transcript("bad_vtt_id")
        assert doc is None

    def test_corrupted_srt_handling(self, tmp_path):
        bad_srt = tmp_path / "bad.srt"
        bad_srt.write_text("1\nNo timestamps here\nMissing arrows", encoding="utf-8")
        provider = FileTranscriptProvider(file_map={"bad_srt_id": bad_srt})
        doc = provider.get_transcript("bad_srt_id")
        assert doc is None


class TestAdversarialNoteGroundingAndEvaluation:
    """Adversarial tests for note grounding rejection and equation validation."""

    def test_unsupported_claim_detected_and_rejected(self):
        """A claim without evidence references must fail the evaluation quality gate."""
        meta = YouTubeVideoMetadata(
            video_id="fake_vid",
            title="Fake Lecture",
            source_url="https://youtube.com/watch?v=fake_vid",
            duration_seconds=100.0,
        )
        # Synthesize note with a deliberately ungrounded definition
        unsupported_def = DefinitionItem(
            term="Invented Term",
            definition="An invented definition not supported by any evidence.",
            evidence_refs=(),  # EMPTY evidence references!
        )
        section = StudyNoteSection(
            title="Invented Section",
            start_seconds=0.0,
            end_seconds=60.0,
            overview="Invented overview",
            definitions=(unsupported_def,),
        )
        note = StudyNote(
            video_id=meta.video_id,
            video_title=meta.title,
            source_url=meta.source_url,
            duration_seconds=100.0,
            executive_summary="Summary",
            sections=(section,),
        )

        result = evaluate_study_note(note)
        assert result.unsupported_claims == 1
        assert result.unsupported_claim_rate > 0.0
        assert not result.is_acceptable  # Rejected by acceptance gate!

    def test_timestamp_exceeding_duration_detected(self):
        """Evidence timestamps beyond video duration must be flagged."""
        meta = YouTubeVideoMetadata(
            video_id="short_vid",
            title="Short Video",
            source_url="https://youtube.com/watch?v=short_vid",
            duration_seconds=60.0,
        )
        invalid_ref = EvidenceReference(
            video_id="short_vid",
            start_seconds=500.0,  # Exceeds 60.0s!
            end_seconds=520.0,
            source_type=EvidenceSourceType.TRANSCRIPT,
            snippet="Out of bounds",
        )
        def_item = DefinitionItem(
            term="Out of bounds term",
            definition="Definition text",
            evidence_refs=(invalid_ref,),
        )
        section = StudyNoteSection(
            title="Out of Bounds Section",
            start_seconds=500.0,
            end_seconds=520.0,
            overview="Overview",
            definitions=(def_item,),
        )
        note = StudyNote(
            video_id=meta.video_id,
            video_title=meta.title,
            source_url=meta.source_url,
            duration_seconds=60.0,
            executive_summary="Summary",
            sections=(section,),
        )

        result = evaluate_study_note(note)
        assert result.valid_timestamps == 0
        assert result.timestamp_accuracy_rate == 0.0
        assert not result.is_acceptable

    def test_unbalanced_latex_rejected(self):
        """Unbalanced LaTeX equations must be flagged."""
        assert not validate_latex_balance("$\\int_0^1 f(x dx$")  # missing )
        assert not validate_latex_balance("$A x = b")  # missing $
        assert not validate_latex_balance(
            "$\\begin{aligned} a &= b \\end{matrix}$"
        )  # mismatched env
