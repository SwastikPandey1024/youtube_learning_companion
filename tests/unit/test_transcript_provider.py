"""Unit tests for transcript provider interfaces and file parsers."""

from pathlib import Path

import pytest

from youtube_learning_companion.schemas import TranscriptAcquisitionMethod
from youtube_learning_companion.transcript import (
    CompositeTranscriptProvider,
    FileTranscriptProvider,
    YouTubeCaptionProvider,
    parse_timestamp_seconds,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "companion"


class TestTranscriptParsers:
    """Test parsing across multiple caption formats."""

    def test_parse_timestamp_seconds(self):
        assert parse_timestamp_seconds("00:00:15.500") == pytest.approx(15.5)
        assert parse_timestamp_seconds("01:23.456") == pytest.approx(83.456)
        assert parse_timestamp_seconds("01:02:03,400") == pytest.approx(3723.4)
        assert parse_timestamp_seconds("15.5") == pytest.approx(15.5)
        with pytest.raises(ValueError, match="Invalid timestamp"):
            parse_timestamp_seconds("invalid")

    def test_parse_json_transcript(self):
        json_file = FIXTURES_DIR / "DrCeIbpfuzE.json"
        provider = FileTranscriptProvider(file_map={"DrCeIbpfuzE": json_file})
        doc = provider.get_transcript("DrCeIbpfuzE")

        assert doc is not None
        assert doc.video_id == "DrCeIbpfuzE"
        assert doc.acquisition_method == TranscriptAcquisitionMethod.USER_FILE_JSON
        assert len(doc.segments) == 12
        assert doc.segments[0].start_seconds == 0.0
        assert doc.segments[0].end_seconds == 45.0
        assert "Welcome to Linear Algebra" in doc.segments[0].text
        assert doc.total_duration_seconds == 1331.0
        assert doc.word_count > 100

    def test_parse_vtt_transcript(self):
        vtt_file = FIXTURES_DIR / "sample_lecture.vtt"
        provider = FileTranscriptProvider(file_map={"vtt_vid": vtt_file})
        doc = provider.get_transcript("vtt_vid")

        assert doc is not None
        assert doc.acquisition_method == TranscriptAcquisitionMethod.USER_FILE_VTT
        assert len(doc.segments) == 3
        assert doc.segments[0].start_seconds == 0.0
        assert doc.segments[0].end_seconds == pytest.approx(15.5)
        assert doc.segments[1].start_seconds == pytest.approx(15.5)
        assert doc.segments[2].end_seconds == pytest.approx(90.0)

    def test_parse_srt_transcript(self):
        srt_file = FIXTURES_DIR / "sample_lecture.srt"
        provider = FileTranscriptProvider(file_map={"srt_vid": srt_file})
        doc = provider.get_transcript("srt_vid")

        assert doc is not None
        assert doc.acquisition_method == TranscriptAcquisitionMethod.USER_FILE_SRT
        assert len(doc.segments) == 3
        assert doc.segments[0].start_seconds == 0.0
        assert doc.segments[0].end_seconds == pytest.approx(15.5)
        assert "foundational pillar" in doc.segments[1].text

    def test_audit_vtt_and_srt_transcripts(self):
        """Verify VTT and SRT parsed documents can be directly audited by audit_transcript."""
        from youtube_learning_companion.transcript_audit import audit_transcript

        vtt_file = FIXTURES_DIR / "sample_lecture.vtt"
        provider = FileTranscriptProvider(file_map={"vtt_vid": vtt_file})
        doc_vtt = provider.get_transcript("vtt_vid")
        assert doc_vtt is not None
        report_vtt = audit_transcript(doc_vtt, video_duration=90.0)
        assert report_vtt.segment_count == 3
        assert report_vtt.start_seconds == 0.0
        assert report_vtt.end_seconds == pytest.approx(90.0)
        assert report_vtt.total_words > 0
        assert report_vtt.acquisition_method == TranscriptAcquisitionMethod.USER_FILE_VTT

        srt_file = FIXTURES_DIR / "sample_lecture.srt"
        provider_srt = FileTranscriptProvider(file_map={"srt_vid": srt_file})
        doc_srt = provider_srt.get_transcript("srt_vid")
        assert doc_srt is not None
        report_srt = audit_transcript(doc_srt, video_duration=90.0)
        assert report_srt.segment_count == 3
        assert report_srt.start_seconds == 0.0
        assert report_srt.end_seconds == pytest.approx(90.0)
        assert report_srt.total_words > 0
        assert report_srt.acquisition_method == TranscriptAcquisitionMethod.USER_FILE_SRT

    def test_missing_transcript_returns_none(self, tmp_path):
        provider = FileTranscriptProvider(transcript_dir=tmp_path)
        doc = provider.get_transcript("non_existent_id")
        assert doc is None

    def test_corrupted_json_handled_safely(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{corrupted: json [", encoding="utf-8")
        provider = FileTranscriptProvider(file_map={"bad_vid": bad_json})
        doc = provider.get_transcript("bad_vid")
        assert doc is None

    def test_json_preserves_segment_text_and_id(self, tmp_path):
        transcript = tmp_path / "source.json"
        transcript.write_text(
            '[{"segment_id":"source-1","start":1,"end":2,"text":"  exact text  "}]',
            encoding="utf-8",
        )
        doc = FileTranscriptProvider(file_map={"video": transcript}).get_transcript("video")

        assert doc is not None
        assert doc.segments[0].segment_id == "source-1"
        assert doc.segments[0].text == "  exact text  "

    @pytest.mark.parametrize(
        "segments",
        [
            [{"start": 0, "text": "Missing end"}],
            [{"start": 3, "end": 2, "text": "Inverted interval"}],
            [{"start": -1, "end": 2, "text": "Negative start"}],
            [
                {"start": 2, "end": 3, "text": "Later"},
                {"start": 1, "end": 2, "text": "Earlier"},
            ],
                [{"video_id": "other-video", "start": 0, "end": 1, "text": "Wrong video"}],
        ],
    )
    def test_invalid_json_timing_fails_safely(self, tmp_path, segments):
        transcript = tmp_path / "invalid.json"
        transcript.write_text(__import__("json").dumps(segments), encoding="utf-8")
        doc = FileTranscriptProvider(file_map={"video": transcript}).get_transcript("video")

        assert doc is None

    @pytest.mark.parametrize("suffix", ["vtt", "srt"])
    def test_caption_files_reject_out_of_order_cues(self, tmp_path, suffix):
        transcript = tmp_path / f"out_of_order.{suffix}"
        if suffix == "vtt":
            content = (
                "WEBVTT\n\n00:00:02.000 --> 00:00:03.000\nLater\n\n"
                "00:00:01.000 --> 00:00:02.000\nEarlier\n"
            )
        else:
            content = (
                "1\n00:00:02,000 --> 00:00:03,000\nLater\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\nEarlier\n"
            )
        transcript.write_text(content, encoding="utf-8")

        doc = FileTranscriptProvider(file_map={"video": transcript}).get_transcript("video")

        assert doc is None


class TestCompositeTranscriptProvider:
    """Test provider chaining and priority fallback."""

    def test_composite_fallback_to_file(self):
        json_file = FIXTURES_DIR / "DrCeIbpfuzE.json"
        file_provider = FileTranscriptProvider(file_map={"DrCeIbpfuzE": json_file})
        yt_provider = YouTubeCaptionProvider()

        composite = CompositeTranscriptProvider(providers=[yt_provider, file_provider])
        doc = composite.get_transcript("DrCeIbpfuzE")

        assert doc is not None
        assert doc.video_id == "DrCeIbpfuzE"
        assert doc.acquisition_method == TranscriptAcquisitionMethod.USER_FILE_JSON

    def test_composite_returns_none_when_all_fail(self):
        composite = CompositeTranscriptProvider(providers=[YouTubeCaptionProvider()])
        doc = composite.get_transcript("any_id")
        assert doc is None
