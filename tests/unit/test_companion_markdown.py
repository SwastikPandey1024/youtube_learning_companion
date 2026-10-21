"""Unit tests for Markdown and Obsidian exporter."""

from pathlib import Path

import pytest

from youtube_learning_companion.markdown import MarkdownExporter, format_seconds, sanitize_filename
from youtube_learning_companion.schemas import (
    PlaylistProcessingManifest,
    YouTubeVideoMetadata,
)
from youtube_learning_companion.synthesizer import StudyNoteSynthesizer
from youtube_learning_companion.transcript import FileTranscriptProvider

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "companion"


class TestMarkdownExporter:
    """Test Markdown formatting, Obsidian callouts, and index generation."""

    def test_format_seconds(self):
        assert format_seconds(45.0) == "00:45"
        assert format_seconds(125.0) == "02:05"
        assert format_seconds(3725.0) == "01:02:05"

    def test_sanitize_filename(self):
        assert sanitize_filename("1a. Why Study Linear Algebra?") == "1a._Why_Study_Linear_Algebra"
        assert sanitize_filename("Matrices: [A] & {B}") == "Matrices_A_&_B"

    def test_markdown_note_generation(self, tmp_path):
        meta = YouTubeVideoMetadata(
            video_id="DrCeIbpfuzE",
            title=(
                "1a. Why Study Linear Algebra | The Importance of Linear Algebra in "
                "Machine Learning | Sachin Mittal"
            ),
            source_url="https://www.youtube.com/watch?v=DrCeIbpfuzE",
            channel_title="GO Classes",
            duration_seconds=1331.0,
        )
        json_file = FIXTURES_DIR / "DrCeIbpfuzE.json"
        doc = FileTranscriptProvider(file_map={"DrCeIbpfuzE": json_file}).get_transcript(
            "DrCeIbpfuzE"
        )
        note = StudyNoteSynthesizer().synthesize(meta, doc)

        exporter = MarkdownExporter(enable_obsidian=True)
        md_text = exporter.export_note_to_markdown(note)

        # 1. Frontmatter
        assert "---" in md_text
        assert 'video_id: "DrCeIbpfuzE"' in md_text
        assert "linear-algebra" in md_text

        # 2. Clickable Timestamps
        assert "https://youtube.com/watch?v=DrCeIbpfuzE&t=" in md_text

        # 3. Obsidian Callouts
        assert "> [!note] Definition:" in md_text
        assert "> [!caution] GATE Exam Trap" in md_text
        assert "> [!example] Worked Example" in md_text
        assert "segment `" in md_text
        assert "generated_at:" in md_text
        assert 'input_provenance: "user_file_json"' in md_text

        # 4. LaTeX blocks
        assert "$$" in md_text

        # 5. File writing
        out_file = exporter.write_note_file(note, tmp_path)
        assert out_file.exists()
        assert out_file.stat().st_size > 1000
        original_content = out_file.read_text(encoding="utf-8")

        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            exporter.write_note_file(note, tmp_path)
        assert out_file.read_text(encoding="utf-8") == original_content

        assert exporter.write_note_file(note, tmp_path, overwrite=True) == out_file

    def test_frontmatter_escapes_multiline_and_quoted_title(self):
        meta = YouTubeVideoMetadata(
            video_id="video_1",
            title='A "quoted" title\ncontinued',
            source_url="https://youtube.com/watch?v=video_1",
        )
        note = StudyNoteSynthesizer().synthesize(meta, transcript_doc=None)

        markdown = MarkdownExporter().export_note_to_markdown(note)

        assert 'title: "A \\"quoted\\" title\\ncontinued"' in markdown
        assert markdown.startswith("---\n")

    def test_playlist_and_topic_index_generation(self):
        meta = YouTubeVideoMetadata(
            video_id="DrCeIbpfuzE",
            title="1a. Why Study Linear Algebra",
            source_url="https://www.youtube.com/watch?v=DrCeIbpfuzE",
            duration_seconds=1331.0,
        )
        json_file = FIXTURES_DIR / "DrCeIbpfuzE.json"
        doc = FileTranscriptProvider(file_map={"DrCeIbpfuzE": json_file}).get_transcript(
            "DrCeIbpfuzE"
        )
        note = StudyNoteSynthesizer().synthesize(meta, doc)

        manifest = PlaylistProcessingManifest(
            playlist_id="PLgjejdknTfWP9cYIHjxBRRdtGcUBLmCQC",
            playlist_title="GATE Linear Algebra",
            source_url="https://youtube.com/playlist?list=PLgjejdknTfWP9cYIHjxBRRdtGcUBLmCQC",
            total_videos_discovered=1,
            processed_count=1,
        )

        exporter = MarkdownExporter()
        pl_index = exporter.generate_playlist_index(manifest, [note])
        assert "# 📚 GATE Linear Algebra" in pl_index
        assert "| # | Video Title | Duration | Status | Study Note | Source Link |" in pl_index

        topic_index = exporter.generate_topic_index([note])
        assert "# 🔍 GATE Linear Algebra — Topic & Concept Index" in topic_index
        assert "Definition" in topic_index or "Formula" in topic_index
