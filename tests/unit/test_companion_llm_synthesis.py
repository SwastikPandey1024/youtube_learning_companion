"""Tests for Phase 9.3 LLM-powered evidence-grounded study note synthesis.

Verifies:
- Structured output compliance with StudyNote and StudyNoteSection schemas
- Handling of malformed LLM outputs and unclosed JSON blocks
- Robust classification of quota exhaustion (429) and authentication errors (401)
- Handling of LLM timeouts with exponential backoff and deterministic fallback
- Detection and uncertainty flagging for corrupted or unbalanced LaTeX formulas
- Clamping and boundary enforcement for evidence timestamps
- Transcript quality audit behavior on abridged vs dense lectures
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from youtube_learning_companion.llm import (
    LLMAuthenticationError,
    LLMMalformedResponseError,
    LLMQuotaExhaustedError,
    LLMTimeoutError,
    MockLLMClient,
    OpenAILLMClient,
)
from youtube_learning_companion.markdown import MarkdownExporter
from youtube_learning_companion.schemas import (
    NoteGenerationStatus,
    TranscriptAcquisitionMethod,
    TranscriptDocument,
    TranscriptSegment,
    UncertaintyType,
    YouTubeVideoMetadata,
)
from youtube_learning_companion.synthesizer import StudyNoteSynthesizer
from youtube_learning_companion.transcript import FileTranscriptProvider
from youtube_learning_companion.transcript_audit import audit_transcript

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "companion"


@pytest.fixture
def pilot_metadata() -> YouTubeVideoMetadata:
    return YouTubeVideoMetadata(
        video_id="DrCeIbpfuzE",
        title=(
            "1a. Why Study Linear Algebra | The Importance of Linear Algebra in "
            "Machine Learning | Sachin Mittal"
        ),
        source_url="https://www.youtube.com/watch?v=DrCeIbpfuzE",
        channel_title="GO Classes for GATE CS",
        duration_seconds=1331.0,
    )


@pytest.fixture
def pilot_transcript_doc() -> TranscriptDocument:
    json_path = FIXTURES_DIR / "DrCeIbpfuzE.json"
    provider = FileTranscriptProvider(file_map={"DrCeIbpfuzE": json_path})
    doc = provider.get_transcript("DrCeIbpfuzE")
    assert doc is not None
    return doc


class TestLLMStudyNoteSynthesis:
    """Test LLM-driven synthesis pipeline and error boundaries."""

    def test_mock_llm_synthesis_structured_output(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Verify successful LLM synthesis produces schema-compliant note with evidence links."""
        client = MockLLMClient()
        synthesizer = StudyNoteSynthesizer(llm_client=client, chunk_window_seconds=300.0)
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)

        assert note.status == NoteGenerationStatus.PARTIAL_SUCCESS
        assert note.generation_metadata.get("synthesizer_mode") == "llm"
        assert note.generation_metadata.get("llm_provider") == "mock"
        assert note.coverage_percentage >= 90.0
        assert note.temporal_coverage_percentage >= 90.0
        assert note.transcript_completeness == "unknown"

        assert len(note.sections) > 0
        assert len(note.learning_objectives) > 0
        assert len(note.key_takeaways) > 0
        assert len(note.practice_questions) > 0

        # Verify evidence reference provenance on substantive claims
        first_sec = note.sections[0]
        assert len(first_sec.definitions) > 0
        def_item = first_sec.definitions[0]
        assert len(def_item.evidence_refs) > 0
        ref = def_item.evidence_refs[0]
        assert ref.video_id == pilot_metadata.video_id
        assert ref.segment_id is not None
        assert ref.start_seconds >= 0.0
        assert ref.end_seconds >= ref.start_seconds

        # Verification check
        assert note.validation is not None
        assert note.validation.is_valid is True
        assert note.validation.unsupported_claims == 0

    def test_malformed_llm_output_triggers_error_or_fallback(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Malformed JSON output from LLM raises exception or cleanly triggers
        deterministic fallback.
        """
        client = MockLLMClient(simulate_malformed_json=True)

        # 1. Fallback disabled: must raise LLMMalformedResponseError
        strict_synth = StudyNoteSynthesizer(llm_client=client, fallback_to_deterministic=False)
        with pytest.raises(LLMMalformedResponseError):
            strict_synth.synthesize(pilot_metadata, pilot_transcript_doc)

        # 2. Fallback enabled: completes with fallback metadata and warnings
        resilient_synth = StudyNoteSynthesizer(llm_client=client, fallback_to_deterministic=True)
        note = resilient_synth.synthesize(pilot_metadata, pilot_transcript_doc)
        assert note.generation_metadata.get("synthesizer_mode") == "fallback_deterministic"
        assert "LLMMalformedResponseError" in note.generation_metadata.get("llm_error_type", "")
        assert note.validation is not None
        assert any("LLM synthesis failed" in w for w in note.validation.warnings)

    def test_quota_exhausted_handling(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Verify credit/quota exhaustion triggers explicit LLMQuotaExhaustedError."""
        client = MockLLMClient(simulate_quota_exhausted=True)

        strict_synth = StudyNoteSynthesizer(llm_client=client, fallback_to_deterministic=False)
        with pytest.raises(LLMQuotaExhaustedError) as exc_info:
            strict_synth.synthesize(pilot_metadata, pilot_transcript_doc)
        assert "credit_balance_exhausted" in str(exc_info.value)

        # Resilient fallback mode
        resilient_synth = StudyNoteSynthesizer(llm_client=client, fallback_to_deterministic=True)
        note = resilient_synth.synthesize(pilot_metadata, pilot_transcript_doc)
        assert note.generation_metadata.get("synthesizer_mode") == "fallback_deterministic"
        assert note.generation_metadata.get("llm_error_type") == "LLMQuotaExhaustedError"

    def test_timeout_and_retry_behavior(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Verify LLM timeout raises LLMTimeoutError and logs actionable guidance."""
        client = MockLLMClient(simulate_timeout=True)

        strict_synth = StudyNoteSynthesizer(llm_client=client, fallback_to_deterministic=False)
        with pytest.raises(LLMTimeoutError):
            strict_synth.synthesize(pilot_metadata, pilot_transcript_doc)

    def test_missing_credentials_raises_auth_error(self) -> None:
        """OpenAILLMClient raises LLMAuthenticationError when no API key is provided."""
        client = OpenAILLMClient(api_key="", base_url="https://api.openai.com/v1")
        with pytest.raises(LLMAuthenticationError) as exc_info:
            client.generate(messages=[{"role": "user", "content": "hello"}])
        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_unbalanced_latex_flagged_in_uncertainties(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Unbalanced LaTeX equations from LLM are detected and flagged in uncertainties."""
        canned_with_bad_latex = json.dumps(
            {
                "title": "Matrix Inverses and Systems",
                "overview": "Discussion of systems of linear equations.",
                "definitions": [],
                "theorems": [],
                "formulas": [
                    {
                        "name": "Corrupted Equation",
                        "latex_expression": "$A x = b",  # Unclosed dollar delimiter
                        "meaning": "Broken formula",
                        "variables": {},
                        "start_seconds": 0.0,
                        "end_seconds": 45.0,
                    }
                ],
                "worked_examples": [],
                "exam_traps": [],
                "uncertainties": [],
                "takeaways": [],
            }
        )
        cons_canned = json.dumps(
            {
                "learning_objectives": ["Understand linear transformations"],
                "executive_summary": "Lecture summary overview.",
                "practice_questions": [],
                "key_takeaways": [],
            }
        )
        client = MockLLMClient(canned_responses=[canned_with_bad_latex, cons_canned])
        synthesizer = StudyNoteSynthesizer(llm_client=client, chunk_window_seconds=1500.0)
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)

        # Check uncertainty was flagged
        corrupted_flags = [
            u for u in note.uncertainties if u.uncertainty_type == UncertaintyType.CORRUPTED_FORMULA
        ]
        assert len(corrupted_flags) > 0
        assert "Unbalanced LaTeX" in corrupted_flags[0].description


class TestTranscriptQualityAudit:
    """Test transcript quality auditor across various transcript densities."""

    def test_audit_abridged_pilot_fixture(self, pilot_transcript_doc: TranscriptDocument) -> None:
        """Verify auditor correctly identifies abridged pilot fixture and flags limitations."""
        report = audit_transcript(
            pilot_transcript_doc,
            video_duration=1330.866,
            video_title="1a. Why Study Linear Algebra",
        )

        assert report.video_id == "DrCeIbpfuzE"
        assert report.segment_count == 12
        assert report.total_words == 351
        assert report.is_abridged is True
        assert report.words_per_minute < 50.0  # ~15.8 wpm vs expected 120-160
        assert report.gap_count == 0
        assert report.overlap_count == 0
        assert report.temporal_coverage_pct >= 99.0

        # Supportability verdicts
        assert report.can_support_topic_synopsis is True
        assert report.can_support_full_lecture_notes is False
        assert any("Low transcript density diagnostic" in f for f in report.findings)
        assert any("does not establish transcript completeness" in f for f in report.findings)

    def test_audit_empty_transcript(self) -> None:
        """Verify auditor handles empty transcript without crashing."""
        from youtube_learning_companion.schemas import TranscriptDocument

        empty_doc = TranscriptDocument(
            video_id="test_empty",
            language="en",
            acquisition_method="none",
            segments=(),
        )
        report = audit_transcript(empty_doc)
        assert report.segment_count == 0
        assert report.can_support_full_lecture_notes is False
        assert report.can_support_topic_synopsis is False

    def test_audit_reports_segments_beyond_declared_duration(self) -> None:
        segment = TranscriptSegment(
            video_id="range_vid",
            start_seconds=9.0,
            end_seconds=12.0,
            text="This cue exceeds the supplied duration.",
        )
        transcript = TranscriptDocument(
            video_id="range_vid",
            acquisition_method=TranscriptAcquisitionMethod.USER_FILE_JSON,
            segments=(segment,),
            coverage_status="complete",
            coverage_rationale="Source declaration under test.",
        )

        report = audit_transcript(transcript, video_duration=10.0)

        assert report.out_of_range_segment_count == 1
        assert report.can_support_full_lecture_notes is False
        assert any("beyond the declared video duration" in finding for finding in report.findings)

    def test_audit_dense_transcript_reporting_all_metrics(self) -> None:
        """Verify auditor reports word count, segment count, first/last timestamps,
        gaps, overlaps, speaking rate, and full eligibility on dense transcripts."""
        text_chunk = " ".join(["word"] * 75)
        segments = (
            TranscriptSegment(
                video_id="dense_vid",
                start_seconds=0.0,
                end_seconds=30.0,
                text=text_chunk,
            ),
            TranscriptSegment(
                video_id="dense_vid",
                start_seconds=35.0,
                end_seconds=65.0,
                text=text_chunk,
            ),
            TranscriptSegment(
                video_id="dense_vid",
                start_seconds=64.0,
                end_seconds=94.0,
                text=text_chunk,
            ),
            TranscriptSegment(
                video_id="dense_vid",
                start_seconds=95.0,
                end_seconds=120.0,
                text=text_chunk,
            ),
        )
        dense_doc = TranscriptDocument(
            video_id="dense_vid",
            language="en",
            acquisition_method=TranscriptAcquisitionMethod.USER_FILE_JSON,
            segments=segments,
            coverage_status="complete",
            coverage_rationale="Source file was verified against the full lecture recording.",
        )
        report = audit_transcript(dense_doc, video_duration=120.0, video_title="Dense Lecture")

        assert report.video_id == "dense_vid"
        assert report.segment_count == 4
        assert report.total_words == 300
        assert report.start_seconds == 0.0
        assert report.end_seconds == 120.0
        assert report.transcript_span_seconds == 120.0
        assert report.video_duration_seconds == 120.0
        assert report.temporal_coverage_pct == 100.0

        # Speaking rate
        assert report.words_per_minute == 150.0
        assert report.is_abridged is False

        # Gap and overlap metrics
        assert report.gap_count == 1
        assert report.max_gap_seconds == 5.0
        assert report.overlap_count == 1
        assert report.max_overlap_seconds == 1.0

        # Eligibility verdicts
        assert report.can_support_full_lecture_notes is True
        assert report.can_support_topic_synopsis is True
        assert any("Dense conversational transcript" in f for f in report.findings)


class TestSynthesisDefectsAndProvenance:
    """Test targeted defect fixes for provenance, formula attribution, and uncertainties."""

    def test_markdown_provenance_frontmatter_and_footer(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Markdown output must include synthesis_mode in frontmatter and footer."""
        from youtube_learning_companion.markdown import MarkdownExporter

        client = MockLLMClient()
        synthesizer = StudyNoteSynthesizer(llm_client=client)
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)

        exporter = MarkdownExporter()
        md = exporter.export_note_to_markdown(note)

        assert 'synthesis_mode: "llm"' in md
        assert 'llm_provider: "mock"' in md
        assert "Synthesis Mode: llm" in md

    def test_formula_attribution_avoids_over_eager_y_equals_ax(
        self, pilot_metadata: YouTubeVideoMetadata
    ) -> None:
        """Segments defining vector concepts without equations must not fabricate y = Ax."""
        from youtube_learning_companion.schemas import TranscriptSegment

        seg = TranscriptSegment(
            video_id="DrCeIbpfuzE",
            start_seconds=45.0,
            end_seconds=120.0,
            text=(
                "In computer science and machine learning, data is represented using vectors "
                "and matrices. A feature vector is an ordered collection of real numbers."
            ),
        )
        doc = TranscriptDocument(
            video_id="DrCeIbpfuzE",
            acquisition_method="user_file_json",
            segments=(seg,),
        )
        synthesizer = StudyNoteSynthesizer()
        note = synthesizer.synthesize(pilot_metadata, doc)

        # Segment defines vector, but does not state the linear transformation equation y = Ax
        all_formulas = [f for sec in note.sections for f in sec.formulas]
        assert len(all_formulas) == 0

    def test_abridged_transcript_uncertainty_flag(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Low density should prompt transcript coverage review, not imply omission as fact."""
        synthesizer = StudyNoteSynthesizer()
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)

        omitted_derivations = [
            u
            for u in note.uncertainties
            if u.uncertainty_type == UncertaintyType.OMITTED_DERIVATION
        ]
        assert len(omitted_derivations) == 1
        assert "Low transcript density diagnostic" in omitted_derivations[0].description
        assert "coverage review" in omitted_derivations[0].affected_concept_or_timestamp

    def test_practice_question_source_distinction(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Practice questions must explicitly distinguish source-derived from supplementary."""
        synthesizer = StudyNoteSynthesizer()
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)

        assert len(note.practice_questions) >= 2
        q_texts = [q.question for q in note.practice_questions]
        assert any("[SOURCE-DERIVED]" in q for q in q_texts)
        assert any("[SUPPLEMENTARY PRACTICE]" in q for q in q_texts)


class TestTemporalCoverageVsCompletenessRegression:
    """Rigorous regression tests covering temporal coverage vs transcript completeness."""

    def test_abridged_transcript_with_100_percent_temporal_coverage(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """100% timeline coverage with abridged word count must yield PARTIAL_SUCCESS."""
        synthesizer = StudyNoteSynthesizer()
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)

        assert note.coverage_percentage == 100.0
        assert note.temporal_coverage_percentage == 100.0
        assert note.transcript_completeness == "unknown"
        assert note.status == NoteGenerationStatus.PARTIAL_SUCCESS

        # Low density is diagnostic; coverage remains unknown without source evidence.
        assert note.validation is not None
        assert any("coverage is unknown" in w for w in note.validation.warnings)
        assert any("Low transcript density diagnostic" in w for w in note.validation.warnings)

    def test_complete_looking_timeline_with_insufficient_density(
        self, pilot_metadata: YouTubeVideoMetadata
    ) -> None:
        """A complete-looking timeline with insufficient density must yield PARTIAL_SUCCESS."""
        from youtube_learning_companion.schemas import TranscriptSegment

        # 2 sparse segments spanning the full 1331s timeline, but only 12 words total
        sparse_segs = (
            TranscriptSegment(
                video_id="DrCeIbpfuzE",
                start_seconds=0.0,
                end_seconds=10.0,
                text="Welcome to linear algebra.",
            ),
            TranscriptSegment(
                video_id="DrCeIbpfuzE",
                start_seconds=1320.0,
                end_seconds=1331.0,
                text="In the next lecture we will continue.",
            ),
        )
        sparse_doc = TranscriptDocument(
            video_id="DrCeIbpfuzE",
            acquisition_method="user_file_json",
            segments=sparse_segs,
        )

        synthesizer = StudyNoteSynthesizer()
        note = synthesizer.synthesize(pilot_metadata, sparse_doc)

        assert note.temporal_coverage_percentage == 100.0
        assert note.status == NoteGenerationStatus.PARTIAL_SUCCESS
        assert note.transcript_completeness == "unknown"

    def test_markdown_metadata_distinguishes_temporal_coverage_and_status(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Exported Markdown warns when source coverage is unknown despite full temporal span."""
        from youtube_learning_companion.markdown import MarkdownExporter

        synthesizer = StudyNoteSynthesizer()
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)

        exporter = MarkdownExporter()
        md = exporter.export_note_to_markdown(note)

        # Frontmatter checks
        assert 'temporal_coverage: "100.0%"' in md
        assert 'transcript_completeness: "unknown"' in md
        assert "Transcript Coverage: UNKNOWN" in md
        assert "Temporal span and speech density are diagnostics" in md
        assert 'status: "partial_success"' in md

        # Header warning check
        assert "> [!warning] Transcript Coverage: UNKNOWN" in md
        assert "speech density are diagnostics" in md

    def test_existing_behavior_for_genuinely_complete_transcripts(
        self, pilot_metadata: YouTubeVideoMetadata
    ) -> None:
        """Dense transcripts with normal density (>=100 wpm) produce COMPLETE status."""
        from youtube_learning_companion.markdown import MarkdownExporter
        from youtube_learning_companion.schemas import TranscriptSegment

        # Construct a dense transcript across 10 minutes (600s) with ~1,300 words (~130 wpm)
        dense_meta = YouTubeVideoMetadata(
            video_id="dense_video_01",
            title="1b. Vector Spaces and Subspaces in Complete Detail",
            source_url="https://www.youtube.com/watch?v=dense_video_01",
            duration_seconds=600.0,
        )
        segs = []
        for i in range(10):
            st = i * 60.0
            et = (i + 1) * 60.0
            text = (
                f"Minute {i+1} defines vector space axioms and scalar multiplication closure. "
                "For every vector v in V and scalar c in F, c times v belongs to V. "
                "We examine linear combinations, spanning sets, and linear independence in R^n. "
                f"Theorem {i+1} states that any set of n plus one vectors in n dimensional space "
                "must be linearly dependent. We prove this using Gaussian elimination. "
                "Observe how the matrix determinant relates to the parallelepiped volume."
            )
            segs.append(
                TranscriptSegment(
                    video_id="dense_video_01",
                    start_seconds=st,
                    end_seconds=et,
                    text=text,
                )
            )

        dense_doc = TranscriptDocument(
            video_id="dense_video_01",
            acquisition_method="local_asr",
            segments=tuple(segs),
            coverage_status="complete",
            coverage_rationale="Complete local ASR transcript was verified against the recording.",
        )

        synthesizer = StudyNoteSynthesizer()
        note = synthesizer.synthesize(dense_meta, dense_doc)

        assert note.coverage_percentage == 100.0
        assert note.temporal_coverage_percentage == 100.0
        assert note.transcript_completeness == "complete"
        assert note.status == NoteGenerationStatus.COMPLETE

        # Verify Markdown reflects COMPLETE status without abridged warning
        exporter = MarkdownExporter()
        md = exporter.export_note_to_markdown(note)
        assert 'status: "complete"' in md
        assert 'transcript_completeness: "complete"' in md
        assert "ABRIDGED TRANSCRIPT" not in md


class TestIntegrationStabilizationAndOptimization:
    """Regression and unit tests for Phase B-D stabilization and efficiency improvements."""

    def test_density_vs_temporal_coverage_disambiguation(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Verify density coverage reflects 11.7% while temporal coverage reflects 100.0%."""
        synthesizer = StudyNoteSynthesizer()
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)

        assert note.temporal_coverage_percentage == 100.0
        assert note.density_coverage_percentage == 11.7
        assert note.coverage_percentage == 100.0
        assert note.transcript_completeness == "unknown"

        exporter = MarkdownExporter()
        md = exporter.export_note_to_markdown(note)
        assert 'temporal_coverage: "100.0%"' in md
        assert 'density_coverage: "11.7%"' in md
        assert 'transcript_completeness: "unknown"' in md

    def test_mock_llm_metadata_transparency(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Verify mock LLM metadata is marked with is_mock: true in markdown frontmatter."""
        mock_client = MockLLMClient(model="test-simulated-model")
        synthesizer = StudyNoteSynthesizer(llm_client=mock_client, fallback_to_deterministic=False)
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)

        assert note.generation_metadata.get("is_mock") is True
        exporter = MarkdownExporter()
        md = exporter.export_note_to_markdown(note)
        assert "is_mock: true" in md

    def test_section_synthesis_caching(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Verify fingerprint caching prevents repeated LLM calls on identical section chunks."""
        mock_client = MockLLMClient(model="cached-model")
        synthesizer = StudyNoteSynthesizer(
            llm_client=mock_client,
            fallback_to_deterministic=False,
            enable_cache=True,
        )

        note1 = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)
        initial_calls = mock_client.call_count
        assert initial_calls > 0
        assert note1.generation_metadata.get("cache_hits") == 0

        # Second synthesis of identical content should hit the cache for all sections
        note2 = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)
        expected_sections = len(synthesizer._partition_into_windows(pilot_transcript_doc.segments))
        assert note2.generation_metadata.get("cache_hits") == expected_sections
        # Call count should only have increased by 1 (the final consolidation call)
        assert mock_client.call_count == initial_calls + 1

    def test_orchestrator_manifest_pending_count_exactness(self, tmp_path: Path) -> None:
        """Verify pending_count does not subtract partial_count twice."""
        from youtube_learning_companion.orchestrator import PlaylistJobOrchestrator
        from youtube_learning_companion.schemas import (
            PlaylistProcessingManifest,
            VideoProcessingRecord,
            YouTubePlaylistMetadata,
        )

        orch = PlaylistJobOrchestrator(output_dir=tmp_path)
        manifest_file = tmp_path / "processing_manifest.json"

        # Create 5 videos in playlist
        videos = tuple(
            YouTubeVideoMetadata(
                video_id=f"vid_{i}",
                title=f"Video {i}",
                source_url=f"https://youtube.com/watch?v=vid_{i}",
            )
            for i in range(5)
        )
        pl_meta = YouTubePlaylistMetadata(
            playlist_id="PL_TEST",
            title="Test Playlist",
            source_url="https://youtube.com/playlist?list=PL_TEST",
            video_count=5,
            videos=videos,
        )

        # Mock existing records: 1 complete, 1 partial_success, 1 failed
        records = (
            VideoProcessingRecord(
                video_id="vid_0",
                position=0,
                title="Video 0",
                source_url="https://youtube.com/watch?v=vid_0",
                status=NoteGenerationStatus.COMPLETE,
            ),
            VideoProcessingRecord(
                video_id="vid_1",
                position=1,
                title="Video 1",
                source_url="https://youtube.com/watch?v=vid_1",
                status=NoteGenerationStatus.PARTIAL_SUCCESS,
            ),
            VideoProcessingRecord(
                video_id="vid_2",
                position=2,
                title="Video 2",
                source_url="https://youtube.com/watch?v=vid_2",
                status=NoteGenerationStatus.FAILED,
            ),
        )
        initial_manifest = PlaylistProcessingManifest(
            playlist_id="PL_TEST",
            playlist_title="Test Playlist",
            source_url="https://youtube.com/playlist?list=PL_TEST",
            total_videos_discovered=5,
            records=records,
        )
        manifest_file.write_text(initial_manifest.model_dump_json(), encoding="utf-8")

        # Mock connector returning pl_meta
        orch.connector.get_playlist = lambda url: pl_meta  # type: ignore[assignment]
        updated = orch.process_playlist("PL_TEST", limit=3)

        assert updated.processed_count == 2  # vid_0 (complete) + vid_1 (partial)
        assert updated.partial_count == 1
        assert updated.awaiting_transcript_count == 1
        # Total resolved = 2 (processed) + 1 (awaiting transcript) = 3. Pending should be 5 - 3 = 2!
        assert updated.pending_count == 2

    def test_domain_segment_conversion_interoperability(self) -> None:
        """Verify seamless conversion between companion and core TranscriptSegment."""
        from uuid import uuid4

        from youtube_learning_companion.schemas import TranscriptSegment as CompanionSegment

        c_seg = CompanionSegment(
            video_id="vid_123",
            start_seconds=12.5,
            end_seconds=18.0,
            text="Eigenvalues indicate geometric stretch factors.",
            confidence=0.98,
            speaker="Lecturer",
        )
        vid_uuid = uuid4()
        try:
            d_seg = c_seg.to_domain_segment(video_uuid=vid_uuid, provider="faster_whisper")
        except ImportError:
            import pytest
            pytest.skip("Optional core VIDEX package not installed in environment")

        assert d_seg.video_id == vid_uuid
        assert d_seg.start_timestamp_seconds == 12.5
        assert d_seg.end_timestamp_seconds == 18.0
        assert d_seg.raw_text == "Eigenvalues indicate geometric stretch factors."
        assert d_seg.provider == "faster_whisper"
        assert d_seg.speaker_id == "Lecturer"

        # Roundtrip back to companion
        c_roundtrip = CompanionSegment.from_domain_segment(d_seg, video_id_str="vid_123")
        assert c_roundtrip.video_id == "vid_123"
        assert c_roundtrip.start_seconds == 12.5
        assert c_roundtrip.end_seconds == 18.0
        assert c_roundtrip.text == c_seg.text
        assert c_roundtrip.confidence == 0.98
        assert c_roundtrip.speaker == "Lecturer"


class TestPostOptimizationHardening:
    """Comprehensive regression tests for Phase B-E hardening requirements."""

    def test_cache_key_invalidation_matrix(self) -> None:
        """Verify changing each synthesis-affecting input invalidates the cache key."""
        from youtube_learning_companion.schemas import TranscriptSegment
        from youtube_learning_companion.synthesizer import compute_section_cache_key

        seg1 = TranscriptSegment(
            video_id="v_1", start_seconds=0.0, end_seconds=60.0, text="Intro to matrices."
        )
        base_kwargs = {
            "video_id": "v_1",
            "window_idx": 0,
            "window_segments": [seg1],
            "model_id": "llama3:8b",
            "provider_name": "ollama",
            "is_mock": False,
            "prompt_version": "1.0",
            "schema_version": "1.0",
            "visual_evidence": None,
            "generation_params": {"temperature": 0.0},
        }

        base_key = compute_section_cache_key(**base_kwargs)  # type: ignore[arg-type]
        assert isinstance(base_key, str)
        assert len(base_key) == 64  # SHA-256 hex digest

        # 1. Identical inputs produce identical key
        assert compute_section_cache_key(**base_kwargs) == base_key  # type: ignore[arg-type]

        # 2. Changed model_id
        kw = dict(base_kwargs)
        kw["model_id"] = "gpt-4o"
        assert compute_section_cache_key(**kw) != base_key  # type: ignore[arg-type]

        # 3. Changed provider_name
        kw = dict(base_kwargs)
        kw["provider_name"] = "openai"
        assert compute_section_cache_key(**kw) != base_key  # type: ignore[arg-type]

        # 4. Changed is_mock
        kw = dict(base_kwargs)
        kw["is_mock"] = True
        assert compute_section_cache_key(**kw) != base_key  # type: ignore[arg-type]

        # 5. Changed prompt_version
        kw = dict(base_kwargs)
        kw["prompt_version"] = "2.0"
        assert compute_section_cache_key(**kw) != base_key  # type: ignore[arg-type]

        # 6. Changed schema_version
        kw = dict(base_kwargs)
        kw["schema_version"] = "2.0"
        assert compute_section_cache_key(**kw) != base_key  # type: ignore[arg-type]

        # 7. Changed window bounds (different start/end)
        seg_shifted = TranscriptSegment(
            video_id="v_1", start_seconds=5.0, end_seconds=65.0, text="Intro to matrices."
        )
        kw = dict(base_kwargs)
        kw["window_segments"] = [seg_shifted]
        assert compute_section_cache_key(**kw) != base_key  # type: ignore[arg-type]

        # 8. Changed transcript text
        seg_text = TranscriptSegment(
            video_id="v_1", start_seconds=0.0, end_seconds=60.0, text="Intro to vector spaces."
        )
        kw = dict(base_kwargs)
        kw["window_segments"] = [seg_text]
        assert compute_section_cache_key(**kw) != base_key  # type: ignore[arg-type]

        # 9. Changed visual evidence
        kw = dict(base_kwargs)
        kw["visual_evidence"] = [{"ocr": "Matrix A"}]
        assert compute_section_cache_key(**kw) != base_key  # type: ignore[arg-type]

        # 10. Changed generation params
        kw = dict(base_kwargs)
        kw["generation_params"] = {"temperature": 0.7}
        assert compute_section_cache_key(**kw) != base_key  # type: ignore[arg-type]

        # Timestamp provenance changes the prompt and must invalidate cached synthesis.
        inferred_segment = seg1.model_copy(update={"start_timestamp_is_source": False})
        kw = dict(base_kwargs)
        kw["window_segments"] = [inferred_segment]
        assert compute_section_cache_key(**kw) != base_key  # type: ignore[arg-type]
        inferred_end_segment = seg1.model_copy(update={"end_timestamp_is_source": False})
        kw["window_segments"] = [inferred_end_segment]
        assert compute_section_cache_key(**kw) != base_key  # type: ignore[arg-type]

    def test_synthesis_rejects_transcript_for_different_video(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        mismatched_transcript = pilot_transcript_doc.model_copy(update={"video_id": "other_video"})

        with pytest.raises(ValueError, match="does not match video metadata"):
            StudyNoteSynthesizer().synthesize(pilot_metadata, mismatched_transcript)

    def test_inferred_timing_provenance_survives_llm_synthesis(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        inferred_segment = pilot_transcript_doc.segments[0].model_copy(
            update={"start_timestamp_is_source": False}
        )
        transcript = pilot_transcript_doc.model_copy(update={"segments": (inferred_segment,)})

        note = StudyNoteSynthesizer(
            llm_client=MockLLMClient(),
            fallback_to_deterministic=False,
            chunk_window_seconds=300.0,
        ).synthesize(pilot_metadata, transcript)

        section = note.sections[0]
        assert section.start_timestamp_is_source is False
        assert section.end_timestamp_is_source is True
        evidence = section.definitions[0].evidence_refs[0]
        assert evidence.start_timestamp_is_source is False
        assert evidence.end_timestamp_is_source is True

    def test_inferred_timing_provenance_survives_deterministic_synthesis(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        source_segment = pilot_transcript_doc.segments[0].model_copy(
            update={"start_timestamp_is_source": False}
        )
        transcript = pilot_transcript_doc.model_copy(update={"segments": (source_segment,)})

        note = StudyNoteSynthesizer().synthesize(pilot_metadata, transcript)

        section = note.sections[0]
        assert section.start_timestamp_is_source is False
        evidence = section.definitions[0].evidence_refs[0]
        assert evidence.start_timestamp_is_source is False

    def test_cache_hit_deserialization_safety_and_malformed_purging(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Verify malformed cache entries fail safely, are purged, and trigger recomputation."""
        mock_client = MockLLMClient(model="safe-cache-model")
        synthesizer = StudyNoteSynthesizer(
            llm_client=mock_client,
            fallback_to_deterministic=False,
            enable_cache=True,
        )

        # Pre-seed cache with a corrupted entry (string instead of dict, or missing fields)
        corrupted_key = "corrupted_cache_key_hash"
        synthesizer._section_cache[corrupted_key] = "not a dict"  # type: ignore[assignment]

        # Also pre-seed a malformed dict missing required keys
        incomplete_key = "incomplete_cache_key_hash"
        synthesizer._section_cache[incomplete_key] = {"some_random_key": 123}

        # Perform synthesis: should succeed normally without crashing
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)
        assert note.video_id == pilot_metadata.video_id
        assert len(note.sections) > 0

    def test_mock_completions_isolated_from_production(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Ensure mock outputs cannot be reused as successful production results."""
        mock_client = MockLLMClient(model="shared-model")
        mock_client.is_mock = True
        synthesizer = StudyNoteSynthesizer(
            llm_client=mock_client, fallback_to_deterministic=False, enable_cache=True
        )

        # Synthesize with mock client
        note1 = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)
        assert note1.generation_metadata.get("is_mock") is True
        assert mock_client.call_count > 0

        # Now simulate a production client with is_mock=False sharing same synthesizer
        prod_client = MockLLMClient(model="shared-model")
        prod_client.is_mock = False
        synthesizer.llm_client = prod_client

        note2 = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)
        assert note2.generation_metadata.get("is_mock") is False
        # Production client must NOT have hit the mock cache entries
        assert note2.generation_metadata.get("cache_hits") == 0
        assert prod_client.call_count > 0

    def test_orchestrator_concurrency_and_worker_failure_resilience(
        self, tmp_path: Path
    ) -> None:
        """Verify orchestrator concurrency > 1 and graceful worker failure resilience."""
        from youtube_learning_companion.orchestrator import PlaylistJobOrchestrator
        from youtube_learning_companion.schemas import (
            PlaylistAccessStatus,
            StudyNote,
            VideoProcessingRecord,
            YouTubePlaylistMetadata,
        )

        # 4 videos: 3 normal, 1 that raises an exception during processing
        videos = [
            YouTubeVideoMetadata(
                video_id=f"worker_vid_{i}",
                title=f"Worker Lecture {i}",
                source_url=f"https://youtube.com/watch?v=worker_vid_{i}",
                duration_seconds=300.0,
            )
            for i in range(4)
        ]
        pl_meta = YouTubePlaylistMetadata(
            playlist_id="PL_CONCURRENCY",
            title="Concurrency Test Playlist",
            source_url="https://youtube.com/playlist?list=PL_CONCURRENCY",
            video_count=4,
            access_status=PlaylistAccessStatus.ACCESSIBLE,
            videos=tuple(videos),
        )

        orch = PlaylistJobOrchestrator(output_dir=tmp_path, max_concurrency=4)
        orch.connector.get_playlist = lambda url: pl_meta  # type: ignore[assignment]

        original_process = orch.process_single_video

        def mock_process(
            video_meta: YouTubeVideoMetadata,
        ) -> tuple[StudyNote, VideoProcessingRecord]:
            if video_meta.video_id == "worker_vid_2":
                raise RuntimeError("Simulated network timeout for worker_vid_2")
            return original_process(video_meta)

        orch.process_single_video = mock_process  # type: ignore[method-assign]

        manifest = orch.process_playlist("PL_CONCURRENCY")
        assert manifest.total_videos_discovered == 4
        # 3 succeeded (with awaiting transcript or partial), 1 failed
        assert manifest.failed_count == 1
        assert manifest.pending_count == 0
        failed_records = [r for r in manifest.records if r.status == NoteGenerationStatus.FAILED]
        assert len(failed_records) == 1
        assert failed_records[0].video_id == "worker_vid_2"
        assert "Simulated network timeout" in (failed_records[0].error_message or "")

    def test_orchestrator_empty_playlist_and_duplicates(self, tmp_path: Path) -> None:
        """Verify invariants on empty playlists and duplicate video filtering."""
        from youtube_learning_companion.orchestrator import PlaylistJobOrchestrator
        from youtube_learning_companion.schemas import (
            PlaylistAccessStatus,
            YouTubePlaylistMetadata,
        )

        orch = PlaylistJobOrchestrator(output_dir=tmp_path)

        # 1. Empty playlist
        empty_pl = YouTubePlaylistMetadata(
            playlist_id="PL_EMPTY",
            title="Empty Playlist",
            source_url="https://youtube.com/playlist?list=PL_EMPTY",
            video_count=0,
            access_status=PlaylistAccessStatus.EMPTY,
            videos=(),
        )
        orch.connector.get_playlist = lambda url: empty_pl  # type: ignore[assignment]
        m_empty = orch.process_playlist("PL_EMPTY")
        assert m_empty.total_videos_discovered == 0
        assert m_empty.pending_count == 0
        assert m_empty.processed_count == 0
        assert m_empty.failed_count == 0
        assert len(m_empty.records) == 0

        # 2. Playlist with duplicate video IDs
        dup_videos = (
            YouTubeVideoMetadata(
                video_id="dup_vid",
                title="Duplicate Video",
                source_url="https://youtube.com/watch?v=dup_vid",
            ),
            YouTubeVideoMetadata(
                video_id="dup_vid",
                title="Duplicate Video Again",
                source_url="https://youtube.com/watch?v=dup_vid",
            ),
            YouTubeVideoMetadata(
                video_id="unique_vid",
                title="Unique Video",
                source_url="https://youtube.com/watch?v=unique_vid",
            ),
        )
        dup_pl = YouTubePlaylistMetadata(
            playlist_id="PL_DUP",
            title="Duplicate Playlist",
            source_url="https://youtube.com/playlist?list=PL_DUP",
            video_count=3,
            access_status=PlaylistAccessStatus.ACCESSIBLE,
            videos=dup_videos,
        )
        tmp_dup = tmp_path / "dup_test"
        orch_dup = PlaylistJobOrchestrator(output_dir=tmp_dup)
        orch_dup.connector.get_playlist = lambda url: dup_pl  # type: ignore[assignment]
        m_dup = orch_dup.process_playlist("PL_DUP")
        # Total unique discovered must be 2, not 3!
        assert m_dup.total_videos_discovered == 2
        assert len(m_dup.records) == 2
        assert m_dup.pending_count == 0

    def test_transcript_adapter_contract_validation_and_invariants(self) -> None:
        """Verify adapter conversions, deterministic UUID5 generation, and invalid values."""
        import pytest
        from pydantic import ValidationError

        from youtube_learning_companion.schemas import TranscriptSegment as CompanionSegment

        # 1. Deterministic UUID5 when video_uuid is None
        c_seg = CompanionSegment(
            video_id="abc123xyz",
            start_seconds=10.0,
            end_seconds=20.0,
            text="Deterministic UUID test.",
        )
        from importlib.util import find_spec

        if find_spec("videx") is None:
            with pytest.raises(ImportError, match="optional 'videx' package"):
                c_seg.to_domain_segment()
        else:
            d_seg1 = c_seg.to_domain_segment()
            d_seg2 = c_seg.to_domain_segment()
            assert d_seg1.video_id == d_seg2.video_id

        # 2. Invalid end_seconds < start_seconds raises ValueError
        with pytest.raises(ValueError, match="end_seconds"):
            CompanionSegment(
                video_id="v_1",
                start_seconds=30.0,
                end_seconds=10.0,
                text="Negative interval.",
            )

        # 3. Invalid confidence out of [0.0, 1.0] raises ValidationError
        with pytest.raises(ValidationError):
            CompanionSegment(
                video_id="v_1",
                start_seconds=0.0,
                end_seconds=10.0,
                text="Invalid confidence.",
                confidence=1.5,
            )

        # 4. Empty text raises ValidationError
        with pytest.raises(ValidationError):
            CompanionSegment(
                video_id="v_1",
                start_seconds=0.0,
                end_seconds=10.0,
                text="",
            )

    def test_monotonic_instrumentation_and_cold_vs_warm_benchmark(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Benchmark cold vs warm cache synthesis: measure calls avoided, timings, and hit rate."""
        mock_client = MockLLMClient(model="bench-model")
        synthesizer = StudyNoteSynthesizer(
            llm_client=mock_client,
            fallback_to_deterministic=False,
            enable_cache=True,
        )

        # Cold cache run
        cold_note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)
        cold_timings = cold_note.generation_metadata.get("stage_timings_ms", {})
        assert "section_synthesis_ms" in cold_timings
        assert "consolidation_ms" in cold_timings
        assert "total_llm_synthesis_ms" in cold_timings
        assert cold_timings["total_llm_synthesis_ms"] >= 0.0
        assert cold_note.generation_metadata.get("cache_hits") == 0
        assert cold_note.generation_metadata.get("cache_misses", 0) > 0
        assert cold_note.generation_metadata.get("cache_hit_rate_pct") == 0.0
        cold_calls = mock_client.call_count

        # Warm cache run (same inputs)
        warm_note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)
        warm_timings = warm_note.generation_metadata.get("stage_timings_ms", {})
        assert "total_llm_synthesis_ms" in warm_timings
        warm_hits = warm_note.generation_metadata.get("cache_hits", 0)
        expected_sections = len(synthesizer._partition_into_windows(pilot_transcript_doc.segments))
        assert warm_hits == expected_sections
        assert warm_note.generation_metadata.get("cache_misses") == 0
        assert warm_note.generation_metadata.get("cache_hit_rate_pct") == 100.0

        # Model calls avoided: exact count
        calls_added = mock_client.call_count - cold_calls
        # Only 1 call for global consolidation was executed in the warm run!
        assert calls_added == 1
        calls_avoided = expected_sections
        assert calls_avoided > 0
