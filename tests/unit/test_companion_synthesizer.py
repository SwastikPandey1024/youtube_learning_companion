"""Unit tests for GATE Linear Algebra study note synthesizer."""

from pathlib import Path

from youtube_learning_companion.schemas import (
    NoteGenerationStatus,
    TranscriptAcquisitionMethod,
    TranscriptDocument,
    TranscriptSegment,
    UncertaintyType,
    YouTubeVideoMetadata,
)
from youtube_learning_companion.synthesizer import StudyNoteSynthesizer, validate_latex_balance
from youtube_learning_companion.transcript import FileTranscriptProvider

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "companion"


class TestLaTeXValidation:
    """Test LaTeX formula balance validator."""

    def test_balanced_latex_expressions(self):
        assert validate_latex_balance("$A x = \\lambda x$")
        assert validate_latex_balance("$\\sum_{i=1}^n x_i^2$")
        assert validate_latex_balance("$\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$")
        assert validate_latex_balance("$\\|x\\|_2 = \\sqrt{\\langle x, x \\rangle}$")

    def test_unbalanced_latex_expressions(self):
        assert not validate_latex_balance("$A x = \\lambda x")  # missing closing $
        assert not validate_latex_balance("$\\sum_{i=1^n x_i$")  # missing }
        assert not validate_latex_balance(
            "$\\begin{pmatrix} a & b \\end{matrix}$"
        )  # mismatched env


class TestStudyNoteSynthesizer:
    """Test note synthesis, zero-hallucination policy, and validation."""

    def test_missing_transcript_zero_hallucination(self):
        """When transcript is None, synthesizer must NOT invent content."""
        meta = YouTubeVideoMetadata(
            video_id="DrCeIbpfuzE",
            title="1a. Why Study Linear Algebra",
            source_url="https://www.youtube.com/watch?v=DrCeIbpfuzE",
            duration_seconds=1331.0,
        )
        synthesizer = StudyNoteSynthesizer()
        note = synthesizer.synthesize(meta, transcript_doc=None)

        assert note.status == NoteGenerationStatus.AWAITING_TRANSCRIPT
        assert note.coverage_percentage == 0.0
        assert len(note.sections) == 0
        assert len(note.key_takeaways) == 0
        assert len(note.uncertainties) == 1
        assert note.uncertainties[0].uncertainty_type == UncertaintyType.TRANSCRIPT_UNAVAILABLE
        assert "awaiting lecture transcript" in note.executive_summary
        assert note.validation is not None
        assert note.validation.is_valid
        assert note.validation.grounded_claims == 0

    def test_grounded_note_synthesis(self):
        """Verify full note generation from benchmark lecture transcript."""
        meta = YouTubeVideoMetadata(
            video_id="DrCeIbpfuzE",
            title=(
                "1a. Why Study Linear Algebra | The Importance of Linear Algebra in "
                "Machine Learning | Sachin Mittal"
            ),
            source_url="https://www.youtube.com/watch?v=DrCeIbpfuzE",
            channel_title="GO Classes for GATE CS",
            duration_seconds=1331.0,
        )
        json_file = FIXTURES_DIR / "DrCeIbpfuzE.json"
        provider = FileTranscriptProvider(file_map={"DrCeIbpfuzE": json_file})
        doc = provider.get_transcript("DrCeIbpfuzE")
        assert doc is not None

        synthesizer = StudyNoteSynthesizer(chunk_window_seconds=300.0)
        note = synthesizer.synthesize(meta, transcript_doc=doc)

        assert note.video_id == "DrCeIbpfuzE"
        assert note.status == NoteGenerationStatus.PARTIAL_SUCCESS
        assert note.coverage_percentage == 100.0
        assert note.temporal_coverage_percentage == 100.0
        assert note.transcript_completeness == "unknown"
        assert len(note.learning_objectives) >= 3
        assert len(note.sections) >= 3

        # Check section items
        found_def = False
        found_formula = False
        found_trap = False
        found_example = False

        for sec in note.sections:
            if sec.definitions:
                found_def = True
                for d in sec.definitions:
                    assert len(d.evidence_refs) >= 1
                    ref = d.evidence_refs[0]
                    assert ref.video_id == "DrCeIbpfuzE"
                    assert ref.start_seconds >= 0.0
                    assert ref.end_seconds <= 1336.0
                    assert "https://youtube.com/watch?v=DrCeIbpfuzE&t=" in ref.clickable_url
            if sec.formulas:
                found_formula = True
                for f in sec.formulas:
                    assert len(f.evidence_refs) >= 1
                    assert validate_latex_balance(f.latex_expression)
            if sec.exam_traps:
                found_trap = True
            if sec.worked_examples:
                found_example = True

        assert found_def
        assert found_formula
        assert found_trap
        assert found_example

        # Check takeaways and practice questions
        assert len(note.key_takeaways) >= 2
        assert len(note.practice_questions) >= 1

        # Check validation audit
        assert note.validation is not None
        assert note.validation.is_valid
        assert note.validation.unsupported_claims == 0
        assert note.validation.grounding_ratio == 1.0
        assert note.validation.formulas_valid_latex
        assert note.validation.timestamps_within_duration

    def test_epistemic_uncertainty_flagging(self):
        """Verify that noisy/corrupted transcript tokens generate uncertainty flags."""
        meta = YouTubeVideoMetadata(
            video_id="noisy_vid",
            title="Lecture with Noise",
            source_url="https://youtube.com/watch?v=noisy_vid",
            duration_seconds=100.0,
        )
        segments = [
            TranscriptSegment(
                video_id="noisy_vid",
                start_seconds=0.0,
                end_seconds=30.0,
                text="The matrix transformation is [inaudible] when vectors are orthogonal.",
                confidence=0.35,
            ),
            TranscriptSegment(
                video_id="noisy_vid",
                start_seconds=30.0,
                end_seconds=60.0,
                text="We apply the formula y = A x to solve the system.",
                confidence=0.95,
            ),
        ]
        doc = TranscriptDocument(
            video_id="noisy_vid",
            acquisition_method=TranscriptAcquisitionMethod.USER_FILE_JSON,
            segments=tuple(segments),
        )

        synthesizer = StudyNoteSynthesizer()
        note = synthesizer.synthesize(meta, doc)

        assert len(note.uncertainties) >= 1
        unc_types = [u.uncertainty_type for u in note.uncertainties]
        assert UncertaintyType.AMBIGUOUS_SPEECH in unc_types

    def test_validate_note_catches_invalid_formula_and_trap_timestamps(self):
        """validate_note must catch out-of-bounds timestamps in formulas and traps."""
        from youtube_learning_companion.schemas import (
            DefinitionItem,
            EvidenceReference,
            FormulaItem,
            StudyNote,
            StudyNoteSection,
        )

        valid_ref = EvidenceReference(
            video_id="noisy_vid",
            start_seconds=10.0,
            end_seconds=20.0,
            snippet="Valid reference",
        )
        invalid_ref = EvidenceReference(
            video_id="noisy_vid",
            start_seconds=500.0,
            end_seconds=600.0,  # exceeds duration 100.0s
            snippet="Out-of-bounds formula reference",
        )

        sec = StudyNoteSection(
            section_id="sec_1",
            title="Matrix Algebra",
            start_seconds=0.0,
            end_seconds=100.0,
            overview="Overview",
            definitions=(
                DefinitionItem(
                    term="Matrix",
                    definition="Rectangular array",
                    evidence_refs=(valid_ref,),
                ),
            ),
            formulas=(
                FormulaItem(
                    name="Multiplication",
                    latex_expression="$C = A B$",
                    variables={"C": "Product", "A": "Matrix A", "B": "Matrix B"},
                    evidence_refs=(invalid_ref,),
                ),
            ),
        )

        note = StudyNote(
            note_id="test_note_invalid_ts",
            video_id="noisy_vid",
            title="Invalid Timestamp Note",
            video_title="Noisy Video",
            source_url="https://youtube.com/watch?v=noisy_vid",
            duration_seconds=100.0,
            coverage_percentage=100.0,
            executive_summary="Executive summary",
            sections=(sec,),
        )

        synthesizer = StudyNoteSynthesizer()
        val = synthesizer.validate_note(note)

        # Before fix: timestamps_within_duration is True (formula was skipped)
        # After fix: timestamps_within_duration is False, is_valid is False
        assert not val.timestamps_within_duration
        assert not val.is_valid
        assert any("exceeds duration" in w for w in val.warnings)
