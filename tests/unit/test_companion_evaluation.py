"""Unit tests for companion evaluation metrics and quality gates."""

from pathlib import Path

from youtube_learning_companion.evaluation import evaluate_study_note
from youtube_learning_companion.schemas import (
    YouTubeVideoMetadata,
)
from youtube_learning_companion.synthesizer import StudyNoteSynthesizer
from youtube_learning_companion.transcript import FileTranscriptProvider

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "companion"


class TestCompanionEvaluation:
    """Test evaluation metrics and quality gates."""

    def test_evaluate_grounded_study_note(self):
        meta = YouTubeVideoMetadata(
            video_id="DrCeIbpfuzE",
            title=(
                "1a. Why Study Linear Algebra | The Importance of Linear Algebra in "
                "Machine Learning | Sachin Mittal"
            ),
            source_url="https://www.youtube.com/watch?v=DrCeIbpfuzE",
            duration_seconds=1331.0,
        )
        json_file = FIXTURES_DIR / "DrCeIbpfuzE.json"
        doc = FileTranscriptProvider(file_map={"DrCeIbpfuzE": json_file}).get_transcript(
            "DrCeIbpfuzE"
        )
        note = StudyNoteSynthesizer().synthesize(meta, doc)

        ref_concepts = [
            "vector",
            "matrix",
            "linear transformation",
            "machine learning",
        ]

        result = evaluate_study_note(note, reference_concepts=ref_concepts)

        assert result.video_id == "DrCeIbpfuzE"
        assert result.total_claims > 0
        assert result.grounded_claims == result.total_claims
        assert result.unsupported_claim_rate == 0.0
        assert result.latex_equation_fidelity == 1.0
        assert result.timestamp_accuracy_rate == 1.0
        assert result.concept_coverage >= 0.75
        assert result.is_acceptable

    def test_evaluate_empty_transcript_note(self):
        meta = YouTubeVideoMetadata(
            video_id="empty_vid",
            title="Empty Video",
            source_url="https://youtube.com/watch?v=empty_vid",
            duration_seconds=100.0,
        )
        note = StudyNoteSynthesizer().synthesize(meta, transcript_doc=None)
        result = evaluate_study_note(note)

        assert result.total_claims == 0
        assert result.unsupported_claim_rate == 0.0
        assert result.is_acceptable

    def test_formula_timestamp_out_of_bounds_detected(self):
        """Verify that out-of-bounds timestamps in formulas are strictly detected."""
        from youtube_learning_companion.schemas import (
            DefinitionItem,
            EvidenceReference,
            FormulaItem,
            NoteGenerationStatus,
            StudyNote,
            StudyNoteSection,
        )

        valid_ref = EvidenceReference(
            video_id="test_vid",
            start_seconds=10.0,
            end_seconds=20.0,
            snippet="Valid definition reference",
        )
        invalid_ref = EvidenceReference(
            video_id="test_vid",
            start_seconds=500.0,
            end_seconds=600.0,  # Exceeds duration 100.0s
            snippet="Out of bounds formula reference",
        )

        sec = StudyNoteSection(
            section_id="sec_1",
            title="Linear Systems",
            start_seconds=0.0,
            end_seconds=100.0,
            overview="Overview of linear systems",
            definitions=(
                DefinitionItem(
                    term="Vector",
                    definition="Element of vector space",
                    evidence_refs=(valid_ref,),
                ),
            ),
            formulas=(
                FormulaItem(
                    name="Linear Equation",
                    latex_expression="$A x = b$",
                    variables={"A": "Matrix", "x": "Vector", "b": "Vector"},
                    evidence_refs=(invalid_ref,),
                ),
            ),
        )

        note = StudyNote(
            note_id="test_note_1",
            video_id="test_vid",
            title="Test Note",
            video_title="Test Video Title",
            source_url="https://youtube.com/watch?v=test_vid",
            duration_seconds=100.0,
            coverage_percentage=100.0,
            status=NoteGenerationStatus.COMPLETE,
            executive_summary="Executive summary for test note",
            sections=(sec,),
        )

        result = evaluate_study_note(note)

        # Before fix: total_timestamps was 1, valid_timestamps was 1, is_acceptable was True
        # After fix: total_timestamps is 2, valid_timestamps is 1,
        # timestamp_accuracy_rate is 0.5, is_acceptable is False
        assert result.total_timestamps == 2
        assert result.valid_timestamps == 1
        assert result.timestamp_accuracy_rate == 0.5
        assert not result.is_acceptable
        assert any("Invalid timestamp interval" in n for n in result.notes)

    def test_exam_trap_and_takeaway_timestamp_validation(self):
        """Verify that inverted timestamps in traps and takeaways are detected."""
        from youtube_learning_companion.schemas import (
            EvidenceReference,
            ExamTrapItem,
            NoteGenerationStatus,
            StudyNote,
            StudyNoteSection,
            TakeawayItem,
        )

        inverted_ref_1 = EvidenceReference(
            video_id="test_vid",
            start_seconds=50.0,
            end_seconds=40.0,  # Inverted: end < start
            snippet="Inverted trap reference",
        )
        inverted_ref_2 = EvidenceReference(
            video_id="test_vid",
            start_seconds=80.0,
            end_seconds=70.0,  # Inverted: end < start
            snippet="Inverted takeaway reference",
        )

        sec = StudyNoteSection(
            section_id="sec_1",
            title="Matrix Properties",
            start_seconds=0.0,
            end_seconds=100.0,
            overview="Overview of matrices",
            exam_traps=(
                ExamTrapItem(
                    trap_description="Assuming commutativity",
                    correct_understanding="Matrix multiplication is non-commutative",
                    why_it_matters_in_gate="Check dimensions and non-commutativity",
                    evidence_refs=(inverted_ref_1,),
                ),
            ),
        )

        note = StudyNote(
            note_id="test_note_2",
            video_id="test_vid",
            title="Test Traps Note",
            video_title="Test Video Title",
            source_url="https://youtube.com/watch?v=test_vid",
            duration_seconds=100.0,
            coverage_percentage=100.0,
            status=NoteGenerationStatus.COMPLETE,
            executive_summary="Summary of traps note",
            sections=(sec,),
            key_takeaways=(
                TakeawayItem(
                    takeaway="Non-commutative",
                    evidence_refs=(inverted_ref_2,),
                ),
            ),
        )

        result = evaluate_study_note(note)

        assert result.total_timestamps == 2
        assert result.valid_timestamps == 0
        assert result.timestamp_accuracy_rate == 0.0
        assert not result.is_acceptable
        assert len(result.notes) >= 2

    def test_concept_coverage_evaluates_full_document_sections(self):
        """Verify concept coverage checks technical blocks (definitions, formulas, etc.)."""
        from youtube_learning_companion.schemas import (
            DefinitionItem,
            EvidenceReference,
            FormulaItem,
            NoteGenerationStatus,
            StudyNote,
            StudyNoteSection,
        )

        ref = EvidenceReference(
            video_id="test_vid",
            start_seconds=10.0,
            end_seconds=20.0,
            snippet="Reference",
        )

        sec = StudyNoteSection(
            section_id="sec_1",
            title="Section 1",
            start_seconds=0.0,
            end_seconds=100.0,
            overview="General discussion",  # does NOT contain 'eigenvalue' or 'diagonalization'
            definitions=(
                DefinitionItem(
                    term="Eigenvalue",
                    definition="Scalar factor of eigenvector",
                    evidence_refs=(ref,),
                ),
            ),
            formulas=(
                FormulaItem(
                    name="Diagonalization",
                    latex_expression="$A = P D P^{-1}$",
                    variables={"P": "Eigenvector matrix", "D": "Diagonal matrix"},
                    evidence_refs=(ref,),
                ),
            ),
        )

        note = StudyNote(
            note_id="test_note_3",
            video_id="test_vid",
            title="Test Concept Note",
            video_title="Test Video Title",
            source_url="https://youtube.com/watch?v=test_vid",
            duration_seconds=100.0,
            coverage_percentage=100.0,
            status=NoteGenerationStatus.COMPLETE,
            executive_summary="Brief intro",  # does NOT contain the concepts
            sections=(sec,),
        )

        # Before fix: coverage is 0.0 because terms in definitions/formulas were omitted
        # After fix: coverage is 1.0 (both 'eigenvalue' and 'diagonalization' found)
        result = evaluate_study_note(note, reference_concepts=["eigenvalue", "diagonalization"])
        assert result.concept_coverage == 1.0

