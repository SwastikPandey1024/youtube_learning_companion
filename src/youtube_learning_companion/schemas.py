"""Typed, validated Pydantic v2 schemas for VIDEX YouTube Learning Companion.

Defines schemas for:
- YouTube URLs, video and playlist discovery metadata
- Timestamped transcript segments and documents
- Evidence references tying study notes to exact source timestamps and modalities
- Technical learning artifacts: Definitions, Theorems, Formulas, Worked Examples,
  Exam Traps, Key Takeaways, Practice Questions, Uncertainties, and Sections
- Comprehensive Study Notes with validation results and generation status
- Resumable playlist manifests and processing state
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from importlib import import_module
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Optional core VIDEX domain interop placeholder
DomainTranscriptSegment: Any = Any

# ── Access and Status Enums ──────────────────────────────────────────────────


class VideoAccessStatus(StrEnum):
    """Access status of an individual YouTube video."""

    ACCESSIBLE = "accessible"
    PRIVATE = "private"
    UNAVAILABLE = "unavailable"
    DELETED = "deleted"
    REGION_RESTRICTED = "region_restricted"
    UNKNOWN = "unknown"


class PlaylistAccessStatus(StrEnum):
    """Access status of a YouTube playlist."""

    ACCESSIBLE = "accessible"
    PRIVATE = "private"
    NOT_FOUND = "not_found"
    EMPTY = "empty"
    ERROR = "error"


class TranscriptAcquisitionMethod(StrEnum):
    """Method used to acquire the transcript text."""

    OFFICIAL_API = "official_api"
    USER_FILE_JSON = "user_file_json"
    USER_FILE_VTT = "user_file_vtt"
    USER_FILE_SRT = "user_file_srt"
    USER_FILE_TXT = "user_file_txt"
    LOCAL_ASR = "local_asr"
    MOCK_FIXTURE = "mock_fixture"
    NONE = "none"


class TranscriptCoverageStatus(StrEnum):
    """Coverage declaration supplied by the transcript source or user."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class EvidenceSourceType(StrEnum):
    """Modalities supporting a study note claim."""

    TRANSCRIPT = "transcript"
    OCR = "ocr"
    VISUAL_FRAME = "visual_frame"
    AUDIO_ACOUSTIC = "audio_acoustic"
    MULTIMODAL = "multimodal"


class NoteItemType(StrEnum):
    """Semantic type of an individual study note entry."""

    DEFINITION = "definition"
    THEOREM = "theorem"
    FORMULA = "formula"
    WORKED_EXAMPLE = "worked_example"
    EXAM_TRAP = "exam_trap"
    KEY_TAKEAWAY = "key_takeaway"
    PRACTICE_QUESTION = "practice_question"
    UNCERTAINTY = "uncertainty"


class UncertaintyType(StrEnum):
    """Category of epistemic uncertainty or omitted source content."""

    AMBIGUOUS_SPEECH = "ambiguous_speech"
    CORRUPTED_FORMULA = "corrupted_formula"
    MISSING_VISUAL = "missing_visual"
    UNVERIFIED_CLAIM = "unverified_claim"
    OMITTED_DERIVATION = "omitted_derivation"
    TRANSCRIPT_UNAVAILABLE = "transcript_unavailable"


class NoteGenerationStatus(StrEnum):
    """Overall generation and grounding status of a study note."""

    COMPLETE = "complete"
    PARTIAL_SUCCESS = "partial_success"
    AWAITING_TRANSCRIPT = "awaiting_transcript"
    METADATA_ONLY = "metadata_only"
    SKIPPED = "skipped"
    FAILED = "failed"


# ── Metadata Models ──────────────────────────────────────────────────────────


class YouTubeVideoMetadata(BaseModel):
    """Normalized metadata for a single YouTube video."""

    model_config = ConfigDict(frozen=True)

    video_id: str = Field(..., description="Canonical 11-character YouTube video ID")
    title: str = Field(..., description="Video title as listed on YouTube")
    source_url: str = Field(..., description="Full canonical YouTube watch URL")
    channel_id: str | None = Field(default=None, description="YouTube Channel ID")
    channel_title: str | None = Field(default=None, description="Channel / lecturer name")
    duration_seconds: float | None = Field(
        default=None, ge=0.0, description="Video duration in seconds if known"
    )
    description: str | None = Field(default=None, description="Video description text from YouTube")
    published_at: str | None = Field(default=None, description="Publication timestamp ISO string")
    position: int | None = Field(
        default=None, ge=0, description="Zero-based sequence position within playlist"
    )
    access_status: VideoAccessStatus = Field(
        default=VideoAccessStatus.ACCESSIBLE, description="Accessibility state"
    )
    error_message: str | None = Field(
        default=None, description="Error explanation if video is inaccessible"
    )

    @field_validator("video_id")
    @classmethod
    def _validate_video_id(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            msg = "video_id cannot be empty"
            raise ValueError(msg)
        return clean


class YouTubePlaylistMetadata(BaseModel):
    """Normalized metadata for a YouTube playlist."""

    model_config = ConfigDict(frozen=True)

    playlist_id: str = Field(..., description="Canonical YouTube playlist ID (e.g. PL...)")
    title: str = Field(..., description="Playlist title")
    source_url: str = Field(..., description="Canonical playlist URL")
    channel_id: str | None = Field(default=None, description="Channel ID")
    channel_title: str | None = Field(default=None, description="Channel name")
    description: str | None = Field(default=None, description="Playlist description")
    video_count: int = Field(
        default=0, ge=0, description="Reported or discovered total video count"
    )
    access_status: PlaylistAccessStatus = Field(
        default=PlaylistAccessStatus.ACCESSIBLE, description="Playlist access status"
    )
    videos: tuple[YouTubeVideoMetadata, ...] = Field(
        default_factory=tuple, description="Discovered videos preserving playlist sequence"
    )


# ── Transcript Models ────────────────────────────────────────────────────────


class TranscriptSegment(BaseModel):
    """Transcript text with explicit provenance for each timing endpoint."""

    model_config = ConfigDict(frozen=True)

    segment_id: str = Field(default_factory=lambda: str(uuid4()))
    video_id: str = Field(..., description="Parent video ID")
    start_seconds: float = Field(..., ge=0.0, description="Start timestamp in seconds")
    end_seconds: float = Field(..., ge=0.0, description="End timestamp in seconds")
    start_timestamp_is_source: bool = Field(
        default=True, description="Whether start_seconds comes from the transcript source"
    )
    end_timestamp_is_source: bool = Field(
        default=True, description="Whether end_seconds comes from the transcript source"
    )
    text: str = Field(..., min_length=1, description="Spoken transcript text")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score if available"
    )
    speaker: str | None = Field(default=None, description="Speaker or lecturer identifier")

    @property
    def duration_seconds(self) -> float:
        """Temporal duration of this transcript segment."""
        return max(0.0, self.end_seconds - self.start_seconds)

    @model_validator(mode="after")
    def _validate_interval(self) -> TranscriptSegment:
        if self.end_seconds < self.start_seconds:
            msg = (
                f"end_seconds ({self.end_seconds}) cannot be before start_seconds "
                f"({self.start_seconds})"
            )
            raise ValueError(msg)
        return self

    def to_domain_segment(
        self, video_uuid: UUID | None = None, provider: str = "companion"
    ) -> DomainTranscriptSegment:
        """Convert this companion TranscriptSegment to a VIDEX domain TranscriptSegment.

        Mapping & Unit Semantics:
        - Timestamps: Both companion (start_seconds, end_seconds) and core domain
          (start_timestamp_seconds, end_timestamp_seconds) are expressed in float seconds.
        - Video Identifier: Companion uses 11-char YouTube string ID. Domain uses UUID.
          If `video_uuid` is None, a deterministic UUID5 derived from NAMESPACE_URL
          and the video URL is generated.
        - Provider: Defaults to 'companion'.
        """
        if not (self.start_timestamp_is_source and self.end_timestamp_is_source):
            raise ValueError(
                "Cannot convert a segment with inferred timestamps to VIDEX because the "
                "domain segment cannot preserve timestamp provenance."
            )

        from uuid import NAMESPACE_URL, uuid5

        try:
            domain_schemas = import_module("videx.domain.schemas")
            domain_transcript_segment = domain_schemas.TranscriptSegment
        except ImportError as err:
            raise ImportError(
                "Converting to core VIDEX DomainTranscriptSegment requires the optional "
                "'videx' package. Install videx or use standalone TranscriptSegment."
            ) from err

        v_id = video_uuid or uuid5(NAMESPACE_URL, f"https://youtube.com/watch?v={self.video_id}")
        return domain_transcript_segment(
            video_id=v_id,
            start_timestamp_seconds=self.start_seconds,
            end_timestamp_seconds=self.end_seconds,
            raw_text=self.text,
            normalized_text=" ".join(self.text.split()).strip(),
            confidence=self.confidence if self.confidence is not None else 1.0,
            provider=provider,
            speaker_id=self.speaker,
        )

    @classmethod
    def from_domain_segment(
        cls, seg: DomainTranscriptSegment, video_id_str: str = ""
    ) -> TranscriptSegment:
        """Convert a VIDEX domain TranscriptSegment to a companion TranscriptSegment.

        Mapping & Unit Semantics:
        - Timestamps: start_timestamp_seconds and end_timestamp_seconds mapped to
          start_seconds and end_seconds (both float seconds).
        - Video Identifier: Uses video_id_str if provided; otherwise str(seg.video_id).
        - Text: raw_text mapped to text.
        - Validation: Enforces end_seconds >= start_seconds and confidence in [0.0, 1.0].
        """
        return cls(
            video_id=video_id_str or str(getattr(seg, "video_id", "")),
            start_seconds=getattr(seg, "start_timestamp_seconds", 0.0),
            end_seconds=getattr(seg, "end_timestamp_seconds", 0.0),
            text=getattr(seg, "raw_text", getattr(seg, "text", "")),
            confidence=getattr(seg, "confidence", None),
            speaker=getattr(seg, "speaker_id", None),
        )


class TranscriptDocument(BaseModel):
    """Complete collection of transcript segments for a video."""

    model_config = ConfigDict(frozen=True)

    video_id: str = Field(..., description="Parent YouTube video ID")
    language: str = Field(default="en", description="ISO language code (e.g. 'en', 'hi')")
    is_auto_generated: bool = Field(
        default=True, description="Whether captions are machine-generated"
    )
    acquisition_method: TranscriptAcquisitionMethod = Field(
        ..., description="Method through which this transcript was acquired"
    )
    source_path_or_url: str | None = Field(
        default=None, description="Local file path or remote URL used to acquire transcript"
    )
    segments: tuple[TranscriptSegment, ...] = Field(
        default_factory=tuple, description="Ordered transcript segments"
    )
    coverage_status: TranscriptCoverageStatus = Field(
        default=TranscriptCoverageStatus.UNKNOWN,
        description="Coverage status declared by a source or user, not inferred from density",
    )
    coverage_rationale: str | None = Field(
        default=None, description="Evidence or attestation supporting the coverage declaration"
    )
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Acquisition timestamp"
    )

    @model_validator(mode="after")
    def _require_coverage_rationale(self) -> TranscriptDocument:
        if self.coverage_status != TranscriptCoverageStatus.UNKNOWN and not (
            self.coverage_rationale and self.coverage_rationale.strip()
        ):
            raise ValueError("A coverage rationale is required for complete or partial status")
        return self

    @property
    def total_duration_seconds(self) -> float:
        """Total duration spanned by the transcript."""
        if not self.segments:
            return 0.0
        return max(s.end_seconds for s in self.segments)

    @property
    def word_count(self) -> int:
        """Total word count across all segments."""
        return sum(len(s.text.split()) for s in self.segments)


# ── Evidence and Grounding Reference ─────────────────────────────────────────


class EvidenceReference(BaseModel):
    """Explicit provenance reference tying a generated claim to source evidence."""

    model_config = ConfigDict(frozen=True)

    reference_id: str = Field(default_factory=lambda: str(uuid4()))
    video_id: str = Field(..., description="Canonical video ID")
    start_seconds: float = Field(..., ge=0.0, description="Start presentation timestamp in seconds")
    end_seconds: float = Field(..., ge=0.0, description="End presentation timestamp in seconds")
    start_timestamp_is_source: bool = Field(
        default=True, description="Whether start_seconds comes from source evidence"
    )
    end_timestamp_is_source: bool = Field(
        default=True, description="Whether end_seconds comes from source evidence"
    )
    source_type: EvidenceSourceType = Field(
        default=EvidenceSourceType.TRANSCRIPT, description="Evidence source modality"
    )
    segment_id: str | None = Field(default=None, description="Referenced TranscriptSegment ID")
    frame_id: UUID | None = Field(default=None, description="Referenced VIDEX Frame UUID if visual")
    evidence_id: UUID | None = Field(
        default=None, description="Referenced VIDEX Evidence UUID if visual/OCR"
    )
    snippet: str = Field(default="", description="Relevant textual quote or OCR transcription")

    @property
    def clickable_url(self) -> str | None:
        """Clickable YouTube timestamp link when its start time is source-backed."""
        if not self.start_timestamp_is_source:
            return None
        sec = int(self.start_seconds)
        return f"https://youtube.com/watch?v={self.video_id}&t={sec}s"

    @property
    def timestamp_label(self) -> str:
        """Formatted timestamp label like MM:SS or HH:MM:SS."""
        if not self.start_timestamp_is_source:
            return "inferred timing"
        total_sec = int(self.start_seconds)
        hrs = total_sec // 3600
        mins = (total_sec % 3600) // 60
        secs = total_sec % 60
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"


# ── Technical Note Items ─────────────────────────────────────────────────────


class DefinitionItem(BaseModel):
    """A formal mathematical or conceptual definition."""

    model_config = ConfigDict(frozen=True)

    term: str = Field(..., description="The mathematical term or concept defined")
    definition: str = Field(..., description="Precise definition statement")
    notation: str | None = Field(
        default=None, description="Standard mathematical notation (e.g. 'v in R^n')"
    )
    assumptions: tuple[str, ...] = Field(
        default_factory=tuple, description="Explicit assumptions or preconditions"
    )
    evidence_refs: tuple[EvidenceReference, ...] = Field(
        default_factory=tuple, description="Source evidence grounding this definition"
    )


class TheoremItem(BaseModel):
    """A formal theorem, lemma, proposition, or corollary."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Theorem or property name")
    statement: str = Field(..., description="Precise mathematical statement")
    conditions: tuple[str, ...] = Field(
        default_factory=tuple, description="Required conditions / hypotheses"
    )
    conclusion: str = Field(..., description="Mathematical consequence or result")
    intuition_or_proof: str | None = Field(
        default=None, description="High-level intuition or proof summary from lecture"
    )
    evidence_refs: tuple[EvidenceReference, ...] = Field(
        default_factory=tuple, description="Source evidence grounding this theorem"
    )


class FormulaItem(BaseModel):
    """A mathematical formula, equation, or derivation step."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Formula name or purpose")
    latex_expression: str = Field(
        ..., description="Standard LaTeX formula (e.g. '$A x = \\lambda x$')"
    )
    variables: dict[str, str] = Field(
        default_factory=dict, description="Explanation of each variable or symbol"
    )
    conditions: tuple[str, ...] = Field(
        default_factory=tuple, description="Conditions under which formula applies"
    )
    evidence_refs: tuple[EvidenceReference, ...] = Field(
        default_factory=tuple, description="Source evidence grounding this formula"
    )


class WorkedExample(BaseModel):
    """A concrete worked example or calculation presented in the lecture."""

    model_config = ConfigDict(frozen=True)

    problem_statement: str = Field(..., description="The example problem statement")
    step_by_step_solution: tuple[str, ...] = Field(
        default_factory=tuple, description="Ordered steps of the calculation or derivation"
    )
    final_result: str = Field(..., description="Final answer or conclusion")
    key_insight: str | None = Field(
        default=None, description="Pedagogical takeaway or insight illustrated"
    )
    evidence_refs: tuple[EvidenceReference, ...] = Field(
        default_factory=tuple, description="Source evidence grounding this example"
    )


class ExamTrapItem(BaseModel):
    """A common pitfall, misconception, or exam trap for competitive exams like GATE."""

    model_config = ConfigDict(frozen=True)

    trap_description: str = Field(..., description="The common mistake or false assumption")
    correct_understanding: str = Field(
        ..., description="The mathematically rigorous correct concept"
    )
    why_it_matters_in_gate: str = Field(
        ..., description="Why GATE examiners test this and how questions exploit it"
    )
    is_directly_stated_in_lecture: bool = Field(
        default=True,
        description="True if lecturer explicitly highlighted this; False if inferred from concept",
    )
    evidence_refs: tuple[EvidenceReference, ...] = Field(
        default_factory=tuple, description="Source evidence grounding this trap"
    )


class TakeawayItem(BaseModel):
    """A critical high-yield takeaway or revision bullet."""

    model_config = ConfigDict(frozen=True)

    takeaway: str = Field(..., description="Core concept or key takeaway statement")
    importance: str = Field(
        default="high", description="Importance rating: 'high', 'essential', 'revision'"
    )
    evidence_refs: tuple[EvidenceReference, ...] = Field(
        default_factory=tuple, description="Source evidence grounding this takeaway"
    )


class PracticeQuestionItem(BaseModel):
    """A practice question directly grounded in the lecture content."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(..., description="Self-test question for exam preparation")
    answer_or_hint: str = Field(..., description="Correct answer or solution hint")
    question_type: str = Field(
        default="conceptual", description="Type: 'conceptual', 'numerical', 'true_false'"
    )
    evidence_refs: tuple[EvidenceReference, ...] = Field(
        default_factory=tuple, description="Source evidence grounding this question"
    )


class UncertaintyItem(BaseModel):
    """Epistemic uncertainty or omitted source content flag."""

    model_config = ConfigDict(frozen=True)

    uncertainty_type: UncertaintyType = Field(..., description="Category of uncertainty")
    description: str = Field(..., description="Explanation of what is uncertain or omitted")
    affected_concept_or_timestamp: str = Field(
        ..., description="Timestamp or concept impacted by the uncertainty"
    )
    suggested_resolution: str = Field(
        ..., description="Action needed to resolve (e.g. 'Supply visual frame OCR')"
    )


# ── Study Note Section & Complete Note ───────────────────────────────────────


class StudyNoteSection(BaseModel):
    """A thematic section of a study note spanning a temporal window."""

    model_config = ConfigDict(frozen=True)

    section_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Section title")
    start_seconds: float = Field(..., ge=0.0, description="Section start timestamp")
    end_seconds: float = Field(..., ge=0.0, description="Section end timestamp")
    start_timestamp_is_source: bool = Field(
        default=True, description="Whether start_seconds comes from source transcript timing"
    )
    end_timestamp_is_source: bool = Field(
        default=True, description="Whether end_seconds comes from source transcript timing"
    )
    overview: str = Field(..., description="Section overview and narrative explanation")
    definitions: tuple[DefinitionItem, ...] = Field(
        default_factory=tuple, description="Definitions introduced in this section"
    )
    theorems: tuple[TheoremItem, ...] = Field(
        default_factory=tuple, description="Theorems / properties introduced in this section"
    )
    formulas: tuple[FormulaItem, ...] = Field(
        default_factory=tuple, description="Formulas introduced in this section"
    )
    worked_examples: tuple[WorkedExample, ...] = Field(
        default_factory=tuple, description="Worked examples covered in this section"
    )
    exam_traps: tuple[ExamTrapItem, ...] = Field(
        default_factory=tuple, description="Common traps covered in this section"
    )

    @property
    def clickable_url(self) -> str | None:
        if not self.start_timestamp_is_source:
            return None
        sec = int(self.start_seconds)
        return f"&t={sec}s"


class NoteValidationResult(BaseModel):
    """Quality and grounding validation audit for intra-synthesis section claims.

    Evaluates the 5 core section-level technical item types (definitions, theorems,
    formulas, worked examples, and section exam traps) produced during segment synthesis.
    Global lecture-level elements (e.g. key takeaways) are independently evaluated by
    evaluate_study_note().
    """

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(..., description="Whether note passes validation checks")
    total_claims: int = Field(
        default=0, ge=0, description="Total substantive section claims assessed"
    )
    grounded_claims: int = Field(
        default=0, ge=0, description="Claims with valid evidence references"
    )
    unsupported_claims: int = Field(
        default=0, ge=0, description="Claims lacking valid evidence references"
    )
    grounding_ratio: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Grounded claims / total claims"
    )
    formulas_valid_latex: bool = Field(
        default=True, description="Whether all formulas have balanced LaTeX syntax"
    )
    timestamps_within_duration: bool = Field(
        default=True, description="Whether all evidence timestamps are within video duration"
    )
    warnings: tuple[str, ...] = Field(
        default_factory=tuple, description="Non-fatal validation warnings"
    )


class StudyNote(BaseModel):
    """Complete, evidence-grounded study note for a single lecture video."""

    model_config = ConfigDict(frozen=True)

    note_id: str = Field(default_factory=lambda: str(uuid4()))
    video_id: str = Field(..., description="Canonical YouTube video ID")
    video_title: str = Field(..., description="Video title")
    source_url: str = Field(..., description="Canonical YouTube watch URL")
    playlist_id: str | None = Field(default=None, description="Parent playlist ID if applicable")
    channel_title: str | None = Field(default=None, description="Channel / lecturer name")
    duration_seconds: float = Field(
        default=0.0, ge=0.0, description="Total video duration in seconds"
    )
    coverage_percentage: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Temporal video duration percentage covered by transcript",
    )
    temporal_coverage_percentage: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Explicit temporal span percentage of video timeline covered",
    )
    density_coverage_percentage: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Transcript text density coverage vs typical conversational speaking rate",
    )
    transcript_completeness: str = Field(
        default="unknown",
        description="Coverage classification: 'complete', 'partial', 'unknown', or 'empty'",
    )
    transcript_completeness_rationale: str = Field(
        default="Transcript coverage was not declared by the source.",
        description="Reason for the transcript coverage classification",
    )
    learning_objectives: tuple[str, ...] = Field(
        default_factory=tuple, description="Concise learning objectives of this lecture"
    )
    executive_summary: str = Field(
        ..., description="Executive summary and high-level conceptual roadmap"
    )
    sections: tuple[StudyNoteSection, ...] = Field(
        default_factory=tuple, description="Thematic sections with clickable timestamps"
    )
    key_takeaways: tuple[TakeawayItem, ...] = Field(
        default_factory=tuple, description="High-yield takeaways for revision"
    )
    exam_traps: tuple[ExamTrapItem, ...] = Field(
        default_factory=tuple, description="Exam traps and common pitfalls"
    )
    practice_questions: tuple[PracticeQuestionItem, ...] = Field(
        default_factory=tuple, description="Source-grounded practice questions"
    )
    uncertainties: tuple[UncertaintyItem, ...] = Field(
        default_factory=tuple, description="Ambiguities or omitted content warnings"
    )
    status: NoteGenerationStatus = Field(
        default=NoteGenerationStatus.COMPLETE, description="Generation completion status"
    )
    validation: NoteValidationResult | None = Field(
        default=None, description="Validation and grounding audit result"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Generation timestamp"
    )
    generation_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Synthesizer configuration, prompt version, model info"
    )


# ── Playlist Orchestration & Manifest Models ─────────────────────────────────


class VideoProcessingRecord(BaseModel):
    """Processing and export status of a single video in a playlist."""

    model_config = ConfigDict(frozen=True)

    video_id: str = Field(..., description="YouTube video ID")
    position: int = Field(..., ge=0, description="Order index in playlist")
    title: str = Field(..., description="Video title")
    source_url: str = Field(..., description="Source URL")
    access_status: VideoAccessStatus = Field(
        default=VideoAccessStatus.ACCESSIBLE, description="Video accessibility"
    )
    status: NoteGenerationStatus = Field(
        default=NoteGenerationStatus.METADATA_ONLY, description="Note generation status"
    )
    transcript_method: TranscriptAcquisitionMethod = Field(
        default=TranscriptAcquisitionMethod.NONE, description="Method used for transcript"
    )
    note_file_path: str | None = Field(
        default=None, description="Relative or absolute path to generated markdown file"
    )
    error_message: str | None = Field(default=None, description="Error explanation if failed")
    last_attempt_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last attempt timestamp"
    )


class PlaylistProcessingManifest(BaseModel):
    """Resumable job manifest tracking processing of a complete YouTube playlist."""

    model_config = ConfigDict(frozen=True)

    manifest_id: str = Field(default_factory=lambda: str(uuid4()))
    playlist_id: str = Field(..., description="YouTube playlist ID")
    playlist_title: str = Field(..., description="Playlist title")
    source_url: str = Field(..., description="Playlist URL")
    total_videos_discovered: int = Field(
        default=0, ge=0, description="Total videos discovered in playlist"
    )
    processed_count: int = Field(default=0, ge=0, description="Successfully generated notes count")
    partial_count: int = Field(default=0, ge=0, description="Partially generated notes count")
    awaiting_transcript_count: int = Field(
        default=0, ge=0, description="Videos with metadata only awaiting transcript"
    )
    skipped_count: int = Field(
        default=0, ge=0, description="Skipped videos (e.g. private, deleted)"
    )
    failed_count: int = Field(default=0, ge=0, description="Failed videos count")
    pending_count: int = Field(default=0, ge=0, description="Videos not yet attempted")
    records: tuple[VideoProcessingRecord, ...] = Field(
        default_factory=tuple, description="Per-video tracking records in playlist order"
    )
    output_dir: str = Field(default="data/notes", description="Output export directory")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Manifest creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Manifest last updated timestamp"
    )
