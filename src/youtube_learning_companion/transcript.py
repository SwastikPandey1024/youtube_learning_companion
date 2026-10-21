"""Compliant transcript and media access layer for VIDEX Learning Companion.

Defines:
- BaseTranscriptProvider interface
- FileTranscriptProvider supporting JSON, VTT, SRT, and TXT files with timestamps
- YouTubeCaptionProvider with compliance safeguards and clear error classifications
- VIDEXASRTranscriptProvider invoking local VIDEX ASR on authorized media files
- CompositeTranscriptProvider resolving transcripts across providers with detailed provenance
"""

from __future__ import annotations

import abc
import json
import logging
import math
import re
from importlib import import_module
from pathlib import Path
from typing import Any
from uuid import uuid4

from youtube_learning_companion.schemas import (
    TranscriptAcquisitionMethod,
    TranscriptCoverageStatus,
    TranscriptDocument,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)


def parse_timestamp_seconds(ts_str: str) -> float:
    """Parse a timestamp string like '01:23.456', '01:23:45.678', or '123.45' into seconds."""
    s = ts_str.strip().replace(",", ".")
    parts = s.split(":")
    if not s or len(parts) not in {1, 2, 3}:
        raise ValueError(f"Invalid timestamp: {ts_str!r}")

    try:
        values = [float(part) for part in parts]
    except ValueError as err:
        raise ValueError(f"Invalid timestamp: {ts_str!r}") from err

    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError(f"Timestamp values must be finite and non-negative: {ts_str!r}")
    if len(values) == 3:
        if values[1] >= 60.0 or values[2] >= 60.0:
            raise ValueError(f"Invalid minute or second component: {ts_str!r}")
        return values[0] * 3600.0 + values[1] * 60.0 + values[2]
    if len(values) == 2:
        if values[1] >= 60.0:
            raise ValueError(f"Invalid seconds component: {ts_str!r}")
        return values[0] * 60.0 + values[1]
    return values[0]


class BaseTranscriptProvider(abc.ABC):
    """Abstract interface for lecture transcript acquisition."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Provider name identifier."""

    @abc.abstractmethod
    def get_transcript(self, video_id: str, **kwargs: object) -> TranscriptDocument | None:
        """Retrieve transcript for the given video ID, or return None if unavailable."""


class FileTranscriptProvider(BaseTranscriptProvider):
    """Loads user-supplied or benchmark transcript files (JSON, VTT, SRT, TXT)."""

    def __init__(
        self,
        transcript_dir: str | Path | None = None,
        file_map: dict[str, str | Path] | None = None,
    ) -> None:
        self.transcript_dir = Path(transcript_dir) if transcript_dir else None
        self.file_map = {k: Path(v) for k, v in (file_map or {}).items()}

    @property
    def name(self) -> str:
        return "file_transcript_provider"

    def register_file(self, video_id: str, file_path: str | Path) -> None:
        """Explicitly register a transcript file path for a video ID."""
        self.file_map[video_id] = Path(file_path)

    def get_transcript(self, video_id: str, **kwargs: object) -> TranscriptDocument | None:
        """Locate and parse transcript file for video_id."""
        # 1. Check explicit file mapping
        if video_id in self.file_map:
            p = self.file_map[video_id]
            if p.exists():
                return self._parse_file(video_id, p)
            logger.warning("Mapped transcript file not found: %s", p)

        # 2. Check transcript_dir for common extensions
        if self.transcript_dir and self.transcript_dir.exists():
            for ext in [".json", ".vtt", ".srt", ".txt"]:
                candidate = self.transcript_dir / f"{video_id}{ext}"
                if candidate.exists():
                    return self._parse_file(video_id, candidate)

        return None

    def _parse_file(self, video_id: str, path: Path) -> TranscriptDocument | None:
        """Route to appropriate file parser based on extension."""
        suffix = path.suffix.lower()
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("Error reading transcript file %s: %s", path, e)
            return None

        try:
            if suffix == ".json":
                return self._parse_json(video_id, content, path)
            if suffix == ".vtt":
                return self._parse_vtt(video_id, content, path)
            if suffix == ".srt":
                return self._parse_srt(video_id, content, path)
            if suffix == ".txt":
                return self._parse_txt(video_id, content, path)
        except (TypeError, ValueError, OverflowError) as e:
            logger.error("Invalid transcript data in %s: %s", path, e)
            return None

        logger.warning("Unsupported transcript extension: %s", suffix)
        return None

    def _parse_json(self, video_id: str, content: str, path: Path) -> TranscriptDocument | None:
        """Parse structured JSON transcript."""
        try:
            data = json.loads(content)
        except Exception as e:
            logger.error("Invalid JSON in transcript %s: %s", path, e)
            return None

        raw_segments: list[dict[str, Any]] = []
        lang = "en"
        auto_gen = False
        coverage_status = TranscriptCoverageStatus.UNKNOWN
        coverage_rationale = None

        if isinstance(data, list):
            raw_segments = [s for s in data if isinstance(s, dict)]
        elif isinstance(data, dict):
            source_video_id = data.get("video_id")
            if source_video_id is not None and str(source_video_id) != video_id:
                raise ValueError(
                    f"Transcript video_id {source_video_id!r} does not match requested "
                    f"{video_id!r}"
                )
            lang = str(data.get("language", "en"))
            auto_gen = bool(data.get("is_auto_generated", False))
            coverage_status = TranscriptCoverageStatus(data.get("coverage_status", "unknown"))
            coverage_rationale = data.get("coverage_rationale")
            candidate = data.get("segments", data.get("items", []))
            if isinstance(candidate, list):
                raw_segments = [s for s in candidate if isinstance(s, dict)]

        segments: list[TranscriptSegment] = []
        for item in raw_segments:
            source_video_id = item.get("video_id")
            if source_video_id is not None and str(source_video_id) != video_id:
                raise ValueError(
                    f"Segment video_id {source_video_id!r} does not match requested {video_id!r}"
                )
            start_value = item.get("start_seconds", item.get("start"))
            if start_value is None:
                raise ValueError("Segment is missing a start timestamp")
            start = float(start_value)
            if "end_seconds" in item:
                end = float(item["end_seconds"])
            elif "end" in item:
                end = float(item["end"])
            elif "dur" in item or "duration" in item:
                end = start + float(item.get("dur", item.get("duration", 0.0)))
            else:
                raise ValueError("Segment is missing an end timestamp or duration")

            text_value = item.get("text", "")
            if not isinstance(text_value, str):
                raise ValueError("Segment text must be a string")
            text = text_value
            if not text:
                raise ValueError("Segment text cannot be empty")

            conf = float(item["confidence"]) if "confidence" in item else None
            spk = item.get("speaker")
            segment_id = str(item.get("segment_id") or uuid4())

            segments.append(
                TranscriptSegment(
                    segment_id=segment_id,
                    video_id=video_id,
                    start_seconds=start,
                    end_seconds=end,
                    text=text,
                    confidence=conf,
                    speaker=spk,
                )
            )

        if not segments:
            return None

        self._validate_segment_order(segments)
        return TranscriptDocument(
            video_id=video_id,
            language=lang,
            is_auto_generated=auto_gen,
            acquisition_method=TranscriptAcquisitionMethod.USER_FILE_JSON,
            source_path_or_url=str(path),
            segments=tuple(segments),
            coverage_status=coverage_status,
            coverage_rationale=coverage_rationale,
        )

    def _parse_vtt(self, video_id: str, content: str, path: Path) -> TranscriptDocument | None:
        """Parse WebVTT transcript file."""
        lines = content.splitlines()
        segments: list[TranscriptSegment] = []
        cue_re = re.compile(
            r"((?:\d{2}:)?\d{2}:\d{2}[\.,]\d{3})\s*-->\s*((?:\d{2}:)?\d{2}:\d{2}[\.,]\d{3})"
        )

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            m = cue_re.search(line)
            if m:
                start_sec = parse_timestamp_seconds(m.group(1))
                end_sec = parse_timestamp_seconds(m.group(2))
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip() and not cue_re.search(lines[i]):
                    text_lines.append(lines[i].strip())
                    i += 1
                text = " ".join(text_lines).strip()
                # Clean html/vtt tags like <c> </c>
                text = re.sub(r"<[^>]+>", "", text)
                if text:
                    segments.append(
                        TranscriptSegment(
                            segment_id=str(uuid4()),
                            video_id=video_id,
                            start_seconds=start_sec,
                            end_seconds=end_sec,
                            text=text,
                        )
                    )
            else:
                i += 1

        if not segments:
            return None

        self._validate_segment_order(segments)
        return TranscriptDocument(
            video_id=video_id,
            language="en",
            is_auto_generated=False,
            acquisition_method=TranscriptAcquisitionMethod.USER_FILE_VTT,
            source_path_or_url=str(path),
            segments=tuple(segments),
        )

    def _parse_srt(self, video_id: str, content: str, path: Path) -> TranscriptDocument | None:
        """Parse SubRip (.srt) transcript file."""
        blocks = re.split(r"\n\s*\n", content.strip())
        segments: list[TranscriptSegment] = []
        cue_re = re.compile(r"(\d{2}:\d{2}:\d{2}[\.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[\.,]\d{3})")

        for block in blocks:
            lines = [b.strip() for b in block.splitlines() if b.strip()]
            if not lines:
                continue
            # Find timing line
            for idx, line in enumerate(lines):
                m = cue_re.search(line)
                if m:
                    start_sec = parse_timestamp_seconds(m.group(1))
                    end_sec = parse_timestamp_seconds(m.group(2))
                    text_lines = lines[idx + 1 :]
                    text = " ".join(text_lines).strip()
                    text = re.sub(r"<[^>]+>", "", text)
                    if text:
                        segments.append(
                            TranscriptSegment(
                                segment_id=str(uuid4()),
                                video_id=video_id,
                                start_seconds=start_sec,
                                end_seconds=end_sec,
                                text=text,
                            )
                        )
                    break

        if not segments:
            return None

        self._validate_segment_order(segments)
        return TranscriptDocument(
            video_id=video_id,
            language="en",
            is_auto_generated=False,
            acquisition_method=TranscriptAcquisitionMethod.USER_FILE_SRT,
            source_path_or_url=str(path),
            segments=tuple(segments),
        )

    @staticmethod
    def _validate_segment_order(segments: list[TranscriptSegment]) -> None:
        for previous, current in zip(segments, segments[1:], strict=False):
            if current.start_seconds < previous.start_seconds:
                raise ValueError(
                    f"Segments are out of order: {current.segment_id} starts before "
                    f"{previous.segment_id}"
                )

    def _parse_txt(self, video_id: str, content: str, path: Path) -> TranscriptDocument | None:
        """Parse line-by-line timestamped text (e.g. '[00:15] hello world')."""
        lines = content.splitlines()
        segments: list[TranscriptSegment] = []
        ts_re = re.compile(r"^\[?((?:\d{2}:)?\d{2}:\d{2})\]?\s*(.*)$")

        current_start = 0.0
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            m = ts_re.match(line_str)
            if m:
                start_sec = parse_timestamp_seconds(m.group(1))
                text = m.group(2).strip()
                if text:
                    segments.append(
                        TranscriptSegment(
                            segment_id=str(uuid4()),
                            video_id=video_id,
                            start_seconds=start_sec,
                            end_seconds=start_sec + 5.0,
                            text=text,
                        )
                    )
            else:
                segments.append(
                    TranscriptSegment(
                        segment_id=str(uuid4()),
                        video_id=video_id,
                        start_seconds=current_start,
                        end_seconds=current_start + 10.0,
                        text=line_str,
                    )
                )
                current_start += 10.0

        if not segments:
            return None

        return TranscriptDocument(
            video_id=video_id,
            language="en",
            is_auto_generated=False,
            acquisition_method=TranscriptAcquisitionMethod.USER_FILE_TXT,
            source_path_or_url=str(path),
            segments=tuple(segments),
        )


class YouTubeCaptionProvider(BaseTranscriptProvider):
    """Compliant YouTube caption provider adhering to official API and policy boundaries."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "youtube_caption_provider"

    def get_transcript(self, video_id: str, **kwargs: object) -> TranscriptDocument | None:
        """Attempt caption acquisition.

        If official caption download requires OAuth scopes or timedtext is blocked,
        logs clear compliance guidance without attempting unauthorized circumvention.
        """
        logger.info(
            "YouTube caption download for %s requires OAuth authorization or user transcript.",
            video_id,
        )
        return None


class VIDEXASRTranscriptProvider(BaseTranscriptProvider):
    """Transcribes an authorized local audio/video media file using VIDEX ASR."""

    def __init__(self, media_dir: str | Path | None = None) -> None:
        self.media_dir = Path(media_dir) if media_dir else None

    @property
    def name(self) -> str:
        return "videx_asr_transcript_provider"

    def get_transcript(self, video_id: str, **kwargs: object) -> TranscriptDocument | None:
        """Transcribe authorized media if present."""
        if not self.media_dir or not self.media_dir.exists():
            return None

        for ext in [".mp4", ".mkv", ".wav", ".mp3", ".m4a"]:
            media_path = self.media_dir / f"{video_id}{ext}"
            if media_path.exists():
                return self._transcribe_media(video_id, media_path)

        return None

    def _transcribe_media(self, video_id: str, media_path: Path) -> TranscriptDocument | None:
        """Invoke VIDEX ASR on the media file."""
        try:
            audio_pipeline = import_module("videx.audio.pipeline").AudioPipeline

            pipeline = audio_pipeline()
            # Process authorized media
            pipeline_result = pipeline.process_video(video_source=media_path)
            asr_segments = pipeline_result.fused_segments or pipeline_result.raw_segments

            segments = [
                TranscriptSegment(
                    segment_id=str(uuid4()),
                    video_id=video_id,
                    start_seconds=s.start_timestamp_seconds,
                    end_seconds=s.end_timestamp_seconds,
                    text=s.raw_text,
                    confidence=s.confidence,
                )
                for s in asr_segments
            ]
            if segments:
                return TranscriptDocument(
                    video_id=video_id,
                    language="en",
                    is_auto_generated=True,
                    acquisition_method=TranscriptAcquisitionMethod.LOCAL_ASR,
                    source_path_or_url=str(media_path),
                    segments=tuple(segments),
                )
        except Exception as e:
            logger.warning("Local VIDEX ASR transcription failed for %s: %s", media_path, e)

        return None


class CompositeTranscriptProvider(BaseTranscriptProvider):
    """Chained provider resolving transcripts in priority order."""

    def __init__(self, providers: list[BaseTranscriptProvider] | None = None) -> None:
        self.providers = providers or [
            FileTranscriptProvider(),
            YouTubeCaptionProvider(),
        ]

    @property
    def name(self) -> str:
        return "composite_transcript_provider"

    def get_transcript(self, video_id: str, **kwargs: object) -> TranscriptDocument | None:
        for provider in self.providers:
            try:
                doc = provider.get_transcript(video_id, **kwargs)
                if doc is not None and doc.segments:
                    logger.info(
                        "Resolved transcript for %s via %s (%d segments)",
                        video_id,
                        provider.name,
                        len(doc.segments),
                    )
                    return doc
            except Exception as e:
                logger.warning("Provider %s error on %s: %s", provider.name, video_id, e)
        return None
