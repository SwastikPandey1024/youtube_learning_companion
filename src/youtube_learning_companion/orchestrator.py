"""Resumable playlist orchestration and batch study note pipeline.

Coordinates:
- Video discovery via YouTubeConnector
- Compliant transcript acquisition via BaseTranscriptProvider
- Evidence-grounded synthesis via StudyNoteSynthesizer
- Markdown and Obsidian artifact export via MarkdownExporter
- Persistent manifest tracking with per-video retry and resumability
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from youtube_learning_companion.config import get_settings
from youtube_learning_companion.connector import YouTubeConnector
from youtube_learning_companion.markdown import MarkdownExporter
from youtube_learning_companion.schemas import (
    NoteGenerationStatus,
    PlaylistProcessingManifest,
    StudyNote,
    TranscriptAcquisitionMethod,
    VideoAccessStatus,
    VideoProcessingRecord,
    YouTubePlaylistMetadata,
    YouTubeVideoMetadata,
)
from youtube_learning_companion.synthesizer import StudyNoteSynthesizer
from youtube_learning_companion.transcript import (
    BaseTranscriptProvider,
    CompositeTranscriptProvider,
)

logger = logging.getLogger(__name__)


class PlaylistJobOrchestrator:
    """Orchestrates end-to-end study note generation for playlists and single videos."""

    def __init__(
        self,
        connector: YouTubeConnector | None = None,
        transcript_provider: BaseTranscriptProvider | None = None,
        synthesizer: StudyNoteSynthesizer | None = None,
        exporter: MarkdownExporter | None = None,
        output_dir: str | Path | None = None,
        max_concurrency: int = 1,
    ) -> None:
        settings = get_settings()
        self.connector = connector or YouTubeConnector()
        self.transcript_provider = transcript_provider or CompositeTranscriptProvider()
        self.synthesizer = synthesizer or StudyNoteSynthesizer()
        self.exporter = exporter or MarkdownExporter()
        self.output_dir = Path(output_dir or settings.notes_export_dir)
        self.max_concurrency = max(1, max_concurrency)

    def _safe_process_single_video(
        self, video_meta: YouTubeVideoMetadata
    ) -> tuple[StudyNote | None, VideoProcessingRecord]:
        """Safely process a single video, catching worker exceptions and forming a FAILED record."""
        try:
            return self.process_single_video(video_meta)
        except Exception as e:
            logger.error("Failed to process video %s: %s", video_meta.video_id, e)
            record = VideoProcessingRecord(
                video_id=video_meta.video_id,
                position=video_meta.position or 0,
                title=video_meta.title,
                source_url=video_meta.source_url,
                access_status=video_meta.access_status,
                status=NoteGenerationStatus.FAILED,
                error_message=str(e),
            )
            return None, record

    def process_playlist(
        self,
        playlist_url_or_id: str,
        limit: int | None = None,
        retry_failed: bool = False,
        single_video_id: str | None = None,
    ) -> PlaylistProcessingManifest:
        """Process playlist into structured notes, indexes, and manifest."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = self.output_dir / "processing_manifest.json"

        # 1. Discover playlist metadata and video entries
        pl_meta = self.connector.get_playlist(playlist_url_or_id)
        logger.info("Discovered playlist '%s' with %d videos.", pl_meta.title, len(pl_meta.videos))

        # 2. Load or initialize manifest
        manifest = self._load_or_create_manifest(manifest_file, pl_meta)

        # 3. Determine videos to process (filtering duplicates)
        target_videos = list(pl_meta.videos)
        if single_video_id:
            target_videos = [v for v in target_videos if v.video_id == single_video_id]
        if limit is not None:
            target_videos = target_videos[:limit]

        records_by_id = {r.video_id: r for r in manifest.records}
        processed_notes: list[StudyNote] = []
        seen_in_run: set[str] = set()
        videos_to_process: list[YouTubeVideoMetadata] = []

        for video_meta in target_videos:
            if video_meta.video_id in seen_in_run:
                logger.warning(
                    "Skipping duplicate video entry in playlist: %s", video_meta.video_id
                )
                continue
            seen_in_run.add(video_meta.video_id)

            existing_record = records_by_id.get(video_meta.video_id)

            # Check if we should skip already completed or skipped
            if existing_record and not retry_failed:
                if existing_record.status in (
                    NoteGenerationStatus.COMPLETE,
                    NoteGenerationStatus.PARTIAL_SUCCESS,
                    NoteGenerationStatus.SKIPPED,
                ):
                    logger.info("Skipping already resolved video: %s", video_meta.video_id)
                    continue

            # Skip inaccessible videos
            if video_meta.access_status in (VideoAccessStatus.PRIVATE, VideoAccessStatus.DELETED):
                logger.info(
                    "Skipping inaccessible video %s: %s",
                    video_meta.video_id,
                    video_meta.access_status,
                )
                rec = VideoProcessingRecord(
                    video_id=video_meta.video_id,
                    position=video_meta.position or 0,
                    title=video_meta.title,
                    source_url=video_meta.source_url,
                    access_status=video_meta.access_status,
                    status=NoteGenerationStatus.SKIPPED,
                    error_message=video_meta.error_message or "Video inaccessible on YouTube",
                )
                records_by_id[video_meta.video_id] = rec
                continue

            videos_to_process.append(video_meta)

        # Execute processing (concurrently if max_concurrency > 1, else sequentially)
        if self.max_concurrency > 1 and len(videos_to_process) > 1:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                futures = {
                    executor.submit(self._safe_process_single_video, vm): vm
                    for vm in videos_to_process
                }
                for future in as_completed(futures):
                    note, record = future.result()
                    records_by_id[record.video_id] = record
                    if note:
                        processed_notes.append(note)
        else:
            for vm in videos_to_process:
                note, record = self._safe_process_single_video(vm)
                records_by_id[record.video_id] = record
                if note:
                    processed_notes.append(note)

        # 4. Authoritative manifest counts derived from records_by_id
        all_records = tuple(records_by_id.values())
        processed_count = sum(
            1
            for r in all_records
            if r.status in (NoteGenerationStatus.COMPLETE, NoteGenerationStatus.PARTIAL_SUCCESS)
        )
        partial_count = sum(
            1 for r in all_records if r.status == NoteGenerationStatus.PARTIAL_SUCCESS
        )
        awaiting_count = sum(
            1 for r in all_records if r.status == NoteGenerationStatus.AWAITING_TRANSCRIPT
        )
        failed_count = sum(1 for r in all_records if r.status == NoteGenerationStatus.FAILED)
        skipped_count = sum(1 for r in all_records if r.status == NoteGenerationStatus.SKIPPED)

        total_unique_discovered = len({v.video_id for v in pl_meta.videos})
        resolved_count = len(all_records)
        pending_count = max(0, total_unique_discovered - resolved_count)

        updated_manifest = manifest.model_copy(
            update={
                "total_videos_discovered": total_unique_discovered,
                "processed_count": processed_count,
                "partial_count": partial_count,
                "awaiting_transcript_count": awaiting_count,
                "failed_count": failed_count,
                "skipped_count": skipped_count,
                "pending_count": pending_count,
                "records": all_records,
                "updated_at": datetime.now(UTC),
            }
        )

        # 5. Persist manifest and generate playlist indexes
        manifest_file.write_text(updated_manifest.model_dump_json(indent=2), encoding="utf-8")

        index_md = self.exporter.generate_playlist_index(updated_manifest, processed_notes)
        (self.output_dir / "00_Playlist_Index.md").write_text(index_md, encoding="utf-8")

        topic_md = self.exporter.generate_topic_index(processed_notes)
        (self.output_dir / "00_Topic_Index.md").write_text(topic_md, encoding="utf-8")

        return updated_manifest

    def process_single_video(
        self, video_meta: YouTubeVideoMetadata
    ) -> tuple[StudyNote, VideoProcessingRecord]:
        """Process a single video metadata object into a StudyNote and write to disk."""
        # 1. Acquire transcript
        transcript_doc = self.transcript_provider.get_transcript(video_meta.video_id)
        acq_method = (
            transcript_doc.acquisition_method
            if transcript_doc
            else TranscriptAcquisitionMethod.NONE
        )

        # 2. Synthesize note
        note = self.synthesizer.synthesize(
            video_meta=video_meta,
            transcript_doc=transcript_doc,
        )

        # 3. Export note markdown
        out_path = self.exporter.write_note_file(note, self.output_dir, overwrite=True)

        # 4. Form processing record
        record = VideoProcessingRecord(
            video_id=video_meta.video_id,
            position=video_meta.position or 0,
            title=video_meta.title,
            source_url=video_meta.source_url,
            access_status=video_meta.access_status,
            status=note.status,
            transcript_method=acq_method,
            note_file_path=str(out_path),
            error_message=None
            if note.status != NoteGenerationStatus.FAILED
            else "Synthesis failure",
        )

        return note, record

    def _load_or_create_manifest(
        self, manifest_file: Path, pl_meta: YouTubePlaylistMetadata
    ) -> PlaylistProcessingManifest:
        """Load existing manifest if present, otherwise initialize a new one."""
        if manifest_file.exists():
            try:
                data = json.loads(manifest_file.read_text(encoding="utf-8"))
                return PlaylistProcessingManifest.model_validate(data)
            except Exception as e:
                logger.warning(
                    "Could not read existing manifest %s: %s. Re-creating.", manifest_file, e
                )

        # Initialize fresh manifest with pending records (deduplicating by video_id)
        seen_ids: set[str] = set()
        records: list[VideoProcessingRecord] = []
        for idx, v in enumerate(pl_meta.videos):
            if v.video_id in seen_ids:
                continue
            seen_ids.add(v.video_id)
            records.append(
                VideoProcessingRecord(
                    video_id=v.video_id,
                    position=v.position or idx,
                    title=v.title,
                    source_url=v.source_url,
                    access_status=v.access_status,
                    status=NoteGenerationStatus.METADATA_ONLY,
                )
            )

        return PlaylistProcessingManifest(
            playlist_id=pl_meta.playlist_id,
            playlist_title=pl_meta.title,
            source_url=pl_meta.source_url,
            total_videos_discovered=len(records),
            pending_count=len(records),
            records=tuple(records),
            output_dir=str(self.output_dir),
        )
