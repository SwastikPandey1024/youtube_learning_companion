"""Playlist metadata inventory and per-video evidence readiness reporting."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from youtube_learning_companion.connector import (
    YouTubeConnector,
    YouTubeURLParser,
)
from youtube_learning_companion.schemas import TranscriptCoverageStatus
from youtube_learning_companion.transcript import FileTranscriptProvider
from youtube_learning_companion.transcript_audit import audit_transcript


class MetadataExportVideo(BaseModel):
    """One video record from an authorized metadata export."""

    model_config = ConfigDict(extra="ignore")

    position: int = Field(ge=0)
    video_id: str | None = None
    title: str = ""
    duration_seconds: float | None = Field(default=None, ge=0)
    metadata_status: str = "available"
    failure_category: str | None = None


class MetadataExport(BaseModel):
    """Portable, credential-free metadata export accepted by the inventory CLI."""

    model_config = ConfigDict(extra="ignore")

    playlist_id: str
    title: str = ""
    declared_item_count: int | None = Field(default=None, ge=0)
    pagination_complete: bool
    source_method: Literal[
        "authorized_metadata_export",
        "youtube_public_playlist_page",
        "youtube_data_api_v3",
    ] = "authorized_metadata_export"
    videos: list[MetadataExportVideo]


class _VideoRecord(BaseModel):
    position: int
    video_id: str | None
    title: str
    duration_seconds: float | None = None
    metadata_status: str = "available"
    access_status: str = "unknown"
    failure_category: str | None = None


class PlaylistInventoryService:
    """Build a credential-safe inventory from the official API or metadata export."""

    def __init__(
        self,
        connector: YouTubeConnector,
        transcript_dirs: list[str | Path] | None = None,
        *,
        retries: int = 2,
        backoff_seconds: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.connector = connector
        self.transcript_dirs = [Path(path) for path in (transcript_dirs or [])]
        self.retries = max(0, retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self._sleep = sleep

    def build_from_api(self, playlist_url_or_id: str) -> dict[str, Any]:
        """Fetch playlist items and video details; preserve partial results on failure."""
        _, playlist_id = YouTubeURLParser.extract_ids(playlist_url_or_id)
        if playlist_id is None:
            return self._report(
                playlist_id="",
                title="",
                records=[],
                source_method="youtube_data_api_v3",
                enumeration_status="failed",
                pagination_complete=False,
                declared_item_count=None,
                failure_category="invalid_playlist_id",
            )
        if not self.connector.is_api_configured():
            return self._report(
                playlist_id=playlist_id,
                title="",
                records=[],
                source_method="youtube_data_api_v3",
                enumeration_status="failed",
                pagination_complete=False,
                declared_item_count=None,
                failure_category="api_key_not_configured",
            )

        title = ""
        declared_count: int | None = None
        records: list[_VideoRecord] = []
        pagination_complete = False
        failure_category: str | None = None
        enumeration_status = "failed"

        try:
            with self.connector._get_client() as client:
                playlist_data = self._request_json(
                    client,
                    "playlists",
                    {
                        "part": "snippet,contentDetails,status",
                        "id": playlist_id,
                    },
                )
                playlist_items = playlist_data.get("items", [])
                if not playlist_items:
                    failure_category = "playlist_not_found_or_not_accessible"
                    return self._report(
                        playlist_id=playlist_id,
                        title=title,
                        records=records,
                        source_method="youtube_data_api_v3",
                        enumeration_status="failed",
                        pagination_complete=False,
                        declared_item_count=declared_count,
                        failure_category=failure_category,
                    )

                playlist_item = playlist_items[0]
                snippet = playlist_item.get("snippet", {})
                content = playlist_item.get("contentDetails", {})
                title = str(snippet.get("title") or "")
                raw_count = content.get("itemCount")
                declared_count = int(raw_count) if raw_count is not None else None

                page_token: str | None = None
                while True:
                    params: dict[str, Any] = {
                        "part": "snippet,contentDetails,status",
                        "playlistId": playlist_id,
                        "maxResults": 50,
                    }
                    if page_token:
                        params["pageToken"] = page_token
                    page_data = self._request_json(client, "playlistItems", params)
                    items = page_data.get("items", [])
                    for item in items:
                        item_snippet = item.get("snippet", {})
                        item_content = item.get("contentDetails", {})
                        item_status = item.get("status", {})
                        resource = item_snippet.get("resourceId", {})
                        video_id = item_content.get("videoId") or resource.get("videoId")
                        position = item_snippet.get("position")
                        if position is None:
                            position = len(records)
                        item_title = str(item_snippet.get("title") or "")
                        access_status = "unknown"
                        if item_title.casefold() == "private video":
                            access_status = "private"
                        elif item_title.casefold() == "deleted video":
                            access_status = "deleted"
                        elif str(item_status.get("privacyStatus", "")).casefold() == "private":
                            access_status = "private"
                        elif video_id:
                            access_status = "listed"
                        records.append(
                            _VideoRecord(
                                position=int(position),
                                video_id=str(video_id) if video_id else None,
                                title=item_title,
                                access_status=access_status,
                                metadata_status="pending" if video_id else "unavailable",
                                failure_category=None if video_id else "missing_video_id",
                            )
                        )
                    page_token = page_data.get("nextPageToken")
                    if not page_token:
                        pagination_complete = True
                        enumeration_status = "complete"
                        break
        except _InventoryRequestError as error:
            failure_category = error.category
            enumeration_status = "partial" if records else "failed"

        if records:
            self._enrich_video_details(records)

        return self._report(
            playlist_id=playlist_id,
            title=title,
            records=records,
            source_method="youtube_data_api_v3",
            enumeration_status=enumeration_status,
            pagination_complete=pagination_complete,
            declared_item_count=declared_count,
            failure_category=failure_category,
        )

    def build_from_metadata_export(self, export_path: str | Path) -> dict[str, Any]:
        """Build a report from the documented credential-free export format."""
        path = Path(export_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            export = MetadataExport.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError, TypeError, ValueError):
            return self._report(
                playlist_id="",
                title="",
                records=[],
                source_method="authorized_metadata_export",
                enumeration_status="failed",
                pagination_complete=False,
                declared_item_count=None,
                failure_category="invalid_metadata_export",
            )

        records = [
            _VideoRecord(
                position=video.position,
                video_id=video.video_id,
                title=video.title,
                duration_seconds=video.duration_seconds,
                metadata_status=video.metadata_status,
                access_status="unknown",
                failure_category=video.failure_category,
            )
            for video in export.videos
        ]
        enumeration_status = "complete" if export.pagination_complete else "partial"
        if any(record.metadata_status != "available" for record in records):
            enumeration_status = "partial"
        return self._report(
            playlist_id=export.playlist_id,
            title=export.title,
            records=records,
            source_method=export.source_method,
            enumeration_status=enumeration_status,
            pagination_complete=export.pagination_complete,
            declared_item_count=export.declared_item_count,
            failure_category=None if export.pagination_complete else "export_incomplete",
        )

    def _request_json(
        self, client: httpx.Client, resource: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        url = f"{self.connector.base_url}/{resource}"
        for attempt in range(self.retries + 1):
            try:
                response = client.get(
                    url,
                    params={**params, "key": self.connector.api_key},
                )
            except httpx.TimeoutException as error:
                if attempt < self.retries:
                    self._sleep(self.backoff_seconds * (2**attempt))
                    continue
                raise _InventoryRequestError("timeout") from error
            except httpx.NetworkError as error:
                if attempt < self.retries:
                    self._sleep(self.backoff_seconds * (2**attempt))
                    continue
                raise _InventoryRequestError("network_error") from error
            except httpx.RequestError as error:
                if attempt < self.retries:
                    self._sleep(self.backoff_seconds * (2**attempt))
                    continue
                raise _InventoryRequestError("request_error") from error

            category = self._response_failure_category(response)
            retryable = (
                response.status_code == 429
                or response.status_code >= 500
                or category == "rate_limited"
            )
            if category and retryable and attempt < self.retries:
                self._sleep(self.backoff_seconds * (2**attempt))
                continue
            if category:
                raise _InventoryRequestError(category)
            try:
                data = response.json()
            except (ValueError, json.JSONDecodeError) as error:
                raise _InventoryRequestError("invalid_json_response") from error
            if not isinstance(data, dict):
                raise _InventoryRequestError("invalid_response_shape")
            return data
        raise _InventoryRequestError("request_failed")

    @staticmethod
    def _response_failure_category(response: httpx.Response) -> str | None:
        if response.status_code == 200:
            return None
        try:
            error_data = response.json().get("error", {})
            errors = error_data.get("errors", [])
            reason = errors[0].get("reason", "") if errors else ""
        except (AttributeError, IndexError, TypeError, ValueError):
            reason = ""
        if reason in {"quotaExceeded", "dailyLimitExceeded"}:
            return "quota_exceeded"
        if response.status_code == 429 or reason in {
            "rateLimitExceeded",
            "userRateLimitExceeded",
        }:
            return "rate_limited"
        if response.status_code == 401 or reason == "keyInvalid":
            return "authentication_error"
        if reason == "accessNotConfigured":
            return "api_not_enabled"
        if reason in {"ipRefererBlocked", "ipAddressBlocked"}:
            return "api_key_restricted"
        if response.status_code == 403:
            return "api_forbidden"
        if response.status_code >= 500:
            return "api_server_error"
        return "api_request_rejected"

    @staticmethod
    def _parse_duration(value: str) -> float | None:
        match = re.fullmatch(
            r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?"
            r"(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?",
            value,
        )
        if not match:
            return None
        values = {key: float(value or 0) for key, value in match.groupdict().items()}
        return (
            values["days"] * 86400
            + values["hours"] * 3600
            + values["minutes"] * 60
            + values["seconds"]
        )

    def _enrich_video_details(self, records: list[_VideoRecord]) -> None:
        ids = list(dict.fromkeys(record.video_id for record in records if record.video_id))
        if not ids:
            return
        by_id: dict[str, dict[str, Any]] = {}
        for start in range(0, len(ids), 50):
            batch = ids[start : start + 50]
            try:
                with self.connector._get_client() as client:
                    data = self._request_json(
                        client,
                        "videos",
                        {"part": "snippet,contentDetails,status", "id": ",".join(batch)},
                    )
            except _InventoryRequestError as error:
                for record in records:
                    if record.video_id in batch and record.metadata_status == "pending":
                        record.metadata_status = "partial"
                        record.failure_category = error.category
                continue
            for item in data.get("items", []):
                video_id = item.get("id")
                if video_id:
                    by_id[str(video_id)] = item

        for record in records:
            if not record.video_id or record.metadata_status != "pending":
                continue
            item = by_id.get(record.video_id)
            if item is None:
                record.metadata_status = "unavailable"
                record.access_status = (
                    record.access_status
                    if record.access_status in {"private", "deleted"}
                    else "unavailable_or_inaccessible"
                )
                record.failure_category = "video_details_not_returned"
                continue
            snippet = item.get("snippet", {})
            content = item.get("contentDetails", {})
            status = item.get("status", {})
            record.title = str(snippet.get("title") or record.title)
            record.duration_seconds = self._parse_duration(
                str(content.get("duration") or "")
            )
            privacy = str(status.get("privacyStatus", "")).casefold()
            record.access_status = "private" if privacy == "private" else "accessible"
            if record.duration_seconds is None:
                record.metadata_status = "partial"
                record.failure_category = "duration_unavailable"
            else:
                record.metadata_status = "available"

    def _readiness(self, record: _VideoRecord) -> dict[str, Any]:
        result: dict[str, Any] = {
            "metadata_availability": record.metadata_status,
            "local_transcript_availability": "not_checked",
            "transcript_parse_status": "not_run",
            "source_timestamp_provenance": "unknown",
            "source_start_timestamp_count": 0,
            "inferred_start_timestamp_count": 0,
            "source_end_timestamp_count": 0,
            "inferred_end_timestamp_count": 0,
            "transcript_completeness": {
                "status": "unknown",
                "rationale": None,
                "audit_can_support_full_lecture_notes": False,
            },
            "audio_processing_eligibility": "blocked_no_authorized_media_attestation",
            "visual_processing_eligibility": "blocked_no_authorized_media_attestation",
            "study_note_generation_readiness": "blocked_no_usable_transcript",
        }
        if not record.video_id:
            result["local_transcript_availability"] = "not_matchable_missing_video_id"
            return result

        matching_paths = sorted(
            {
                path
                for directory in self.transcript_dirs
                if directory.is_dir()
                for path in directory.iterdir()
                if path.is_file()
                and path.stem == record.video_id
                and path.suffix.casefold() in {".json", ".vtt", ".srt", ".txt"}
            }
        )
        if not matching_paths:
            result["local_transcript_availability"] = "not_found"
            return result
        result["local_transcript_availability"] = "found"
        document = None
        for path in matching_paths:
            document = FileTranscriptProvider(
                file_map={record.video_id: path}
            ).get_transcript(record.video_id)
            if document is not None:
                break
        if document is None:
            result["transcript_parse_status"] = "failed"
            result["study_note_generation_readiness"] = "blocked_transcript_parse_failed"
            return result

        result["transcript_parse_status"] = "parsed"
        source_starts = sum(segment.start_timestamp_is_source for segment in document.segments)
        source_ends = sum(segment.end_timestamp_is_source for segment in document.segments)
        result["source_start_timestamp_count"] = source_starts
        result["inferred_start_timestamp_count"] = len(document.segments) - source_starts
        result["source_end_timestamp_count"] = source_ends
        result["inferred_end_timestamp_count"] = len(document.segments) - source_ends
        if source_starts == len(document.segments) and source_ends == len(document.segments):
            result["source_timestamp_provenance"] = "all_endpoints_source_provided"
        elif source_starts or source_ends:
            result["source_timestamp_provenance"] = "mixed_source_and_inferred"
        else:
            result["source_timestamp_provenance"] = "all_endpoints_inferred"

        audit = audit_transcript(
            document,
            video_duration=record.duration_seconds,
            video_title=record.title,
        )
        coverage_status = document.coverage_status
        if coverage_status == TranscriptCoverageStatus.COMPLETE:
            completeness_status = "source_declared_complete"
        elif coverage_status == TranscriptCoverageStatus.PARTIAL:
            completeness_status = "source_declared_partial"
        else:
            completeness_status = "unknown"
        result["transcript_completeness"] = {
            "status": completeness_status,
            "rationale": document.coverage_rationale,
            "audit_can_support_full_lecture_notes": audit.can_support_full_lecture_notes,
            "segment_count": audit.segment_count,
            "word_count": audit.total_words,
            "transcript_span_seconds": audit.transcript_span_seconds,
            "audit_findings": list(audit.findings),
        }
        if audit.can_support_full_lecture_notes and record.duration_seconds is not None:
            result["study_note_generation_readiness"] = "ready_for_full_lecture_notes"
        elif audit.can_support_topic_synopsis:
            result["study_note_generation_readiness"] = "limited_topic_synopsis_only"
        else:
            result["study_note_generation_readiness"] = "blocked_transcript_ineligible"
        return result

    def _report(
        self,
        *,
        playlist_id: str,
        title: str,
        records: list[_VideoRecord],
        source_method: str,
        enumeration_status: str,
        pagination_complete: bool,
        declared_item_count: int | None,
        failure_category: str | None,
    ) -> dict[str, Any]:
        ordered = sorted(records, key=lambda record: record.position)
        seen: set[str] = set()
        duplicate_count = 0
        videos: list[dict[str, Any]] = []
        for record in ordered:
            if record.video_id:
                if record.video_id in seen:
                    duplicate_count += 1
                seen.add(record.video_id)
            videos.append(
                {
                    "position": record.position,
                    "video_id": record.video_id,
                    "title": record.title,
                    "duration_seconds": record.duration_seconds,
                    "metadata_status": record.metadata_status,
                    "access_status": record.access_status,
                    "failure_category": record.failure_category,
                    "readiness": self._readiness(record),
                }
            )
        readiness_summary = {
            field: dict(
                Counter(
                    str(video["readiness"][field])
                    for video in videos
                    if video["readiness"].get(field) is not None
                )
            )
            for field in (
                "local_transcript_availability",
                "transcript_parse_status",
                "source_timestamp_provenance",
                "audio_processing_eligibility",
                "visual_processing_eligibility",
                "study_note_generation_readiness",
            )
        }
        completeness_statuses = Counter(
            str(video["readiness"]["transcript_completeness"]["status"])
            for video in videos
        )
        readiness_summary["transcript_completeness"] = dict(completeness_statuses)
        has_metadata_gaps = any(
            record.metadata_status != "available" for record in records
        )
        inventory_status = (
            "failed"
            if enumeration_status == "failed"
            else "partial"
            if enumeration_status == "partial" or has_metadata_gaps
            else "complete"
        )
        return {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "source_method": source_method,
            "playlist_id": playlist_id,
            "playlist_title": title,
            "enumeration_status": enumeration_status,
            "inventory_status": inventory_status,
            "pagination_complete": pagination_complete,
            "declared_item_count": declared_item_count,
            "enumerated_item_count": len(records),
            "unique_video_id_count": len(seen),
            "duplicate_item_count": duplicate_count,
            "failure_category": failure_category,
            "readiness_summary": readiness_summary,
            "videos": videos,
        }


class _InventoryRequestError(Exception):
    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)
