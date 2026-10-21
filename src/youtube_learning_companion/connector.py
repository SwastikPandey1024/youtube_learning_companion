"""Provider-oriented YouTube playlist and video connector.

Provides:
- URL validation, normalization, and ID extraction (video and playlist)
- Official YouTube Data API v3 integration with pagination and quota handling
- Robust error classification (private, deleted, rate-limited, quota exceeded)
- Direct public metadata extraction fallback when credentials are not configured
- Order-preserving video discovery with duplicate detection
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from youtube_learning_companion.config import get_settings
from youtube_learning_companion.schemas import (
    PlaylistAccessStatus,
    VideoAccessStatus,
    YouTubePlaylistMetadata,
    YouTubeVideoMetadata,
)

logger = logging.getLogger(__name__)

# Valid YouTube video ID pattern (11 alphanumeric, underscore, or hyphen chars)
VIDEO_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{11}$")
# Valid YouTube playlist ID pattern (typically starts with PL, UU, FL, RD, etc.)
PLAYLIST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{4,64}$")


class YouTubeURLParser:
    """Robust parser and normalizer for all standard YouTube URL variations."""

    @staticmethod
    def extract_ids(url: str) -> tuple[str | None, str | None]:
        """Extract (video_id, playlist_id) from any supported YouTube URL format.

        Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/playlist?list=PLAYLIST_ID
        - https://www.youtube.com/watch?v=VIDEO_ID&list=PLAYLIST_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://m.youtube.com/watch?v=VIDEO_ID
        - Bare video IDs or playlist IDs directly
        """
        clean_url = url.strip()
        if not clean_url:
            return None, None

        # Check if bare ID was passed
        if VIDEO_ID_REGEX.match(clean_url):
            return clean_url, None
        if clean_url.startswith(("PL", "UU", "FL", "RD")) and PLAYLIST_ID_REGEX.match(clean_url):
            return None, clean_url

        try:
            parsed = urlparse(clean_url)
        except Exception:
            return None, None

        hostname = (parsed.hostname or "").lower()
        path = parsed.path
        query = parse_qs(parsed.query)

        video_id: str | None = None
        playlist_id: str | None = None

        # Check query string for list
        if "list" in query:
            p_val = query["list"][0]
            if PLAYLIST_ID_REGEX.match(p_val):
                playlist_id = p_val

        # youtu.be/VIDEO_ID
        if "youtu.be" in hostname:
            path_parts = [p for p in path.split("/") if p]
            if path_parts:
                candidate = path_parts[0]
                if VIDEO_ID_REGEX.match(candidate):
                    video_id = candidate

        # youtube.com or m.youtube.com
        elif "youtube.com" in hostname or "youtube-nocookie.com" in hostname:
            if "v" in query:
                candidate = query["v"][0]
                if VIDEO_ID_REGEX.match(candidate):
                    video_id = candidate
            elif path.startswith("/embed/"):
                parts = path.split("/embed/")
                if len(parts) > 1:
                    candidate = parts[1].split("/")[0].split("?")[0]
                    if VIDEO_ID_REGEX.match(candidate):
                        video_id = candidate
            elif path.startswith("/shorts/"):
                parts = path.split("/shorts/")
                if len(parts) > 1:
                    candidate = parts[1].split("/")[0].split("?")[0]
                    if VIDEO_ID_REGEX.match(candidate):
                        video_id = candidate
            elif path.startswith("/live/"):
                parts = path.split("/live/")
                if len(parts) > 1:
                    candidate = parts[1].split("/")[0].split("?")[0]
                    if VIDEO_ID_REGEX.match(candidate):
                        video_id = candidate

        return video_id, playlist_id

    @staticmethod
    def normalize_video_url(video_id: str) -> str:
        """Return the canonical watch URL for a video ID."""
        return f"https://www.youtube.com/watch?v={video_id}"

    @staticmethod
    def normalize_playlist_url(playlist_id: str) -> str:
        """Return the canonical playlist URL for a playlist ID."""
        return f"https://www.youtube.com/playlist?list={playlist_id}"


class YouTubeConnectorError(Exception):
    """Base exception for YouTube connector errors."""


class YouTubeQuotaExceededError(YouTubeConnectorError):
    """Raised when YouTube API quota is exceeded."""


class YouTubeRateLimitError(YouTubeConnectorError):
    """Raised when YouTube API rate limits are encountered."""


class YouTubeConnector:
    """Provider-oriented connector to YouTube Data API and playlist discovery."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.youtube_api_key
        self.base_url = (base_url or settings.youtube_api_base_url).rstrip("/")
        self.timeout_seconds = timeout_seconds or settings.youtube_request_timeout_seconds
        self._external_client = client

    def _get_client(self) -> httpx.Client:
        if self._external_client is not None:
            return self._external_client
        return httpx.Client(timeout=self.timeout_seconds, follow_redirects=True)

    def is_api_configured(self) -> bool:
        """Check whether an official YouTube Data API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def get_setup_guidance(self) -> str:
        """Return actionable setup instructions when API key is missing."""
        return (
            "YouTube Data API key is not configured. To use official Google Cloud API:\n"
            "1. Create a project in Google Cloud Console: https://console.cloud.google.com/\n"
            "2. Enable the YouTube Data API v3 for your project.\n"
            "3. Generate an API Key under Credentials.\n"
            "4. Add YOUTUBE_API_KEY=your_key_here to your .env file or environment.\n"
            "VIDEX will automatically fall back to direct public web metadata and local fixtures "
            "when the API key is not supplied."
        )

    # ── Official API Implementation ──────────────────────────────────────────

    def _fetch_playlist_metadata_api(
        self, playlist_id: str, client: httpx.Client
    ) -> tuple[str, str, str | None, str | None, int]:
        """Fetch playlist title, description, and channel details via official API."""
        url = f"{self.base_url}/playlists"
        params = {
            "part": "snippet,contentDetails,status",
            "id": playlist_id,
            "key": self.api_key,
        }
        resp = client.get(url, params=params)
        self._check_api_response_errors(resp)

        data = resp.json()
        items = data.get("items", [])
        if not items:
            msg = f"Playlist {playlist_id} not found on YouTube"
            raise YouTubeConnectorError(msg)

        pl_item = items[0]
        snippet = pl_item.get("snippet", {})
        content_details = pl_item.get("contentDetails", {})
        title = snippet.get("title", f"Playlist {playlist_id}")
        description = snippet.get("description", "")
        channel_id = snippet.get("channelId")
        channel_title = snippet.get("channelTitle")
        item_count = int(content_details.get("itemCount", 0))

        return title, description, channel_id, channel_title, item_count

    def _fetch_playlist_items_api(
        self, playlist_id: str, client: httpx.Client
    ) -> list[YouTubeVideoMetadata]:
        """Paginate through playlist items using official API with deduplication."""
        url = f"{self.base_url}/playlistItems"
        page_token: str | None = None
        discovered_videos: list[YouTubeVideoMetadata] = []
        seen_video_ids: set[str] = set()

        while True:
            params: dict[str, Any] = {
                "part": "snippet,contentDetails,status",
                "playlistId": playlist_id,
                "maxResults": 50,
                "key": self.api_key,
            }
            if page_token:
                params["pageToken"] = page_token

            resp = client.get(url, params=params)
            self._check_api_response_errors(resp)
            data = resp.json()

            items = data.get("items", [])
            for item in items:
                snippet = item.get("snippet", {})
                content_details = item.get("contentDetails", {})
                status_dict = item.get("status", {})

                video_id = content_details.get("videoId") or snippet.get("resourceId", {}).get(
                    "videoId"
                )
                if not video_id:
                    continue

                # Deduplicate entries while preserving first occurrence
                if video_id in seen_video_ids:
                    continue
                seen_video_ids.add(video_id)

                title = snippet.get("title", "")
                privacy_status = status_dict.get("privacyStatus", "public").lower()

                # Classify accessibility
                access_status = VideoAccessStatus.ACCESSIBLE
                err_msg = None
                if title == "Private video" or privacy_status == "private":
                    access_status = VideoAccessStatus.PRIVATE
                    err_msg = "Video is marked as private by uploader"
                elif title == "Deleted video":
                    access_status = VideoAccessStatus.DELETED
                    err_msg = "Video has been deleted from YouTube"

                position = int(snippet.get("position", len(discovered_videos)))
                meta = YouTubeVideoMetadata(
                    video_id=video_id,
                    title=title,
                    source_url=YouTubeURLParser.normalize_video_url(video_id),
                    channel_id=snippet.get("videoOwnerChannelId") or snippet.get("channelId"),
                    channel_title=snippet.get("videoOwnerChannelTitle")
                    or snippet.get("channelTitle"),
                    description=snippet.get("description"),
                    published_at=content_details.get("videoPublishedAt")
                    or snippet.get("publishedAt"),
                    position=position,
                    access_status=access_status,
                    error_message=err_msg,
                )
                discovered_videos.append(meta)

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return discovered_videos

    def _check_api_response_errors(self, resp: httpx.Response) -> None:
        """Inspect HTTP response and raise classified exceptions for YouTube API errors."""
        if resp.status_code == 200:
            return

        status = resp.status_code
        try:
            err_data = resp.json().get("error", {})
            errors = err_data.get("errors", [{}])
            reason = errors[0].get("reason", "")
            message = err_data.get("message", resp.text)
        except Exception:
            reason = ""
            message = resp.text

        if status == 403 and reason in ("quotaExceeded", "dailyLimitExceeded"):
            msg = f"YouTube Data API quota exceeded: {message}"
            raise YouTubeQuotaExceededError(msg)
        if status == 429 or reason == "rateLimitExceeded":
            msg = f"YouTube Data API rate limit exceeded: {message}"
            raise YouTubeRateLimitError(msg)

        msg = f"YouTube API returned HTTP {status} ({reason}): {message}"
        raise YouTubeConnectorError(msg)

    # ── Fallback Public Web Extraction ────────────────────────────────────────

    def _fetch_playlist_web_fallback(
        self, playlist_id: str, client: httpx.Client
    ) -> YouTubePlaylistMetadata:
        """Extract playlist metadata and ordered videos directly from public YouTube webpage."""
        url = YouTubeURLParser.normalize_playlist_url(playlist_id)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = client.get(url, headers=headers)
        if resp.status_code != 200:
            msg = f"Could not fetch public playlist page: HTTP {resp.status_code}"
            raise YouTubeConnectorError(msg)

        html = resp.text
        m = re.search(r"var ytInitialData = ({.*?});</script>", html)
        if not m:
            msg = "Could not parse playlist initial data from YouTube HTML"
            raise YouTubeConnectorError(msg)

        try:
            data = json.loads(m.group(1))
        except Exception as e:
            msg = f"Failed to decode ytInitialData JSON: {e}"
            raise YouTubeConnectorError(msg) from e

        # Extract playlist title
        meta_renderer = data.get("metadata", {}).get("playlistMetadataRenderer", {})
        pl_title = meta_renderer.get("title") or f"Playlist {playlist_id}"
        description = meta_renderer.get("description", "")

        # Extract items
        tabs = data.get("contents", {}).get("twoColumnBrowseResultsRenderer", {}).get("tabs", [])
        tab_content = tabs[0].get("tabRenderer", {}).get("content", {}) if tabs else {}
        section = (
            tab_content.get("sectionListRenderer", {})
            .get("contents", [{}])[0]
            .get("itemSectionRenderer", {})
            .get("contents", [{}])[0]
        )
        pl_renderer = section.get("playlistVideoListRenderer", {})
        raw_items = pl_renderer.get("contents", [])

        discovered_videos: list[YouTubeVideoMetadata] = []
        seen_ids: set[str] = set()

        for item in raw_items:
            pvr = item.get("playlistVideoRenderer")
            if not pvr:
                continue

            vid = pvr.get("videoId")
            if not vid or vid in seen_ids:
                continue
            seen_ids.add(vid)

            title_runs = pvr.get("title", {}).get("runs", [])
            vid_title = title_runs[0].get("text", "") if title_runs else "Untitled Video"

            duration_sec: float | None = None
            length_str = pvr.get("lengthSeconds")
            if length_str and length_str.isdigit():
                duration_sec = float(length_str)

            is_playable = pvr.get("isPlayable", True)
            access_status = (
                VideoAccessStatus.ACCESSIBLE if is_playable else VideoAccessStatus.UNAVAILABLE
            )
            err_msg = None if is_playable else "Video marked as unplayable by YouTube"

            meta = YouTubeVideoMetadata(
                video_id=vid,
                title=vid_title,
                source_url=YouTubeURLParser.normalize_video_url(vid),
                channel_id=None,
                channel_title=pvr.get("shortBylineText", {}).get("runs", [{}])[0].get("text"),
                duration_seconds=duration_sec,
                position=len(discovered_videos),
                access_status=access_status,
                error_message=err_msg,
            )
            discovered_videos.append(meta)

        return YouTubePlaylistMetadata(
            playlist_id=playlist_id,
            title=pl_title,
            source_url=url,
            description=description,
            video_count=len(discovered_videos),
            access_status=PlaylistAccessStatus.ACCESSIBLE,
            videos=tuple(discovered_videos),
        )

    # ── Public High-Level Connector Methods ──────────────────────────────────

    def get_playlist(self, url_or_id: str) -> YouTubePlaylistMetadata:
        """Fetch complete playlist metadata and ordered video list.

        Uses official YouTube Data API v3 if API key is configured.
        Otherwise falls back to public web extraction with a logged notice.
        """
        _, playlist_id = YouTubeURLParser.extract_ids(url_or_id)
        if not playlist_id:
            msg = f"Invalid YouTube playlist URL or ID: {url_or_id}"
            raise YouTubeConnectorError(msg)

        with self._get_client() as client:
            if self.is_api_configured():
                try:
                    title, desc, ch_id, ch_title, _ = self._fetch_playlist_metadata_api(
                        playlist_id, client
                    )
                    videos = self._fetch_playlist_items_api(playlist_id, client)
                    return YouTubePlaylistMetadata(
                        playlist_id=playlist_id,
                        title=title,
                        source_url=YouTubeURLParser.normalize_playlist_url(playlist_id),
                        channel_id=ch_id,
                        channel_title=ch_title,
                        description=desc,
                        video_count=len(videos),
                        access_status=PlaylistAccessStatus.ACCESSIBLE,
                        videos=tuple(videos),
                    )
                except (YouTubeQuotaExceededError, YouTubeRateLimitError):
                    logger.warning(
                        "YouTube API quota/rate limit hit. Falling back to public web extraction."
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to query YouTube API (%s). Falling back to public web extraction.",
                        e,
                    )

            logger.info("Accessing playlist %s via public extraction.", playlist_id)
            return self._fetch_playlist_web_fallback(playlist_id, client)

    def get_video_metadata(self, url_or_id: str) -> YouTubeVideoMetadata:
        """Fetch metadata for a single YouTube video."""
        video_id, _ = YouTubeURLParser.extract_ids(url_or_id)
        if not video_id:
            msg = f"Invalid YouTube video URL or ID: {url_or_id}"
            raise YouTubeConnectorError(msg)

        watch_url = YouTubeURLParser.normalize_video_url(video_id)

        with self._get_client() as client:
            if self.is_api_configured():
                try:
                    url = f"{self.base_url}/videos"
                    params = {
                        "part": "snippet,contentDetails,status",
                        "id": video_id,
                        "key": self.api_key,
                    }
                    resp = client.get(url, params=params)
                    self._check_api_response_errors(resp)
                    items = resp.json().get("items", [])
                    if items:
                        item = items[0]
                        snippet = item.get("snippet", {})
                        status_dict = item.get("status", {})
                        privacy_status = status_dict.get("privacyStatus", "public").lower()
                        access = (
                            VideoAccessStatus.PRIVATE
                            if privacy_status == "private"
                            else VideoAccessStatus.ACCESSIBLE
                        )
                        return YouTubeVideoMetadata(
                            video_id=video_id,
                            title=snippet.get("title", f"Video {video_id}"),
                            source_url=watch_url,
                            channel_id=snippet.get("channelId"),
                            channel_title=snippet.get("channelTitle"),
                            description=snippet.get("description"),
                            published_at=snippet.get("publishedAt"),
                            access_status=access,
                        )
                except Exception as e:
                    logger.warning("API single video fetch failed: %s. Using public watch page.", e)

            # Public watch page fallback
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
            resp = client.get(watch_url, headers=headers)
            if resp.status_code != 200:
                return YouTubeVideoMetadata(
                    video_id=video_id,
                    title=f"Video {video_id}",
                    source_url=watch_url,
                    access_status=VideoAccessStatus.UNAVAILABLE,
                    error_message=f"HTTP {resp.status_code}",
                )

            m = re.search(r"var ytInitialPlayerResponse = ({.*?});</script>", resp.text)
            if m:
                try:
                    player_data = json.loads(m.group(1))
                    vd = player_data.get("videoDetails", {})
                    dur_sec = (
                        float(vd.get("lengthSeconds"))
                        if vd.get("lengthSeconds", "").isdigit()
                        else None
                    )
                    return YouTubeVideoMetadata(
                        video_id=video_id,
                        title=vd.get("title", f"Video {video_id}"),
                        source_url=watch_url,
                        channel_id=vd.get("channelId"),
                        channel_title=vd.get("author"),
                        duration_seconds=dur_sec,
                        description=vd.get("shortDescription"),
                        access_status=VideoAccessStatus.ACCESSIBLE,
                    )
                except Exception as e:
                    logger.debug("Could not parse player details: %s", e)

            return YouTubeVideoMetadata(
                video_id=video_id,
                title=f"Video {video_id}",
                source_url=watch_url,
                access_status=VideoAccessStatus.ACCESSIBLE,
            )
