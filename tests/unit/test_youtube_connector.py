"""Unit tests for YouTube URL parser and YouTube connector."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from youtube_learning_companion.connector import (
    YouTubeConnector,
    YouTubeQuotaExceededError,
    YouTubeRateLimitError,
    YouTubeURLParser,
)
from youtube_learning_companion.schemas import VideoAccessStatus


class TestYouTubeURLParser:
    """Test URL parsing, ID extraction, and normalization across all supported formats."""

    def test_standard_watch_url(self):
        url = "https://www.youtube.com/watch?v=DrCeIbpfuzE"
        vid, pl = YouTubeURLParser.extract_ids(url)
        assert vid == "DrCeIbpfuzE"
        assert pl is None

    def test_short_url(self):
        url = "https://youtu.be/DrCeIbpfuzE"
        vid, pl = YouTubeURLParser.extract_ids(url)
        assert vid == "DrCeIbpfuzE"
        assert pl is None

    def test_playlist_url(self):
        url = "https://youtube.com/playlist?list=PLgjejdknTfWP9cYIHjxBRRdtGcUBLmCQC"
        vid, pl = YouTubeURLParser.extract_ids(url)
        assert vid is None
        assert pl == "PLgjejdknTfWP9cYIHjxBRRdtGcUBLmCQC"

    def test_video_in_playlist_url(self):
        url = "https://www.youtube.com/watch?v=DrCeIbpfuzE&list=PLgjejdknTfWP9cYIHjxBRRdtGcUBLmCQC&index=1"
        vid, pl = YouTubeURLParser.extract_ids(url)
        assert vid == "DrCeIbpfuzE"
        assert pl == "PLgjejdknTfWP9cYIHjxBRRdtGcUBLmCQC"

    def test_embed_url(self):
        url = "https://www.youtube.com/embed/DrCeIbpfuzE?autoplay=1"
        vid, pl = YouTubeURLParser.extract_ids(url)
        assert vid == "DrCeIbpfuzE"
        assert pl is None

    def test_shorts_url(self):
        url = "https://www.youtube.com/shorts/DrCeIbpfuzE"
        vid, pl = YouTubeURLParser.extract_ids(url)
        assert vid == "DrCeIbpfuzE"
        assert pl is None

    def test_bare_video_id(self):
        vid, pl = YouTubeURLParser.extract_ids("DrCeIbpfuzE")
        assert vid == "DrCeIbpfuzE"
        assert pl is None

    def test_bare_playlist_id(self):
        vid, pl = YouTubeURLParser.extract_ids("PLgjejdknTfWP9cYIHjxBRRdtGcUBLmCQC")
        assert vid is None
        assert pl == "PLgjejdknTfWP9cYIHjxBRRdtGcUBLmCQC"

    def test_malformed_url(self):
        vid, pl = YouTubeURLParser.extract_ids("not_a_valid_youtube_url")
        assert vid is None
        assert pl is None

    def test_normalize_video_url(self):
        assert (
            YouTubeURLParser.normalize_video_url("DrCeIbpfuzE")
            == "https://www.youtube.com/watch?v=DrCeIbpfuzE"
        )

    def test_normalize_playlist_url(self):
        assert (
            YouTubeURLParser.normalize_playlist_url("PLgjejdknTfWP9cYIHjxBRRdtGcUBLmCQC")
            == "https://www.youtube.com/playlist?list=PLgjejdknTfWP9cYIHjxBRRdtGcUBLmCQC"
        )


class TestYouTubeConnectorAPI:
    """Test YouTubeConnector official API operations and error handling."""

    def test_setup_guidance_when_key_missing(self):
        connector = YouTubeConnector(api_key="")
        assert not connector.is_api_configured()
        guidance = connector.get_setup_guidance()
        assert "YOUTUBE_API_KEY" in guidance
        assert "Google Cloud Console" in guidance

    def test_api_pagination_and_deduplication(self):
        """Verify multi-page discovery, pagination with nextPageToken, and duplicate removal."""
        connector = YouTubeConnector(api_key="mock_key")

        mock_client = MagicMock(spec=httpx.Client)

        # Mock playlist metadata response
        pl_meta_resp = MagicMock(spec=httpx.Response)
        pl_meta_resp.status_code = 200
        pl_meta_resp.json.return_value = {
            "items": [
                {
                    "snippet": {
                        "title": "GATE Linear Algebra",
                        "description": "Complete Course",
                        "channelTitle": "GO Classes",
                    },
                    "contentDetails": {"itemCount": 3},
                }
            ]
        }

        # Mock page 1 response (2 items, 1 duplicate of page 2)
        page1_resp = MagicMock(spec=httpx.Response)
        page1_resp.status_code = 200
        page1_resp.json.return_value = {
            "nextPageToken": "token_page_2",
            "items": [
                {
                    "snippet": {"title": "Lecture 1", "position": 0},
                    "contentDetails": {"videoId": "vid_001"},
                    "status": {"privacyStatus": "public"},
                },
                {
                    "snippet": {"title": "Lecture 2", "position": 1},
                    "contentDetails": {"videoId": "vid_002"},
                    "status": {"privacyStatus": "public"},
                },
            ],
        }

        # Mock page 2 response (contains duplicate vid_002 and a private video)
        page2_resp = MagicMock(spec=httpx.Response)
        page2_resp.status_code = 200
        page2_resp.json.return_value = {
            "items": [
                {
                    "snippet": {"title": "Lecture 2 Duplicate", "position": 2},
                    "contentDetails": {"videoId": "vid_002"},
                    "status": {"privacyStatus": "public"},
                },
                {
                    "snippet": {"title": "Private video", "position": 3},
                    "contentDetails": {"videoId": "vid_003"},
                    "status": {"privacyStatus": "private"},
                },
            ]
        }

        mock_client.get.side_effect = [pl_meta_resp, page1_resp, page2_resp]
        mock_client.__enter__.return_value = mock_client

        with patch.object(connector, "_get_client", return_value=mock_client):
            playlist = connector.get_playlist("PLgjejdknTfWP9cYIHjxBRRdtGcUBLmCQC")

        assert playlist.title == "GATE Linear Algebra"
        # Total unique videos: vid_001, vid_002, vid_003 (deduplicated)
        assert len(playlist.videos) == 3
        assert playlist.videos[0].video_id == "vid_001"
        assert playlist.videos[0].access_status == VideoAccessStatus.ACCESSIBLE
        assert playlist.videos[1].video_id == "vid_002"
        assert playlist.videos[2].video_id == "vid_003"
        assert playlist.videos[2].access_status == VideoAccessStatus.PRIVATE

    def test_quota_exceeded_error_handling(self):
        """Verify that quotaExceeded raises YouTubeQuotaExceededError."""
        connector = YouTubeConnector(api_key="mock_key")
        mock_client = MagicMock(spec=httpx.Client)

        quota_resp = MagicMock(spec=httpx.Response)
        quota_resp.status_code = 403
        quota_resp.json.return_value = {
            "error": {
                "errors": [{"reason": "quotaExceeded"}],
                "message": "The request cannot be completed because you have exceeded your quota.",
            }
        }
        mock_client.get.return_value = quota_resp

        with pytest.raises(YouTubeQuotaExceededError):
            connector._fetch_playlist_metadata_api("PLtest", mock_client)

    def test_rate_limit_error_handling(self):
        """Verify that HTTP 429 raises YouTubeRateLimitError."""
        connector = YouTubeConnector(api_key="mock_key")
        mock_client = MagicMock(spec=httpx.Client)

        rate_resp = MagicMock(spec=httpx.Response)
        rate_resp.status_code = 429
        rate_resp.text = "Too Many Requests"
        rate_resp.json.side_effect = Exception("Not JSON")
        mock_client.get.return_value = rate_resp

        with pytest.raises(YouTubeRateLimitError):
            connector._fetch_playlist_metadata_api("PLtest", mock_client)
