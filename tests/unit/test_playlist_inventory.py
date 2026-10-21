"""Tests for reliable playlist inventory and evidence-readiness reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from youtube_learning_companion.connector import YouTubeConnector
from youtube_learning_companion.playlist_inventory import PlaylistInventoryService


class _QueueClient:
    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> _QueueClient:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def get(self, url: str, *, params: dict[str, Any]) -> httpx.Response:
        self.calls.append((url, params))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _response(status: int, payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        request=httpx.Request("GET", "https://example.test"),
    )


def _playlist_response() -> httpx.Response:
    return _response(
        200,
        {
            "items": [
                {
                    "snippet": {"title": "Linear Algebra"},
                    "contentDetails": {"itemCount": 4},
                }
            ]
        },
    )


def _service(
    tmp_path: Path,
    client: _QueueClient,
    *,
    retries: int = 0,
) -> PlaylistInventoryService:
    connector = YouTubeConnector(api_key="test-secret-key", client=client)
    return PlaylistInventoryService(
        connector,
        transcript_dirs=[tmp_path],
        retries=retries,
        backoff_seconds=0,
        sleep=lambda _: None,
    )


def test_api_inventory_paginates_enriches_durations_and_preserves_duplicate_items(
    tmp_path: Path,
) -> None:
    page_one = _response(
        200,
        {
            "nextPageToken": "next",
            "items": [
                {
                    "snippet": {"position": 0, "title": "Lecture A"},
                    "contentDetails": {"videoId": "video_a"},
                },
                {
                    "snippet": {"position": 1, "title": "Lecture B"},
                    "contentDetails": {"videoId": "video_b"},
                },
            ],
        },
    )
    page_two = _response(
        200,
        {
            "items": [
                {
                    "snippet": {"position": 2, "title": "Lecture B repeat"},
                    "contentDetails": {"videoId": "video_b"},
                },
                {
                    "snippet": {
                        "position": 3,
                        "title": "Private video",
                        "resourceId": {"videoId": "video_c"},
                    },
                    "contentDetails": {},
                },
            ]
        },
    )
    details = _response(
        200,
        {
            "items": [
                {
                    "id": "video_a",
                    "snippet": {"title": "Lecture A"},
                    "contentDetails": {"duration": "PT1H2M3S"},
                    "status": {"privacyStatus": "public"},
                },
                {
                    "id": "video_b",
                    "snippet": {"title": "Lecture B"},
                    "contentDetails": {"duration": "PT5M"},
                    "status": {"privacyStatus": "public"},
                },
            ]
        },
    )
    client = _QueueClient([_playlist_response(), page_one, page_two, details])
    service = _service(tmp_path, client)

    report = service.build_from_api("PLplaylist123")

    assert report["enumeration_status"] == "complete"
    assert report["inventory_status"] == "partial"
    assert report["pagination_complete"] is True
    assert report["declared_item_count"] == 4
    assert report["enumerated_item_count"] == 4
    assert report["unique_video_id_count"] == 3
    assert report["duplicate_item_count"] == 1
    rows = report["videos"]
    assert [row["position"] for row in rows] == [0, 1, 2, 3]
    assert [row["video_id"] for row in rows] == [
        "video_a",
        "video_b",
        "video_b",
        "video_c",
    ]
    assert rows[0]["duration_seconds"] == 3723
    assert rows[1]["duration_seconds"] == rows[2]["duration_seconds"] == 300
    assert rows[3]["access_status"] == "private"
    assert rows[3]["metadata_status"] == "unavailable"
    assert [call[1].get("pageToken") for call in client.calls if "pageToken" in call[1]] == [
        "next"
    ]
    assert "test-secret-key" not in json.dumps(report)


def test_incomplete_pagination_retains_partial_items_and_reports_incomplete_status(
    tmp_path: Path,
) -> None:
    page_one = _response(
        200,
        {
            "nextPageToken": "next",
            "items": [
                {
                    "snippet": {"position": 0, "title": "Lecture"},
                    "contentDetails": {"videoId": "video_a"},
                }
            ],
        },
    )
    details = _response(200, {"items": []})
    client = _QueueClient(
        [
            httpx.ReadTimeout("temporary"),
            _playlist_response(),
            page_one,
            httpx.ReadTimeout("timed out"),
            httpx.ReadTimeout("timed out"),
            details,
        ]
    )
    service = _service(tmp_path, client, retries=1)

    report = service.build_from_api("PLplaylist123")

    assert report["enumeration_status"] == "partial"
    assert report["pagination_complete"] is False
    assert report["failure_category"] == "timeout"
    assert report["enumerated_item_count"] == 1
    assert len(client.calls) == 6


def test_video_metadata_enrichment_uses_api_batches_of_at_most_fifty(
    tmp_path: Path,
) -> None:
    playlist_items = [
        {
            "snippet": {"position": position, "title": f"Lecture {position}"},
            "contentDetails": {"videoId": f"video_{position:03}"},
        }
        for position in range(51)
    ]
    detail_batches = [
        _response(
            200,
            {
                "items": [
                    {
                        "id": f"video_{position:03}",
                        "snippet": {"title": f"Lecture {position}"},
                        "contentDetails": {"duration": "PT1M"},
                        "status": {"privacyStatus": "public"},
                    }
                    for position in batch
                ]
            },
        )
        for batch in (range(50), range(50, 51))
    ]
    client = _QueueClient(
        [
            _playlist_response(),
            _response(200, {"items": playlist_items}),
            *detail_batches,
        ]
    )

    report = _service(tmp_path, client).build_from_api("PLplaylist123")

    videos_calls = [
        params for url, params in client.calls if url.endswith("/videos") and "id" in params
    ]
    assert [len(params["id"].split(",")) for params in videos_calls] == [50, 1]
    assert report["enumeration_status"] == "complete"
    assert report["inventory_status"] == "complete"
    assert report["enumerated_item_count"] == 51
    assert [video["position"] for video in report["videos"]] == list(range(51))


def test_transient_timeout_retries_but_quota_is_not_retried_or_hidden(
    tmp_path: Path,
) -> None:
    success = _response(
        200,
        {"items": [{"snippet": {"title": "Linear Algebra"}, "contentDetails": {}}]},
    )
    quota_client = _QueueClient(
        [
            _response(
                403,
                {
                    "error": {
                        "errors": [{"reason": "quotaExceeded"}],
                        "message": "sensitive text test-secret-key",
                    }
                },
            )
        ]
    )
    quota_report = _service(tmp_path, quota_client, retries=3).build_from_api(
        "PLplaylist123"
    )
    assert quota_report["failure_category"] == "quota_exceeded"
    assert len(quota_client.calls) == 1
    assert "test-secret-key" not in json.dumps(quota_report)
    assert "sensitive text" not in json.dumps(quota_report)

    forbidden_client = _QueueClient(
        [
            _response(
                403,
                {"error": {"errors": [{"reason": "forbidden"}], "message": "Forbidden"}},
            )
        ]
    )
    forbidden_report = _service(tmp_path, forbidden_client).build_from_api(
        "PLplaylist123"
    )
    assert forbidden_report["failure_category"] == "api_forbidden"

    client = _QueueClient(
        [
            httpx.ReadTimeout("temporary"),
            success,
            _response(200, {"items": []}),
        ]
    )
    service = _service(tmp_path, client, retries=1)
    report = service.build_from_api("PLplaylist123")
    assert report["enumeration_status"] == "complete"
    assert len(client.calls) == 3

    rate_limited = _response(
        403,
        {
            "error": {
                "errors": [{"reason": "userRateLimitExceeded"}],
                "message": "retry later",
            }
        },
    )
    rate_client = _QueueClient(
        [rate_limited, success, _response(200, {"items": []})]
    )
    rate_report = _service(tmp_path, rate_client, retries=1).build_from_api(
        "PLplaylist123"
    )
    assert rate_report["enumeration_status"] == "complete"
    assert len(rate_client.calls) == 3


def test_google_error_categories_distinguish_auth_configuration_quota_rate_and_unknown() -> None:
    cases = [
        (
            401,
            {"status": "UNAUTHENTICATED", "errors": [{"reason": "required"}]},
            "authentication_error",
        ),
        (
            400,
            {"status": "INVALID_ARGUMENT", "errors": [{"reason": "keyInvalid"}]},
            "authentication_error",
        ),
        (
            403,
            {"status": "PERMISSION_DENIED", "errors": [{"reason": "accessNotConfigured"}]},
            "api_not_enabled",
        ),
        (
            403,
            {"status": "PERMISSION_DENIED", "errors": [{"reason": "ipRefererBlocked"}]},
            "api_key_restricted",
        ),
        (
            403,
            {"status": "RESOURCE_EXHAUSTED", "errors": [{"reason": "quotaExceeded"}]},
            "quota_exceeded",
        ),
        (
            403,
            {"status": "RESOURCE_EXHAUSTED", "errors": [{"reason": "userRateLimitExceeded"}]},
            "rate_limited",
        ),
        (
            403,
            {"status": "PERMISSION_DENIED", "errors": [{"reason": "forbidden"}]},
            "api_forbidden",
        ),
        (503, {"status": "UNAVAILABLE", "errors": []}, "api_server_error"),
    ]
    for http_status, error_data, expected in cases:
        response = _response(http_status, {"error": error_data})
        assert PlaylistInventoryService._response_failure_category(response) == expected


def test_network_failure_is_sanitized_and_api_failure_does_not_fallback(
    tmp_path: Path,
) -> None:
    client = _QueueClient(
        [
            httpx.ConnectError(
                "contains sensitive test-secret-key",
                request=httpx.Request("GET", "https://example.test"),
            )
        ]
    )
    report = _service(tmp_path, client).build_from_api("PLplaylist123")

    assert report["enumeration_status"] == "failed"
    assert report["failure_category"] == "network_error"
    assert len(client.calls) == 1
    assert "test-secret-key" not in json.dumps(report)
    assert all("youtube.com" not in url for url, _ in client.calls)


def test_video_detail_batch_failure_retains_items_as_partial_metadata(
    tmp_path: Path,
) -> None:
    client = _QueueClient(
        [
            _playlist_response(),
            _response(
                200,
                {
                    "items": [
                        {
                            "snippet": {"position": 0, "title": "Lecture"},
                            "contentDetails": {"videoId": "video_a"},
                        }
                    ]
                },
            ),
            _response(
                403,
                {
                    "error": {
                        "errors": [{"reason": "accessNotConfigured"}],
                        "message": "API configuration details must not appear",
                    }
                },
            ),
        ]
    )

    report = _service(tmp_path, client).build_from_api("PLplaylist123")

    assert report["enumeration_status"] == "complete"
    assert report["pagination_complete"] is True
    assert report["inventory_status"] == "partial"
    assert report["videos"][0]["metadata_status"] == "partial"
    assert report["videos"][0]["failure_category"] == "api_not_enabled"
    assert len(client.calls) == 3
    assert "API configuration details" not in json.dumps(report)


def test_exact_video_id_transcript_match_and_completeness_are_reported(
    tmp_path: Path,
) -> None:
    video_id = "DrCeIbpfuzE"
    exact = tmp_path / f"{video_id}.srt"
    exact.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nSource transcript.\n",
        encoding="utf-8",
    )
    (tmp_path / f"{video_id}_abridged.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nWrong near-match.\n",
        encoding="utf-8",
    )
    client = _QueueClient(
        [
            _playlist_response(),
            _response(
                200,
                {
                    "items": [
                        {
                            "snippet": {"position": 0, "title": "Lecture"},
                            "contentDetails": {"videoId": video_id},
                        }
                    ]
                },
            ),
            _response(
                200,
                {
                    "items": [
                        {
                            "id": video_id,
                            "snippet": {"title": "Lecture"},
                            "contentDetails": {"duration": "PT10M"},
                            "status": {"privacyStatus": "public"},
                        }
                    ]
                },
            ),
        ]
    )
    report = _service(tmp_path, client).build_from_api("PLplaylist123")

    readiness = report["videos"][0]["readiness"]
    assert readiness["local_transcript_availability"] == "found"
    assert readiness["transcript_parse_status"] == "parsed"
    assert readiness["source_timestamp_provenance"] == "all_endpoints_source_provided"
    assert readiness["transcript_completeness"]["status"] == "unknown"
    assert readiness["study_note_generation_readiness"] != "ready_for_full_lecture_notes"
    assert readiness["audio_processing_eligibility"].startswith("blocked")
    assert readiness["visual_processing_eligibility"].startswith("blocked")


def test_metadata_export_marks_incomplete_inventory_partial(tmp_path: Path) -> None:
    export_path = tmp_path / "metadata.json"
    export_path.write_text(
        json.dumps(
            {
                "playlist_id": "PLplaylist123",
                "title": "Exported playlist",
                "declared_item_count": 2,
                "pagination_complete": False,
                "source_method": "authorized_metadata_export",
                "videos": [
                    {
                        "position": 0,
                        "video_id": "video_a",
                        "title": "Lecture",
                        "duration_seconds": 60,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    service = PlaylistInventoryService(
        YouTubeConnector(api_key="unused"), transcript_dirs=[tmp_path]
    )

    report = service.build_from_metadata_export(export_path)

    assert report["source_method"] == "authorized_metadata_export"
    assert report["enumeration_status"] == "partial"
    assert report["pagination_complete"] is False
    assert report["failure_category"] == "export_incomplete"
    assert report["enumerated_item_count"] == 1
