"""Export a credential-safe YouTube playlist inventory and evidence-readiness report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import gettempdir

from youtube_learning_companion.connector import YouTubeConnector, YouTubeURLParser
from youtube_learning_companion.playlist_inventory import PlaylistInventoryService


def _default_output(playlist_id: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        Path(gettempdir())
        / "youtube_learning_companion"
        / "playlist_inventories"
        / f"{playlist_id or 'invalid'}_{timestamp}.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory playlist metadata and local evidence readiness."
    )
    parser.add_argument(
        "playlist",
        help="YouTube playlist URL or playlist ID",
    )
    parser.add_argument(
        "--metadata-export",
        type=Path,
        help="Use a credential-free metadata export instead of the live API",
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory to scan for exact video-ID transcript filenames (repeatable)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path (default: a unique path under the system temp directory)",
    )
    args = parser.parse_args()

    _, playlist_id = YouTubeURLParser.extract_ids(args.playlist)
    output_path = args.output or _default_output(playlist_id or "")
    if output_path.exists():
        print("Error: Refusing to overwrite existing inventory report.", file=sys.stderr)
        return 1
    transcript_dirs = args.transcript_dir or [
        Path("data/transcripts"),
        Path("tests/fixtures/companion"),
    ]

    service = PlaylistInventoryService(
        connector=YouTubeConnector(),
        transcript_dirs=transcript_dirs,
    )
    if args.metadata_export:
        report = service.build_from_metadata_export(args.metadata_export)
    else:
        report = service.build_from_api(args.playlist)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8", newline="\n") as report_file:
            json.dump(report, report_file, indent=2, ensure_ascii=False)
            report_file.write("\n")
    except FileExistsError:
        print("Error: Refusing to overwrite existing inventory report.", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Error: Could not write inventory report ({type(error).__name__}).", file=sys.stderr)
        return 1

    print(f"Inventory status: {report['enumeration_status']}")
    print(f"Metadata inventory status: {report['inventory_status']}")
    print(f"Pagination complete: {report['pagination_complete']}")
    print(f"Enumerated playlist items: {report['enumerated_item_count']}")
    print(f"Unique video IDs: {report['unique_video_id_count']}")
    if report["failure_category"]:
        print(f"Failure category: {report['failure_category']}")
    print(f"Inventory report: {output_path}")
    return 0 if report["inventory_status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
