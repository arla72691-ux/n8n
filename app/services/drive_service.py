"""
Google Drive service for searching and downloading engineering drawings.

Authentication: uses a service account JSON (file path or inline JSON string).
The folder to search is configured via GOOGLE_DRIVE_DRAWINGS_FOLDER_ID.
"""
import io
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

from app.config import get_settings


_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


@dataclass
class DriveFile:
    id: str
    name: str
    mime_type: str


def _build_drive_service():
    settings = get_settings()
    sa_json = settings.google_service_account_json

    if not sa_json:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON is not configured")

    # Accept either a file path or an inline JSON string
    if os.path.isfile(sa_json):
        creds = service_account.Credentials.from_service_account_file(
            sa_json, scopes=_SCOPES
        )
    else:
        sa_info = json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=_SCOPES
        )

    return build("drive", "v3", credentials=creds, cache_discovery=False)


@lru_cache(maxsize=1)
def get_drive_service():
    return _build_drive_service()


def _normalise_rev_for_filename(revision: str) -> str:
    """Strip leading zeros for filename matching: '00' → '0', '01' → '1'."""
    try:
        return str(int(revision))
    except (ValueError, TypeError):
        return revision.upper()


def search_drawing(
    drawing_number: str,
    revision: str,
    folder_id: Optional[str] = None,
) -> Optional[DriveFile]:
    """
    Search the Drawings folder for a file matching the given drawing number.

    Match priority:
      1. {drawing_number}-REV{revision}.pdf  (exact revision match)
      2. {drawing_number}-REV{norm_rev}.pdf  (normalised revision, e.g. REV0 for 00)
      3. {drawing_number}.pdf                (no revision in name — fallback)

    Returns the first matching DriveFile or None.
    """
    settings = get_settings()
    folder_id = folder_id or settings.google_drive_drawings_folder_id

    if not folder_id:
        raise ValueError("GOOGLE_DRIVE_DRAWINGS_FOLDER_ID is not configured")

    service = get_drive_service()

    query = (
        f"'{folder_id}' in parents"
        f" and name contains '{drawing_number}'"
        f" and trashed=false"
    )
    results = (
        service.files()
        .list(q=query, fields="files(id, name, mimeType)", pageSize=50)
        .execute()
    )
    files: list = results.get("files", [])

    if not files:
        return None

    norm_rev = _normalise_rev_for_filename(revision)

    # Build candidate name patterns in priority order
    candidates = [
        f"{drawing_number}-REV{revision}",   # as-is
        f"{drawing_number}-REV{norm_rev}",   # normalised
        drawing_number,                       # bare name (no rev)
    ]

    def score(f: dict) -> int:
        name_upper = f["name"].upper()
        for i, candidate in enumerate(candidates):
            if candidate.upper() in name_upper:
                return i
        return 999

    files_sorted = sorted(files, key=score)
    best = files_sorted[0]

    if score(best) == 999:
        return None

    return DriveFile(id=best["id"], name=best["name"], mime_type=best.get("mimeType", ""))


def download_file_bytes(file_id: str) -> bytes:
    """Download a file from Google Drive and return its raw bytes."""
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()
