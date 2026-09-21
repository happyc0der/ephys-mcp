"""DANDI Archive: dataset search and streamed NWB access.

Files are streamed over HTTP range requests, never mirrored. URLs come only
from the archive's API, so this source cannot be pointed at arbitrary hosts.
"""

from __future__ import annotations

import re

import httpx

from .nwb import NwbSource

API = "https://api.dandiarchive.org/api"
_ID = re.compile(r"^\d{6}$")
_client: httpx.Client | None = None

# Small, well-documented intracortical datasets that suit decoding demos.
CATALOG = [
    {
        "dandiset_id": "000140",
        "name": "MC_Maze_Small",
        "why": "29 MB. Macaque M1/PMd units with hand velocity; fastest way to try a decoder on real data.",
    },
    {"dandiset_id": "000128", "name": "MC_Maze", "why": "Full delayed-reaching session from the same benchmark."},
    {
        "dandiset_id": "000129",
        "name": "MC_RTT",
        "why": "Continuous random-target reaching; good for velocity decoding.",
    },
]


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(base_url=API, timeout=30.0, headers={"User-Agent": "ephys-mcp"})
    return _client


def _get(path: str, **params) -> dict:
    r = client().get(path, params=params)
    if r.status_code == 404:
        raise ValueError(f"not found on DANDI: {path}")
    r.raise_for_status()
    return r.json()


def _check_id(dandiset_id: str) -> str:
    if not _ID.match(dandiset_id):
        raise ValueError("dandiset_id must be six digits, e.g. '000140'")
    return dandiset_id


def search(query: str, limit: int = 10) -> list[dict]:
    data = _get("/dandisets/", search=query, page_size=max(1, min(limit, 25)), embargoed="false", empty="false")
    out = []
    for d in data["results"]:
        v = d.get("most_recent_published_version") or d.get("draft_version") or {}
        out.append(
            {
                "dandiset_id": d["identifier"],
                "name": v.get("name", ""),
                "version": v.get("version", ""),
                "n_files": v.get("asset_count"),
                "size_gb": round(v.get("size", 0) / 1e9, 2),
            }
        )
    return out


def resolve_version(dandiset_id: str, version: str | None) -> str:
    if version:
        return version
    d = _get(f"/dandisets/{_check_id(dandiset_id)}/")
    return (d.get("most_recent_published_version") or d["draft_version"])["version"]


def describe(dandiset_id: str, version: str | None = None) -> dict:
    version = resolve_version(dandiset_id, version)
    meta = _get(f"/dandisets/{_check_id(dandiset_id)}/versions/{version}/")
    return {
        "dandiset_id": dandiset_id,
        "version": version,
        "name": meta.get("name", ""),
        "license": ", ".join(s.removeprefix("spdx:") for s in meta.get("license", [])) or "unknown",
        "citation": meta.get("citation", ""),
        "description": (meta.get("description") or "")[:600],
    }


def list_files(dandiset_id: str, version: str | None = None, limit: int = 50) -> list[dict]:
    version = resolve_version(dandiset_id, version)
    data = _get(
        f"/dandisets/{_check_id(dandiset_id)}/versions/{version}/assets/",
        page_size=max(1, min(limit, 100)),
        glob="*.nwb",
    )
    return [
        {"path": a["path"], "size_mb": round(a["size"] / 1e6, 1), "asset_id": a["asset_id"]} for a in data["results"]
    ]


def _download_url(asset_id: str) -> str:
    r = client().get(f"/assets/{asset_id}/download/", follow_redirects=False)
    url = r.headers.get("location", "")
    if r.status_code not in (301, 302, 307) or not url.startswith("https://"):
        raise ValueError(f"DANDI did not return a download location for asset {asset_id}")
    return url


class DandiSource(NwbSource):
    kind = "dandi"

    def __init__(self, dandiset_id: str = "000140", path: str = "", version: str = ""):
        import remfile

        meta = describe(dandiset_id, version or None)
        files = list_files(dandiset_id, meta["version"], limit=100)
        if not files:
            raise ValueError(f"dandiset {dandiset_id} has no .nwb files")
        if path:
            match = [f for f in files if f["path"] == path]
            if not match:
                raise ValueError(f"no file {path!r} in dandiset {dandiset_id}; call list_dataset_files")
        else:  # prefer a file that carries behaviour, since that is what decoding needs
            match = sorted(files, key=lambda f: ("behavior" not in f["path"], f["size_mb"]))
        asset = match[0]
        super().__init__(
            _file=remfile.File(_download_url(asset["asset_id"])),
            _uri=f"dandi://{dandiset_id}/{meta['version']}/{asset['path']}",
            _license=meta["license"],
            _citation=meta["citation"],
        )
