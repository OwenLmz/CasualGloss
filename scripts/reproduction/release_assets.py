#!/usr/bin/env python3
"""Install checksummed final-paper assets from the authorized private release.

Examples (all data remains outside this source checkout)::

    python scripts/reproduction/release_assets.py list
    python scripts/reproduction/release_assets.py install --destination /private/paper_assets \
        --asset-dir /private/downloaded_release --groups data,adapters,audio --accept-data-terms
    python scripts/reproduction/release_assets.py verify --destination /private/paper_assets

Without --asset-dir, install downloads with authenticated gh, or the existing
Git credential helper. Credentials are used only in memory and never printed.
Original JSONL contents and their historical absolute path strings are not edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SCOPED_REPOSITORY = "OwenLmz/CasualGloss"
GROUPS = frozenset({"data", "adapters", "audio"})
METADATA_NAME = "._release_install.json"
BLOCK_SIZE = 8 * 1024 * 1024
MANIFEST_LIMIT = 64 * 1024 * 1024
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
ASSET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
            value.update(block)
    return value.hexdigest()


def relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("Invalid portable relative asset path")
    pieces = value.split("/")
    if any(part in ("", ".", "..") or part.endswith((".", " ")) for part in pieces):
        raise ValueError("Unsafe relative asset path")
    if any(part.split(".", 1)[0].upper() in RESERVED_NAMES for part in pieces):
        raise ValueError("Reserved device name in asset path")
    if PurePosixPath(value).is_absolute() or value == METADATA_NAME:
        raise ValueError("Absolute or reserved asset path")
    return value


def check_digest_record(value: dict[str, Any]) -> None:
    if not isinstance(value.get("sha256"), str) or not SHA256.fullmatch(value["sha256"]):
        raise ValueError("Invalid SHA-256 in asset registry")
    if type(value.get("size_bytes")) is not int or value["size_bytes"] < 0:
        raise ValueError("Invalid asset size in registry")


def load_catalog(path: Path) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(catalog, dict):
        raise ValueError("Release catalog must be a JSON object")
    if catalog.get("repository") != SCOPED_REPOSITORY:
        raise ValueError("Release catalog must name the authorized OwenLmz/CasualGloss repository")
    if not isinstance(catalog.get("tag"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", catalog["tag"]):
        raise ValueError("Invalid release tag")
    manifest = catalog.get("manifest")
    parts = catalog.get("parts")
    if not isinstance(manifest, dict) or not isinstance(parts, list) or not parts:
        raise ValueError("Catalog requires a manifest and nonempty parts list")
    names = set()
    for entry in [manifest, *parts]:
        if not isinstance(entry, dict) or not ASSET_NAME.fullmatch(str(entry.get("name", ""))):
            raise ValueError("Invalid release asset name")
        if entry["name"].casefold() in names:
            raise ValueError("Duplicate release asset name")
        names.add(entry["name"].casefold())
        check_digest_record(entry)
        if entry is not manifest:
            if entry.get("group") not in GROUPS or not entry["name"].endswith(".tar.gz"):
                raise ValueError("Invalid release part group or archive extension")
            if type(entry.get("member_count")) is not int or entry["member_count"] < 0:
                raise ValueError("Invalid part member count")
    if manifest["size_bytes"] > MANIFEST_LIMIT:
        raise ValueError("External file manifest is unexpectedly large")
    return catalog


def parse_file_manifest(raw: bytes, catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = catalog["manifest"]
    if len(raw) != expected["size_bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError("External file manifest checksum or size mismatch")
    document = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(document, dict) or not isinstance(document.get("files"), list):
        raise ValueError("External manifest requires a files list")
    result = {}
    folded = set()
    for item in document["files"]:
        if not isinstance(item, dict):
            raise ValueError("Invalid external manifest entry")
        path = relative_path(item.get("path"))
        check_digest_record(item)
        if item.get("group") not in GROUPS:
            raise ValueError("Invalid file group")
        if path.casefold() in folded:
            raise ValueError("Duplicate or case-colliding file path")
        result[path] = item
        folded.add(path.casefold())
    for path in result:
        parents = PurePosixPath(path).parents
        if any(str(parent).casefold() in folded for parent in parents if str(parent) != "."):
            raise ValueError("Manifest file/directory path collision")
    for group in GROUPS:
        expected_count = sum(part["member_count"] for part in catalog["parts"] if part["group"] == group)
        if sum(item["group"] == group for item in result.values()) != expected_count:
            raise ValueError("Group file count differs between outer and external manifests")
    return result


def is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def no_link_ancestors(path: Path) -> None:
    absolute = path.absolute()
    for candidate in [*reversed(absolute.parents), absolute]:
        if is_link(candidate):
            raise ValueError("Refusing a path containing a symlink or junction")


def outside_checkout(path: Path) -> Path:
    if ".." in path.parts:
        raise ValueError("Use an absolute or normalized path without parent traversal")
    no_link_ancestors(path)
    absolute = path.absolute()
    resolved = absolute.resolve()
    root = ROOT.resolve()
    if resolved == root or root in resolved.parents or resolved == Path(resolved.anchor):
        raise ValueError("Use a private directory outside the source checkout, not a filesystem root")
    return resolved


def verified_file(path: Path, entry: dict[str, Any]) -> None:
    no_link_ancestors(path)
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ValueError("Asset must be a regular file, not a link")
    if value.st_size != entry["size_bytes"] or file_sha256(path) != entry["sha256"]:
        raise ValueError(f"Checksum or size mismatch for {path.name}; refusing to overwrite")


def inventory_destination(destination: Path, files: dict[str, dict[str, Any]]) -> set[str]:
    """Reject unrelated content and preexisting mismatches before any extraction."""
    no_link_ancestors(destination)
    if not destination.exists():
        return set()
    if not destination.is_dir():
        raise ValueError("Destination is not a directory")
    allowed_dirs = {str(parent) for path in files for parent in PurePosixPath(path).parents if str(parent) != "."}
    found = set()
    pending = [destination]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            relative = path.relative_to(destination).as_posix()
            if is_link(path):
                raise ValueError("Link found in destination; refusing extraction")
            if path.is_dir():
                if relative not in allowed_dirs:
                    raise ValueError("Unregistered directory found in destination")
                pending.append(path)
            elif relative == METADATA_NAME:
                if not path.is_file() or path.stat().st_nlink != 1:
                    raise ValueError("Installation metadata must be a regular file")
            elif relative in files:
                verified_file(path, files[relative])
                found.add(relative)
            else:
                raise ValueError("Unregistered file found in destination")
    return found


def read_metadata(destination: Path, catalog: dict[str, Any]) -> dict[str, Any] | None:
    path = destination / METADATA_NAME
    if not path.exists():
        return None
    no_link_ancestors(path)
    if path.stat().st_size > 2 * MANIFEST_LIMIT:
        raise ValueError("Installation metadata is unexpectedly large")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if (metadata.get("repository") != catalog["repository"] or metadata.get("tag") != catalog["tag"]
            or metadata.get("manifest_sha256") != catalog["manifest"]["sha256"]):
        raise ValueError("Destination metadata belongs to a different frozen release")
    return metadata


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        parsed = urllib.parse.urlparse(new_url)
        allowed = {"api.github.com", "github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com", "github-releases.githubusercontent.com"}
        if (parsed.scheme != "https" or parsed.hostname not in allowed or parsed.port not in (None, 443)
                or parsed.username is not None or parsed.password is not None):
            raise ValueError("Unexpected download redirect host")
        redirected = super().redirect_request(request, fp, code, message, headers, new_url)
        if redirected is not None and parsed.hostname != "api.github.com":
            redirected.remove_header("Authorization")
        return redirected


def git_credential(repository: str) -> str:
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
    result = subprocess.run(["git", "-c", "credential.interactive=false", "credential", "fill"],
        input=f"protocol=https\nhost=github.com\npath={repository}.git\n\n", text=True,
        capture_output=True, cwd=ROOT, env=env, check=False, timeout=60)
    if result.returncode:
        raise RuntimeError("Authenticate gh or authorize the existing Git credential helper for this private repository")
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if not fields.get("password"):
        raise RuntimeError("No credential available for the authorized private repository")
    if any(ord(character) <= 32 or ord(character) >= 127 for character in fields["password"]):
        raise RuntimeError("Credential helper returned an invalid credential format")
    return fields["password"]


class GitHubDownload:
    def __init__(self, catalog: dict[str, Any]):
        self.catalog = catalog
        self.token = None
        self.assets = None
        self.opener = urllib.request.build_opener(SafeRedirect())

    def response(self, path: str, accept: str):
        if self.token is None:
            self.token = git_credential(self.catalog["repository"])
        request = urllib.request.Request("https://api.github.com" + path, headers={
            "Authorization": "Bearer " + self.token, "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "CausalGloss-private-reproduction"})
        return self.opener.open(request, timeout=120)

    def asset_map(self):
        if self.assets is None:
            repository = self.catalog["repository"]
            with self.response(f"/repos/{repository}/releases/tags/" + urllib.parse.quote(self.catalog["tag"], safe=""), "application/vnd.github+json") as response:
                release = json.load(response)
            release_id = release.get("id")
            if type(release_id) is not int:
                raise ValueError("Invalid GitHub release identity")
            assets = []
            for page in range(1, 101):
                with self.response(f"/repos/{repository}/releases/{release_id}/assets?per_page=100&page={page}", "application/vnd.github+json") as response:
                    values = json.load(response)
                if not isinstance(values, list):
                    raise ValueError("Invalid GitHub asset listing")
                assets.extend(values)
                if len(values) < 100:
                    break
            else:
                raise ValueError("Unexpectedly large GitHub release asset listing")
            self.assets = {item["name"]: item for item in assets}
        return self.assets

    def download(self, entry: dict[str, Any], target: Path):
        # Use a separate new directory so neither gh nor REST can overwrite a cache file.
        with tempfile.TemporaryDirectory(prefix=".release-download-", dir=target.parent) as temp:
            candidate = Path(temp) / entry["name"]
            if shutil.which("gh"):
                result = subprocess.run(["gh", "release", "download", self.catalog["tag"],
                    "--repo", self.catalog["repository"], "--pattern", entry["name"], "--dir", temp],
                    capture_output=True, check=False, timeout=7200)
                if result.returncode == 0 and candidate.is_file():
                    verified_file(candidate, entry)
                    publish_file(candidate, target)
                    return
                if candidate.exists():
                    raise RuntimeError("gh left an incomplete asset; refusing an ambiguous retry")
            try:
                asset = self.asset_map().get(entry["name"])
                if not asset or type(asset.get("id")) is not int or asset.get("size") != entry["size_bytes"]:
                    raise ValueError("GitHub release asset identity or size mismatch")
                with self.response(f"/repos/{self.catalog['repository']}/releases/assets/{asset['id']}", "application/octet-stream") as response, candidate.open("xb") as output:
                    copied = 0
                    for block in iter(lambda: response.read(BLOCK_SIZE), b""):
                        copied += len(block)
                        if copied > entry["size_bytes"]:
                            raise ValueError("Download exceeded registered asset size")
                        output.write(block)
            except (urllib.error.HTTPError, urllib.error.URLError) as error:
                code = error.code if isinstance(error, urllib.error.HTTPError) else "network"
                raise RuntimeError(f"Private GitHub release download failed ({code}); check access and release availability") from None
            verified_file(candidate, entry)
            publish_file(candidate, target)


def publish_file(source: Path, target: Path) -> None:
    """Publish verified bytes without an overwrite race; both paths share a filesystem."""
    no_link_ancestors(target.parent)
    if target.exists() or is_link(target):
        raise FileExistsError("Refusing to overwrite an existing asset")
    if os.name == "nt":
        # Windows rename fails if target exists, unlike POSIX rename. This
        # also works in Windows sandboxes that prohibit creating hardlinks.
        os.rename(source, target)
    else:
        os.link(source, target, follow_symlinks=False)
        source.unlink()


def acquire_asset(directory: Path, entry: dict[str, Any], downloader: GitHubDownload | None) -> Path:
    no_link_ancestors(directory)
    path = directory / entry["name"]
    if path.exists() or is_link(path):
        verified_file(path, entry)
        return path
    if downloader is None:
        raise FileNotFoundError(f"Missing locally downloaded release asset: {entry['name']}")
    directory.mkdir(parents=True, exist_ok=True)
    downloader.download(entry, path)
    verified_file(path, entry)
    return path


def make_parent(destination: Path, relative: str) -> Path:
    no_link_ancestors(destination)
    destination.mkdir(parents=True, exist_ok=True)
    parent = destination
    for piece in PurePosixPath(relative).parts[:-1]:
        parent = parent / piece
        if is_link(parent):
            raise ValueError("Symlink or junction in extraction path")
        parent.mkdir(exist_ok=True)
    no_link_ancestors(parent)
    return parent


def install_part(archive_path: Path, part: dict[str, Any], files: dict[str, dict[str, Any]],
                 destination: Path, seen: set[str]) -> dict[str, int]:
    count = new = resumed = 0
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            path = relative_path(member.name)
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("Archive contains a non-regular-file member")
            if path in seen:
                raise ValueError("Duplicate member across release archives")
            entry = files.get(path)
            if entry is None or entry["group"] != part["group"] or member.size != entry["size_bytes"]:
                raise ValueError("Unregistered archive member, wrong group, or size mismatch")
            seen.add(path)
            count += 1
            target = destination.joinpath(*PurePosixPath(path).parts)
            if target.exists() or is_link(target):
                verified_file(target, entry)
                # Even on resume, check the archived member's own bytes.
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Missing archive member stream")
                value = hashlib.sha256()
                copied = 0
                with stream:
                    for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
                        copied += len(block)
                        value.update(block)
                if copied != entry["size_bytes"] or value.hexdigest() != entry["sha256"]:
                    raise ValueError("Archived member checksum mismatch during resume")
                resumed += 1
                continue
            parent = make_parent(destination, path)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(prefix=".release-member-", dir=parent, delete=False) as output:
                    temporary = Path(output.name)
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError("Missing archive member stream")
                    value = hashlib.sha256()
                    copied = 0
                    with stream:
                        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
                            copied += len(block)
                            if copied > entry["size_bytes"]:
                                raise ValueError("Member exceeded registered size")
                            value.update(block)
                            output.write(block)
                if copied != entry["size_bytes"] or value.hexdigest() != entry["sha256"]:
                    raise ValueError("Archived member checksum mismatch")
                publish_file(temporary, target)
                temporary = None
                new += 1
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
    if count != part["member_count"]:
        raise ValueError("Archive member count differs from release registry")
    return {"members": count, "installed": new, "already_verified": resumed}


def verify_destination(destination: Path, files: dict[str, dict[str, Any]], groups: set[str]) -> dict[str, Any]:
    found = inventory_destination(destination, files)
    selected = {path for path, entry in files.items() if entry["group"] in groups}
    missing = selected - found
    if missing:
        raise ValueError(f"Missing {len(missing)} registered files in selected groups")
    return {"verified_file_count": len(selected), "verified_bytes": sum(files[path]["size_bytes"] for path in selected),
            "groups": sorted(groups), "unregistered_files": 0}


def save_metadata(destination: Path, catalog: dict[str, Any], raw: bytes, files: dict[str, dict[str, Any]]) -> None:
    no_link_ancestors(destination)
    destination.mkdir(parents=True, exist_ok=True)
    found = inventory_destination(destination, files)
    metadata = {"schema_version": 1, "repository": catalog["repository"], "tag": catalog["tag"],
                "manifest_sha256": catalog["manifest"]["sha256"], "file_manifest_utf8": raw.decode("utf-8"),
                "installed_file_count": len(found), "original_file_contents_unchanged": True}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", prefix=".release-metadata-", dir=destination, delete=False) as output:
        temporary = Path(output.name)
        json.dump(metadata, output, ensure_ascii=False, indent=2)
        output.write("\n")
    try:
        target = destination / METADATA_NAME
        no_link_ancestors(target)
        # This is our previously validated metadata, never a scientific asset.
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["list", "install", "verify"])
    parser.add_argument("--catalog", type=Path, default=ROOT / "provenance/release_assets.json")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--groups", default="data,adapters,audio")
    parser.add_argument("--asset-dir", type=Path, help="Read already downloaded assets locally; never access the network")
    parser.add_argument("--cache-dir", type=Path, help="Download cache outside the checkout (defaults to a sibling of destination)")
    parser.add_argument("--accept-data-terms", action="store_true", help="Confirm access and upstream terms in THIRD_PARTY.md")
    args = parser.parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
        groups = set(args.groups.split(","))
        if not groups or not groups <= GROUPS:
            raise ValueError("--groups must be a comma-separated subset of data,adapters,audio")
        parts = [part for part in catalog["parts"] if part["group"] in groups]
        if args.mode == "list":
            print(json.dumps({"repository": catalog["repository"], "tag": catalog["tag"], "groups": sorted(groups),
                "manifest": catalog["manifest"], "parts": parts, "archive_bytes": sum(part["size_bytes"] for part in parts),
                "member_count": sum(part["member_count"] for part in parts)}, indent=2))
            return 0
        if args.destination is None:
            raise ValueError("--destination is required")
        destination = outside_checkout(args.destination)
        if args.mode == "verify" and not destination.is_dir():
            raise ValueError("Verification destination does not exist")
        metadata = read_metadata(destination, catalog)
        if args.mode == "verify" and args.asset_dir is None:
            if metadata is None or not isinstance(metadata.get("file_manifest_utf8"), str):
                raise ValueError("Verification needs valid installation metadata or --asset-dir")
            raw = metadata["file_manifest_utf8"].encode("utf-8")
            files = parse_file_manifest(raw, catalog)
        else:
            if args.mode == "install" and not args.accept_data_terms:
                raise ValueError("Review THIRD_PARTY.md and supply --accept-data-terms before installing")
            if args.asset_dir is not None:
                assets = outside_checkout(args.asset_dir)
                downloader = None
            else:
                assets = outside_checkout(args.cache_dir or destination.with_name(destination.name + "_release_assets"))
                downloader = GitHubDownload(catalog)
            if assets == destination or destination in assets.parents or assets in destination.parents:
                raise ValueError("Release assets/cache and installed destination must be separate sibling trees")
            manifest_path = acquire_asset(assets, catalog["manifest"], downloader)
            raw = manifest_path.read_bytes()
            files = parse_file_manifest(raw, catalog)
        inventory_destination(destination, files)
        if args.mode == "install":
            seen = set()
            installed = resumed = 0
            for part in parts:
                archive_path = acquire_asset(assets, part, downloader)
                summary = install_part(archive_path, part, files, destination, seen)
                installed += summary["installed"]
                resumed += summary["already_verified"]
                print(json.dumps({"verified_part": part["name"], **summary}))
            expected = {path for path, entry in files.items() if entry["group"] in groups}
            if seen != expected:
                raise ValueError("Archives do not exactly cover the selected registered file set")
            summary = verify_destination(destination, files, groups)
            save_metadata(destination, catalog, raw, files)
            summary.update(installed_file_count=installed, already_verified_file_count=resumed)
        else:
            summary = verify_destination(destination, files, groups)
        print(json.dumps({"status": "verified", "repository": catalog["repository"], "tag": catalog["tag"],
                          "destination": str(destination), **summary}, indent=2))
        return 0
    except subprocess.SubprocessError:
        parser.exit(1, "Release asset operation failed: authentication/download subprocess failed or timed out\n")
    except (OSError, ValueError, RuntimeError, tarfile.TarError, json.JSONDecodeError) as error:
        # Do not expose request headers, subprocess stderr, or credential-helper output.
        parser.exit(1, f"Release asset operation failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
