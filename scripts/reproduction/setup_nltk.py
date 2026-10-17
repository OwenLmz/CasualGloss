"""Acquire and verify the exact English Punkt resource used by NLTK 3.9.2.

Standard library only. A mutable official URL is never trusted without the
recorded archive SHA/size and four English member SHA/size checks. The untouched
official ZIP is retained, but only English models are extracted. Keep this
resource outside the Git checkout and original experimental data.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import sys
from urllib.parse import urlsplit
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "provenance/nltk_resources.json"
OFFICIAL_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
ENGLISH_NAMES = {"abbrev_types.txt", "collocations.tab", "ortho_context.tab", "sent_starters.txt"}


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_bytes(data, entry, name):
    require(len(data) == entry["size_bytes"], "Size mismatch: " + name)
    require(checksum(data) == entry["sha256"], "SHA-256 mismatch: " + name)


def checked_file_bytes(path, entry):
    require(path.stat().st_size == entry["size_bytes"], "Size mismatch: " + str(path))
    data = path.read_bytes()
    check_bytes(data, entry, str(path))
    return data


def load_spec():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    require(spec["url"] == OFFICIAL_URL, "Only the recorded official NLTK resource URL is allowed")
    require(spec["archive_relative_path"] == "tokenizers/punkt_tab.zip", "Unexpected archive destination")
    require(spec["language"] == "english" and spec["resource"] == "punkt_tab", "Unexpected resource/language")
    require({row["member"] for row in spec["files"]} == {"punkt_tab/english/" + n for n in ENGLISH_NAMES}
            and len(spec["files"]) == 4, "Exactly four recorded English model files are required")
    return spec


def no_symlinks(path):
    for candidate in (path, *path.parents):
        require(not candidate.is_symlink(), "Refusing a symlink in the destination path: " + str(candidate))
        junction = getattr(candidate, "is_junction", None)
        require(not junction or not junction(), "Refusing a junction in the destination path: " + str(candidate))


def destination_path(value):
    path = Path(value).absolute()
    no_symlinks(path)
    path = path.resolve()
    root = ROOT.resolve()
    require(path != root and root not in path.parents and path not in root.parents,
            "Keep NLTK resources outside the source checkout and its ancestor directories")
    require(path.parent != path, "Refusing a filesystem root destination")
    logical = PurePosixPath(path.as_posix())
    original = PurePosixPath("/123090274/SN_Research")
    require(logical != original and original not in logical.parents and logical not in original.parents,
            "Refusing any overlap with original SN_Research experimental data")
    require(not path.exists() or path.is_dir(), "Destination must be a directory")
    return path


def expected_files(spec):
    entries = {spec["archive_relative_path"]: spec}
    entries.update({"tokenizers/" + row["member"]: row for row in spec["files"]})
    return entries


def inspect_existing(destination, spec, require_complete=False):
    expected = expected_files(spec)
    allowed_dirs = set()
    for name in expected:
        allowed_dirs.update(p.as_posix() for p in PurePosixPath(name).parents if p.as_posix() != ".")
    if destination.exists():
        for path in destination.rglob("*"):
            relative = path.relative_to(destination).as_posix()
            no_symlinks(path)
            if path.is_dir():
                require(relative in allowed_dirs, "Unregistered existing directory preserved: " + relative)
            else:
                require(path.is_file() and relative in expected,
                        "Unregistered existing file preserved: " + relative)
                checked_file_bytes(path, expected[relative])
    if require_complete:
        for relative in expected:
            path = destination / relative
            require(path.is_file(), "Missing recorded NLTK resource: " + relative)


def download_archive(spec):
    require(spec["url"] == OFFICIAL_URL, "Only the recorded official URL may be downloaded")
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "CausalGloss-NLTK-resource-verifier"})
    with urllib.request.urlopen(request, timeout=60) as response:
        final = urlsplit(response.geturl())
        require(final.scheme == "https" and final.hostname == "raw.githubusercontent.com"
                and final.username is None and final.password is None,
                "Refusing an unexpected or insecure resource redirect")
        data = response.read(spec["size_bytes"] + 1)
    check_bytes(data, spec, "official punkt_tab.zip")
    return data


def english_members(data, spec):
    check_bytes(data, spec, "punkt_tab.zip")
    wanted = {row["member"]: row for row in spec["files"]}
    selected = {}
    seen = set()
    total_size = 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for item in archive.infolist():
            name = item.filename
            raw_name = item.orig_filename  # Preserve pre-normalization/NUL names on Windows.
            path = PurePosixPath(name)
            require(name == raw_name and name and "\\" not in raw_name and ":" not in name and "\x00" not in raw_name
                    and not path.is_absolute() and all(p not in ("", ".", "..") for p in name.rstrip("/").split("/")),
                    "Unsafe ZIP member path: " + repr(name))
            key = name.rstrip("/").casefold()
            require(key not in seen, "Duplicate/ambiguous ZIP member: " + name)
            seen.add(key)
            mode = (item.external_attr >> 16) & 0xFFFF
            kind = stat.S_IFMT(mode)
            require(kind in (0, stat.S_IFREG, stat.S_IFDIR) and not stat.S_ISLNK(mode),
                    "Refusing symlink or special ZIP member: " + name)
            require(kind != stat.S_IFDIR or item.is_dir(), "Inconsistent ZIP directory type: " + name)
            require(kind != stat.S_IFREG or not item.is_dir(), "Inconsistent ZIP file type: " + name)
            require(not (item.flag_bits & 1), "Encrypted ZIP members are not accepted")
            total_size += item.file_size
            require(total_size <= spec["max_uncompressed_bytes"], "ZIP uncompressed-size limit exceeded")
            if name in wanted:
                require(not item.is_dir() and item.file_size == wanted[name]["size_bytes"],
                        "Unexpected English member size/type: " + name)
                payload = archive.read(item)
                check_bytes(payload, wanted[name], name)
                selected["tokenizers/" + name] = payload
    require(len(selected) == 4 and set(selected) == {"tokenizers/" + n for n in wanted},
            "Archive does not contain exactly the four required English resources")
    return selected


def setup(destination, spec, *, archive=None, verify_only=False):
    destination = destination_path(destination)
    inspect_existing(destination, spec, require_complete=verify_only)
    target = destination / spec["archive_relative_path"]
    if verify_only:
        require(archive is None, "--archive is not used by offline --verify-only")
        data = checked_file_bytes(target, spec)
    elif target.is_file():
        data = checked_file_bytes(target, spec)
    elif archive is not None:
        data = checked_file_bytes(Path(archive), spec)
    else:
        data = download_archive(spec)
    selected = english_members(data, spec)  # Validate everything before any write.
    if not verify_only:
        payloads = {spec["archive_relative_path"]: data, **selected}
        for relative, payload in payloads.items():
            path = destination / relative
            no_symlinks(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                checked_file_bytes(path, expected_files(spec)[relative])
                continue
            # Exclusive creation never overwrites an original or mismatching file.
            # A disk interruption leaves a visible partial file which a later
            # run refuses; use a new destination rather than discarding evidence.
            with path.open("xb") as handle:
                handle.write(payload)
        inspect_existing(destination, spec, require_complete=True)
    return {"passed": True, "mode": "offline_verify_only" if verify_only else "install_and_verify",
            "destination": str(destination), "nltk_data": str(destination), "resource": "punkt_tab",
            "language": "english", "recorded_runtime_nltk_version": spec["recorded_runtime_nltk_version"],
            "verified_file_count": 5, "english_model_file_count": 4,
            "official_archive_preserved_byte_for_byte": True, "archive_sha256": spec["sha256"],
            "legacy_punkt_installed": False,
            "next_step": "Set NLTK_DATA to this destination or pass --nltk-data to preflight/checkpoint commands."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True, help="NEW external NLTK_DATA directory, never original experiment data")
    parser.add_argument("--archive", type=Path, help="Optional existing official ZIP for an offline installation; read-only")
    parser.add_argument("--verify-only", action="store_true", help="Offline SHA/size verification of an already installed resource; no network/writes")
    args = parser.parse_args(argv)
    try:
        report = setup(args.destination, load_spec(), archive=args.archive, verify_only=args.verify_only)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(json.dumps({"passed": False, "error": type(error).__name__ + ": " + str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
