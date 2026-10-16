"""Offline synthetic tar tests for private frozen-asset installation and safety."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paper_release_assets", ROOT / "scripts/reproduction/release_assets.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def checksum(value):
    return hashlib.sha256(value).hexdigest()


class ReleaseAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="paper_release_safety_")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.assets = self.base / "downloaded"
        self.assets.mkdir()
        self.destination = self.base / "installed"
        self.catalog = self.base / "catalog.json"

    def fixture(self, files=None, archives=None):
        if files is None:
            files = [("data/pairs.jsonl", b'{"path":"/original/server/path"}\n', "data"),
                     ("models/adapter/model.bin", b"synthetic-adapter", "adapters"),
                     ("audio/example.wav", b"synthetic-audio", "audio")]
        manifest = {"files": [{"path": path, "sha256": checksum(content), "size_bytes": len(content), "group": group}
                              for path, content, group in files]}
        manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
        manifest_name = "bundle_file_manifest.json"
        (self.assets / manifest_name).write_bytes(manifest_bytes)
        if archives is None:
            archives = [(group, [(path, content, "file") for path, content, item_group in files if item_group == group])
                        for group in ("data", "adapters", "audio") if any(item_group == group for _, _, item_group in files)]
        parts = []
        for number, (group, members) in enumerate(archives):
            name = f"bundle-{group}-{number:03}.tar.gz"
            path = self.assets / name
            with tarfile.open(path, "w:gz") as tar:
                for member_name, content, kind in members:
                    item = tarfile.TarInfo(member_name)
                    if kind in ("symlink", "hardlink"):
                        item.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                        item.linkname = str(self.base / "outside-target")
                        tar.addfile(item)
                    elif kind == "directory":
                        item.type = tarfile.DIRTYPE
                        tar.addfile(item)
                    else:
                        item.size = len(content)
                        tar.addfile(item, io.BytesIO(content))
            parts.append({"name": name, "sha256": checksum(path.read_bytes()), "size_bytes": path.stat().st_size,
                          "group": group, "member_count": len(members)})
        catalog = {"repository": "OwenLmz/CasualGloss", "tag": "synthetic-frozen-test",
                   "manifest": {"name": manifest_name, "sha256": checksum(manifest_bytes), "size_bytes": len(manifest_bytes)},
                   "parts": parts}
        self.catalog.write_text(json.dumps(catalog), encoding="utf-8")
        return catalog

    def call(self, mode="install", *extra, with_assets=True):
        arguments = [mode, "--catalog", str(self.catalog)]
        if mode != "list":
            arguments.extend(["--destination", str(self.destination)])
        if with_assets:
            arguments.extend(["--asset-dir", str(self.assets)])
        if mode == "install":
            arguments.append("--accept-data-terms")
        arguments.extend(extra)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                status = release.main(arguments)
            except SystemExit as error:
                status = error.code
        return status, stdout.getvalue(), stderr.getvalue()

    def test_offline_install_resume_and_verify_preserve_original_bytes(self):
        self.fixture()
        with patch.object(release.GitHubDownload, "download", side_effect=AssertionError("Network is forbidden")):
            self.assertEqual(self.call()[0], 0)
            self.assertEqual(self.call()[0], 0)
            status, output, _ = self.call("verify", with_assets=False)
        self.assertEqual(status, 0)
        self.assertIn('"verified_file_count": 3', output)
        self.assertEqual((self.destination / "data/pairs.jsonl").read_bytes(), b'{"path":"/original/server/path"}\n')
        self.assertTrue((self.destination / release.METADATA_NAME).is_file())

    def test_selected_groups_can_be_installed_then_extended(self):
        self.fixture()
        self.assertEqual(self.call("install", "--groups", "data")[0], 0)
        self.assertFalse((self.destination / "audio/example.wav").exists())
        self.assertEqual(self.call("install", "--groups", "adapters,audio")[0], 0)
        self.assertEqual(self.call("verify", with_assets=False)[0], 0)

    def test_list_requires_no_network_or_destination(self):
        self.fixture()
        with patch.object(release.GitHubDownload, "download", side_effect=AssertionError("Network is forbidden")):
            status, output, _ = self.call("list", "--groups", "data", with_assets=False)
        self.assertEqual(status, 0)
        self.assertIn('"member_count": 1', output)
        self.assertFalse(self.destination.exists())

    def test_nonmatching_existing_file_is_not_overwritten(self):
        self.fixture()
        path = self.destination / "data/pairs.jsonl"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"do not overwrite")
        self.assertNotEqual(self.call()[0], 0)
        self.assertEqual(path.read_bytes(), b"do not overwrite")

    def test_unregistered_existing_content_is_rejected(self):
        self.fixture()
        self.destination.mkdir()
        path = self.destination / "unrelated.txt"
        path.write_bytes(b"unrelated")
        self.assertNotEqual(self.call()[0], 0)
        self.assertEqual(path.read_bytes(), b"unrelated")

    def test_outer_manifest_checksum_failure(self):
        self.fixture()
        (self.assets / "bundle_file_manifest.json").write_bytes(b"tampered")
        self.assertNotEqual(self.call()[0], 0)
        self.assertFalse(self.destination.exists())

    def test_archive_checksum_failure(self):
        catalog = self.fixture(files=[("safe.txt", b"abc", "data")])
        (self.assets / catalog["parts"][0]["name"]).write_bytes(b"tampered archive")
        self.assertNotEqual(self.call()[0], 0)
        self.assertFalse((self.destination / "safe.txt").exists())

    def test_member_checksum_failure_never_publishes_wrong_bytes(self):
        self.fixture(files=[("safe.txt", b"abc", "data")], archives=[("data", [("safe.txt", b"bad", "file")])])
        self.assertNotEqual(self.call()[0], 0)
        self.assertFalse((self.destination / "safe.txt").exists())
        self.assertFalse((self.destination / release.METADATA_NAME).exists())

    def test_member_size_failure(self):
        self.fixture(files=[("safe.txt", b"abc", "data")], archives=[("data", [("safe.txt", b"too long", "file")])])
        self.assertNotEqual(self.call()[0], 0)
        self.assertFalse((self.destination / "safe.txt").exists())

    def test_archive_path_traversal_and_absolute_paths(self):
        for path in ("../outside-target", "/outside-target", "C:/outside-target", "a\\..\\outside-target", "./safe.txt"):
            with self.subTest(path=path):
                self.fixture(files=[("safe.txt", b"abc", "data")], archives=[("data", [(path, b"abc", "file")])])
                self.assertNotEqual(self.call()[0], 0)
                self.assertFalse((self.base / "outside-target").exists())

    def test_archive_links_and_directories_are_rejected(self):
        for kind in ("symlink", "hardlink", "directory"):
            with self.subTest(kind=kind):
                self.fixture(files=[("safe.txt", b"", "data")], archives=[("data", [("safe.txt", b"", kind)])])
                self.assertNotEqual(self.call()[0], 0)
                self.assertFalse((self.destination / "safe.txt").exists())

    def test_duplicate_archive_member_is_rejected(self):
        self.fixture(files=[("safe.txt", b"abc", "data"), ("other.txt", b"def", "data")],
                     archives=[("data", [("safe.txt", b"abc", "file"), ("safe.txt", b"abc", "file")])])
        self.assertNotEqual(self.call()[0], 0)
        self.assertFalse((self.destination / release.METADATA_NAME).exists())

    def test_unregistered_tar_member_is_rejected(self):
        self.fixture(files=[("safe.txt", b"abc", "data")], archives=[("data", [("not-listed.txt", b"abc", "file")])])
        self.assertNotEqual(self.call()[0], 0)
        self.assertFalse((self.destination / "not-listed.txt").exists())

    def test_case_colliding_manifest_paths_are_rejected(self):
        self.fixture(files=[("Safe.txt", b"a", "data"), ("safe.txt", b"b", "data")])
        self.assertNotEqual(self.call()[0], 0)

    def test_tampered_embedded_manifest_is_rejected_on_offline_verify(self):
        self.fixture()
        self.assertEqual(self.call()[0], 0)
        path = self.destination / release.METADATA_NAME
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["file_manifest_utf8"] += " "
        path.write_text(json.dumps(metadata), encoding="utf-8")
        self.assertNotEqual(self.call("verify", with_assets=False)[0], 0)

    def test_repository_destination_is_rejected(self):
        self.fixture()
        self.destination = ROOT / "must-not-create-release-assets"
        self.assertNotEqual(self.call()[0], 0)
        self.assertFalse(self.destination.exists())

    def test_destination_symlink_cannot_escape(self):
        self.fixture(files=[("data/safe.txt", b"abc", "data")])
        self.destination.mkdir()
        outside = self.base / "outside"
        outside.mkdir()
        try:
            (self.destination / "data").symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"OS does not permit synthetic symlink creation: {error}")
        self.assertNotEqual(self.call()[0], 0)
        self.assertFalse((outside / "safe.txt").exists())

    def test_redirect_strips_credentials_for_asset_hosts(self):
        request = urllib.request.Request("https://api.github.com/repos/test/releases/assets/1", headers={"Authorization": "Bearer synthetic-secret"})
        redirected = release.SafeRedirect().redirect_request(request, None, 302, "redirect", {}, "https://release-assets.githubusercontent.com/example?sig=synthetic")
        self.assertIsNone(redirected.get_header("Authorization"))
        self.assertEqual(request.get_header("Authorization"), "Bearer synthetic-secret")
        with self.assertRaises(ValueError):
            release.SafeRedirect().redirect_request(request, None, 302, "redirect", {}, "https://not-github.example/asset")

    def test_authenticated_rest_download_is_checked_without_logging_credentials(self):
        catalog = self.fixture(files=[("safe.txt", b"abc", "data")])
        part = catalog["parts"][0]
        content = (self.assets / part["name"]).read_bytes()
        downloader = release.GitHubDownload(catalog)
        responses = [io.BytesIO(b'{"id":123}'), io.BytesIO(json.dumps([
            {"id": 456, "name": part["name"], "size": len(content)}
        ]).encode()), io.BytesIO(content)]
        capture = io.StringIO()
        with patch.object(release.shutil, "which", return_value=None), patch.object(release, "git_credential", return_value="synthetic-private-token"), \
                patch.object(downloader.opener, "open", side_effect=responses) as opened, redirect_stdout(capture), redirect_stderr(capture):
            target = self.assets / "rest-downloaded-copy.tar.gz"
            downloader.download(part, target)
        self.assertEqual(target.read_bytes(), content)
        self.assertEqual(opened.call_count, 3)
        self.assertNotIn("synthetic-private-token", capture.getvalue())
        self.assertTrue(all(call.args[0].full_url.startswith("https://api.github.com/repos/OwenLmz/CasualGloss/") for call in opened.call_args_list))

    def test_matching_existing_asset_is_not_replaced_when_tampered_tar_is_supplied(self):
        catalog = self.fixture(files=[("safe.txt", b"abc", "data")])
        self.assertEqual(self.call()[0], 0)
        archive = self.assets / catalog["parts"][0]["name"]
        archive.write_bytes(b"tampered")
        self.assertNotEqual(self.call()[0], 0)
        self.assertEqual((self.destination / "safe.txt").read_bytes(), b"abc")

    def test_manifest_file_directory_collision_is_rejected(self):
        self.fixture(files=[("a", b"file", "data"), ("a/b", b"nested", "data")])
        self.assertNotEqual(self.call()[0], 0)


if __name__ == "__main__":
    unittest.main()
