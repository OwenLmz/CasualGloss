"""Offline/adversarial tests for the public English tokenizer resource setup."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paper_nltk_setup", ROOT / "scripts/reproduction/setup_nltk.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class NltkSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="paper_nltk_safety_")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.destination = self.base / "nltk_data"
        self.payloads = {"punkt_tab/english/" + name: (name + " synthetic model\n").encode()
                         for name in setup.ENGLISH_NAMES}

    def fixture(self, extra=(), omit=None, wrong_member=None):
        payloads = dict(self.payloads)
        if omit:
            del payloads[omit]
        buffer = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in payloads.items():
                    archive.writestr(name, b"changed" if name == wrong_member else data)
                archive.writestr("punkt_tab/README", b"Upstream notice")
                archive.writestr("punkt_tab/french/abbrev_types.txt", b"Unused")
                for name, data, mode in extra:
                    item = zipfile.ZipInfo(name)
                    # ZipInfo normalizes the host separator on Windows; retain
                    # a hostile archive name exactly for cross-platform tests.
                    item.filename = name
                    item.orig_filename = name
                    item.create_system = 3
                    item.external_attr = mode << 16
                    archive.writestr(item, data)
        data = buffer.getvalue()
        spec = {"url": setup.OFFICIAL_URL, "archive_relative_path": "tokenizers/punkt_tab.zip",
                "resource": "punkt_tab", "language": "english", "recorded_runtime_nltk_version": "3.9.2",
                "size_bytes": len(data), "sha256": sha(data), "max_uncompressed_bytes": 1024 * 1024,
                "files": [{"member": name, "size_bytes": len(payload), "sha256": sha(payload)}
                          for name, payload in self.payloads.items()]}
        archive = self.base / "official.zip"
        archive.write_bytes(data)
        return data, spec, archive

    def test_offline_install_retains_whole_official_zip_extracts_only_english(self):
        data, spec, archive = self.fixture()
        with patch.object(setup, "download_archive", side_effect=AssertionError("network forbidden")):
            result = setup.setup(self.destination, spec, archive=archive)
        self.assertTrue(result["passed"])
        self.assertEqual((self.destination / "tokenizers/punkt_tab.zip").read_bytes(), data)
        self.assertEqual(len([p for p in self.destination.rglob("*") if p.is_file()]), 5)
        for member, payload in self.payloads.items():
            self.assertEqual((self.destination / "tokenizers" / member).read_bytes(), payload)
        self.assertFalse((self.destination / "tokenizers/punkt_tab/french").exists())
        self.assertFalse((self.destination / "tokenizers/punkt_tab/README").exists())

    def test_download_install_then_idempotent_and_read_only_verification(self):
        data, spec, _ = self.fixture()
        with patch.object(setup, "download_archive", return_value=data) as download:
            setup.setup(self.destination, spec)
            mtimes = {str(p): p.stat().st_mtime_ns for p in self.destination.rglob("*")}
            setup.setup(self.destination, spec)
            result = setup.setup(self.destination, spec, verify_only=True)
        download.assert_called_once_with(spec)
        self.assertEqual(result["mode"], "offline_verify_only")
        self.assertEqual(mtimes, {str(p): p.stat().st_mtime_ns for p in self.destination.rglob("*")})

    def test_verify_missing_does_not_download_or_create(self):
        _, spec, _ = self.fixture()
        with patch.object(setup, "download_archive", side_effect=AssertionError("network forbidden")):
            with self.assertRaisesRegex(ValueError, "Missing"):
                setup.setup(self.destination, spec, verify_only=True)
        self.assertFalse(self.destination.exists())

    def test_mismatching_archive_is_not_extracted(self):
        data, spec, archive = self.fixture()
        archive.write_bytes(data[:-1] + b"X")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            setup.setup(self.destination, spec, archive=archive)
        self.assertFalse(self.destination.exists())

    def test_wrong_archive_size_is_not_extracted(self):
        data, spec, archive = self.fixture()
        archive.write_bytes(data + b"X")
        with self.assertRaisesRegex(ValueError, "Size mismatch"):
            setup.setup(self.destination, spec, archive=archive)
        self.assertFalse(self.destination.exists())

    def test_existing_mismatching_english_is_preserved_no_new_files(self):
        _, spec, archive = self.fixture()
        target = self.destination / "tokenizers/punkt_tab/english/abbrev_types.txt"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"preserve original")
        with self.assertRaises(ValueError):
            setup.setup(self.destination, spec, archive=archive)
        self.assertEqual(target.read_bytes(), b"preserve original")
        self.assertEqual(len([p for p in self.destination.rglob("*") if p.is_file()]), 1)

    def test_existing_bad_archive_is_preserved_instead_of_redownloaded(self):
        _, spec, _ = self.fixture()
        target = self.destination / "tokenizers/punkt_tab.zip"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"original bad archive")
        with patch.object(setup, "download_archive", side_effect=AssertionError("network forbidden")):
            with self.assertRaises(ValueError):
                setup.setup(self.destination, spec)
        self.assertEqual(target.read_bytes(), b"original bad archive")

    def test_unregistered_existing_files_and_directories_are_preserved(self):
        _, spec, archive = self.fixture()
        self.destination.mkdir()
        unrelated = self.destination / "unrelated.txt"
        unrelated.write_bytes(b"do not delete")
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            setup.setup(self.destination, spec, archive=archive)
        self.assertEqual(unrelated.read_bytes(), b"do not delete")

    def test_partial_valid_install_can_complete_without_overwrite(self):
        _, spec, archive = self.fixture()
        name, payload = next(iter(self.payloads.items()))
        target = self.destination / "tokenizers" / name
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)
        before = target.stat().st_mtime_ns
        setup.setup(self.destination, spec, archive=archive)
        self.assertEqual(target.stat().st_mtime_ns, before)

    def test_checkout_and_ancestor_destinations_rejected(self):
        for destination in (ROOT, ROOT / "private_nltk", ROOT.parent, Path(ROOT.anchor)):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                setup.destination_path(destination)

    def test_original_server_data_and_descendants_rejected(self):
        if __import__('os').name != 'posix':
            self.skipTest("POSIX absolute original-data path is checked on Linux")
        for destination in ("/123090274/SN_Research", "/123090274/SN_Research/anything", "/123090274"):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                setup.destination_path(destination)

    def test_destination_or_ancestor_symlink_rejected(self):
        real = self.base / "real"
        real.mkdir()
        link = self.base / "link"
        try:
            link.symlink_to(real, target_is_directory=True)
        except OSError:
            self.skipTest("Symlink creation unavailable to this Windows account")
        for destination in (link, link / "child"):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "symlink"):
                setup.destination_path(destination)

    def test_zip_path_traversal_absolute_backslash_drive_or_empty_segment_rejected(self):
        for member in ("../escape", "/absolute", "punkt_tab/../../escape", "C:/escape", "punkt_tab\\escape", "a//b", "./a"):
            with self.subTest(member=member):
                _, spec, archive = self.fixture(extra=[(member, b"bad", stat.S_IFREG | 0o644)])
                with self.assertRaisesRegex(ValueError, "Unsafe ZIP"):
                    setup.setup(self.destination, spec, archive=archive)
                self.assertFalse(self.destination.exists())

    def test_zip_symlink_or_special_file_rejected(self):
        for mode in (stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o600, stat.S_IFCHR | 0o600):
            with self.subTest(mode=mode):
                _, spec, archive = self.fixture(extra=[("punkt_tab/unwanted", b"bad", mode)])
                with self.assertRaisesRegex(ValueError, "symlink or special"):
                    setup.setup(self.destination, spec, archive=archive)
                self.assertFalse(self.destination.exists())

    def test_zip_duplicate_or_case_ambiguous_member_rejected(self):
        for member in ("punkt_tab/README", "PUNKT_TAB/README"):
            with self.subTest(member=member):
                _, spec, archive = self.fixture(extra=[(member, b"duplicate", stat.S_IFREG | 0o644)])
                with self.assertRaisesRegex(ValueError, "Duplicate/ambiguous"):
                    setup.setup(self.destination, spec, archive=archive)
                self.assertFalse(self.destination.exists())

    def test_zip_missing_english_member_rejected(self):
        _, spec, archive = self.fixture(omit="punkt_tab/english/abbrev_types.txt")
        with self.assertRaisesRegex(ValueError, "four required"):
            setup.setup(self.destination, spec, archive=archive)
        self.assertFalse(self.destination.exists())

    def test_zip_english_member_mismatch_rejected_before_write(self):
        _, spec, archive = self.fixture(wrong_member="punkt_tab/english/abbrev_types.txt")
        with self.assertRaisesRegex(ValueError, "English member size|SHA-256"):
            setup.setup(self.destination, spec, archive=archive)
        self.assertFalse(self.destination.exists())

    def test_zip_uncompressed_budget_enforced(self):
        _, spec, archive = self.fixture()
        spec["max_uncompressed_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "uncompressed-size"):
            setup.setup(self.destination, spec, archive=archive)
        self.assertFalse(self.destination.exists())

    def test_inconsistent_zip_directory_type_rejected(self):
        _, spec, archive = self.fixture(extra=[("punkt_tab/not-a-directory", b"bad", stat.S_IFDIR | 0o755)])
        with self.assertRaisesRegex(ValueError, "Inconsistent ZIP directory"):
            setup.setup(self.destination, spec, archive=archive)
        self.assertFalse(self.destination.exists())

    def test_verify_only_rejects_unused_archive_argument(self):
        _, spec, archive = self.fixture()
        setup.setup(self.destination, spec, archive=archive)
        with self.assertRaisesRegex(ValueError, "not used"):
            setup.setup(self.destination, spec, archive=archive, verify_only=True)

    def test_checked_manifest_names_and_archive_pin_are_exact(self):
        spec = setup.load_spec()
        self.assertEqual(spec["sha256"], "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106")
        self.assertEqual(spec["size_bytes"], 4319076)
        self.assertEqual(len(spec["files"]), 4)
        self.assertEqual(spec["recorded_runtime_nltk_version"], "3.9.2")

    def test_download_size_checksum_and_secure_origin_checks(self):
        data, spec, _ = self.fixture()
        class Response(io.BytesIO):
            def geturl(self):
                return setup.OFFICIAL_URL
        with patch.object(setup.urllib.request, "urlopen", return_value=Response(data)):
            self.assertEqual(setup.download_archive(spec), data)
        with patch.object(setup.urllib.request, "urlopen", return_value=Response(data + b"extra")):
            with self.assertRaisesRegex(ValueError, "Size mismatch"):
                setup.download_archive(spec)
        class InsecureResponse(Response):
            def geturl(self):
                return "http://untrusted.example/resource.zip"
        with patch.object(setup.urllib.request, "urlopen", return_value=InsecureResponse(data)):
            with self.assertRaisesRegex(ValueError, "redirect"):
                setup.download_archive(spec)

    def test_cli_success_and_structured_failure(self):
        _, spec, archive = self.fixture()
        with patch.object(setup, "load_spec", return_value=spec):
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = setup.main(["--destination", str(self.destination), "--archive", str(archive)])
            self.assertEqual(result, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])
            with redirect_stderr(stderr):
                result = setup.main(["--destination", str(self.base / "missing"), "--verify-only"])
            self.assertEqual(result, 1)
            self.assertFalse(json.loads(stderr.getvalue())["passed"])


if __name__ == "__main__":
    unittest.main()
