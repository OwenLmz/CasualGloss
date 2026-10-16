"""No-network safety tests; synthetic bytes only, no research inputs or secrets."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

MODULE=Path(__file__).resolve().parents[2]/"scripts/verification/fetch_verification_inputs.py"
spec=importlib.util.spec_from_file_location("fetch_verification_inputs_guards",MODULE)
fetch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class FetchGuardTest(unittest.TestCase):
    def test_host_rejects_options_whitespace_and_shell_characters(self):
        fetch.validate_transport("root@10.27.130.23",32654,4)
        for host in ("-oProxyCommand=x","--","root@host;id","root@host x","root@host\n",
                     "root@$(id)","root@`id`","root@host\\path","","@host","root@",
                     "a@@b","root@-option"):
            with self.subTest(host=host),self.assertRaises(ValueError):
                fetch.validate_transport(host,22,4)

    def test_port_and_workers_are_bounded_integers(self):
        for port in (0,-1,65536,True,"22"):
            with self.subTest(port=port),self.assertRaises(ValueError):
                fetch.validate_transport("user@host",port,4)
        for workers in (0,-1,17,True,"4"):
            with self.subTest(workers=workers),self.assertRaises(ValueError):
                fetch.validate_transport("user@host",22,workers)

    def test_target_paths_reject_traversal_drives_and_noncanonical_forms(self):
        for value in ("../file.jsonl","x/../../file","/absolute","C:/file","a\\b",
                      "a//b","./file",".","a/./b","bad path",""):
            with self.subTest(value=value),self.assertRaises(ValueError):
                fetch.validate_relative_path(value)
        self.assertEqual(fetch.validate_relative_path("a/b.jsonl").as_posix(),"a/b.jsonl")

    def test_publication_repository_destination_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=MODULE.parents[2].parent) as folder:
            repo=Path(folder)/"publication"
            repo.mkdir()
            with patch.object(fetch,"REPOSITORY_ROOT",repo):
                with self.assertRaises(ValueError):
                    fetch.validate_private_destination(repo/"private_data")

    def test_target_inside_publication_repository_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=MODULE.parents[2].parent) as folder:
            private=Path(folder)
            repo=private/"publication"
            repo.mkdir()
            with patch.object(fetch,"REPOSITORY_ROOT",repo):
                with self.assertRaises(ValueError):
                    fetch.safe_destination_file(private,"publication/record.jsonl")

    def test_private_cache_under_broader_workspace_git_is_allowed(self):
        with tempfile.TemporaryDirectory(dir=MODULE.parents[2].parent) as folder:
            workspace=Path(folder)
            (workspace/".git").mkdir()
            private=workspace/"private_cache"
            publication=workspace/"publication"
            with patch.object(fetch,"REPOSITORY_ROOT",publication):
                self.assertEqual(fetch.validate_private_destination(private),private)

    def test_resolved_link_escape_and_manifest_escape_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=MODULE.parents[2].parent) as folder:
            private=Path(folder)/"private"
            outside=Path(folder)/"outside"
            private.mkdir();outside.mkdir()
            original_resolve=Path.resolve
            for relative in ("linked/record.jsonl","verification_input_manifest.json"):
                target=private/relative
                def fake_resolve(path,*args,**kwargs):
                    if path==target:return outside/"escaped.jsonl"
                    return original_resolve(path,*args,**kwargs)
                with patch.object(Path,"resolve",fake_resolve):
                    with self.subTest(relative=relative),self.assertRaises(ValueError):
                        fetch.safe_destination_file(private,relative)

    def test_existing_nonmatching_input_is_not_overwritten_or_verified(self):
        with tempfile.TemporaryDirectory(dir=MODULE.parents[2].parent) as folder:
            private=Path(folder)
            (private/"record.jsonl").write_bytes(b"wrong")
            expected=hashlib.sha256(b"expected").hexdigest()
            with patch.object(fetch.subprocess,"run") as run:
                with self.assertRaises(FileExistsError):
                    fetch.fetch_one(("record.jsonl","record.jsonl"),private,Path("key"),"user@host",22,expected)
                run.assert_not_called()
            self.assertEqual((private/"record.jsonl").read_bytes(),b"wrong")

    def test_existing_matching_input_is_verified_without_network(self):
        with tempfile.TemporaryDirectory(dir=MODULE.parents[2].parent) as folder:
            private=Path(folder)
            (private/"record.jsonl").write_bytes(b"expected")
            expected=hashlib.sha256(b"expected").hexdigest()
            with patch.object(fetch.subprocess,"run") as run:
                result=fetch.fetch_one(("record.jsonl","record.jsonl"),private,Path("key"),"user@host",22,expected)
                run.assert_not_called()
            self.assertTrue(result["remote_hash_matches"])
            self.assertEqual(result["transfer_status"],"already_verified")

    def test_downloaded_hash_mismatch_is_preserved_but_not_verified(self):
        with tempfile.TemporaryDirectory(dir=MODULE.parents[2].parent) as folder:
            private=Path(folder)
            expected=hashlib.sha256(b"expected").hexdigest()
            def fake_scp(command,**kwargs):
                Path(command[-1]).write_bytes(b"corrupt-download")
                return subprocess.CompletedProcess(command,0,"","")
            with patch.object(fetch.subprocess,"run",fake_scp):
                with self.assertRaises(ValueError):
                    fetch.fetch_one(("record.jsonl","record.jsonl"),private,Path("key"),"user@host",22,expected)
            self.assertEqual((private/"record.jsonl").read_bytes(),b"corrupt-download")

    def test_remote_hash_response_must_be_complete_valid_and_exact(self):
        targets=[("record.jsonl","record.jsonl")]
        remote=fetch.REMOTE_ROOT+"/record.jsonl"
        h=hashlib.sha256(b"expected").hexdigest()
        for stdout in ("",f"invalid  {remote}\n",f"{h}  /unexpected\n",f"{h}  {remote}\n{h}  {remote}\n"):
            response=subprocess.CompletedProcess([],0,stdout,"")
            with patch.object(fetch.subprocess,"run",return_value=response):
                with self.subTest(stdout=stdout),self.assertRaises(ValueError):
                    fetch.read_remote_hashes(targets,Path("key"),"user@host",22)
        with patch.object(fetch.subprocess,"run",return_value=subprocess.CompletedProcess([],0,f"{h}  {remote}\n","")):
            self.assertEqual(fetch.read_remote_hashes(targets,Path("key"),"user@host",22),{remote:h})


if __name__=="__main__":unittest.main()
