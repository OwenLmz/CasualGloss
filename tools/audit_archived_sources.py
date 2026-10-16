"""Static, non-executing integrity and dependency audit of archived originals."""
from __future__ import annotations
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/"archive/original"
MANIFEST=ROOT/"provenance/source_manifest.json"
EXTERNAL={
"numpy","torch","transformers","peft","nltk","simalign","matplotlib","sacrebleu",
"pandas","tqdm","scipy","soundfile","librosa","torchaudio","sentencepiece",
"qwen_tts","faster_whisper","whisper","pydub","seamless_communication",
"fairseq2","simuleval","datasets","sentence_transformers","jiwer",
"yaml","PIL","regex","sklearn","huggingface_hub","accelerate",
}
def rel(p): return p.relative_to(ROOT).as_posix()
def main():
    manifest=json.loads(MANIFEST.read_text("utf-8"))
    integrity=[]
    sensitive_locations=[]
    credential_pattern=re.compile(
        r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|"
        r"github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|"
        r"AKIA[A-Z0-9]{16}|BEGIN [A-Z ]*PRIVATE KEY|"
        r"https?://[^/\s:@]+:[^/\s@]+@"
    )
    for item in manifest["files"]:
        p=ROOT/item["archive_path"]
        raw=p.read_bytes()
        if len(raw)!=item["size_bytes"] or hashlib.sha256(raw).hexdigest()!=item["sha256"]:
            integrity.append(item["archive_path"])
        for number,line in enumerate(raw.decode("utf-8-sig",errors="replace").splitlines(),1):
            if credential_pattern.search(line):
                sensitive_locations.append({"file":item["archive_path"],"line":number})
    python_files=sorted(ARCHIVE.rglob("*.py"))
    basename=defaultdict(list)
    for p in python_files:
        basename[p.stem].append(p)
    import_records=[]
    parse_errors=[]
    dynamic_paths=[]
    script_refs=[]
    by_name=defaultdict(list)
    for p in ARCHIVE.rglob("*"):
        if p.is_file():
            by_name[p.name].append(rel(p))
    for p in python_files:
        source=p.read_text("utf-8-sig")
        try:
            tree=ast.parse(source,filename=str(p))
        except SyntaxError as err:
            parse_errors.append({"path":rel(p),"line":err.lineno,"error":err.msg})
            continue
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                modules=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                modules=[node.module] if node.module else []
            else:
                continue
            for module in modules:
                root=module.split(".")[0]
                same=p.parent/(root+".py")
                if same.exists():
                    state="local_same_directory"
                    candidates=[rel(same)]
                elif root in basename:
                    state="local_cross_directory_archive_present"
                    candidates=[rel(x) for x in basename[root]]
                elif root in sys.stdlib_module_names:
                    state="python_standard_library"
                    candidates=[]
                elif root in EXTERNAL:
                    state="external_package"
                    candidates=[]
                else:
                    state="unresolved_or_external"
                    candidates=[]
                import_records.append({"file":rel(p),"line":node.lineno,"module":module,
                    "classification":state,"archive_candidates":candidates})
    for p in sorted(ARCHIVE.rglob("*")):
        if not p.is_file() or p.suffix not in (".py",".sh",".json",".toml",".yaml",".yml",".patch"):
            continue
        text=p.read_text("utf-8-sig",errors="replace")
        for no,line in enumerate(text.splitlines(),1):
            markers=[k for k in ("/123090274/","D:/","D:\\","C:/","C:\\",
                ".codex","/venvs/","third_party","_deps","vendor","sys.path.insert",
                "spec_from_file_location","audit_panel_alignment") if k in line]
            if markers:
                dynamic_paths.append({"file":rel(p),"line":no,"markers":markers,
                    "text":line.strip()[:450]})
            for filename in sorted(set(re.findall(r"(?<![\w.-])([A-Za-z_][\w.-]*\.(?:py|sh))\b",line))):
                state="archived" if filename in by_name else "missing_or_external_reference"
                script_refs.append({"file":rel(p),"line":no,"filename":filename,
                    "classification":state,"archive_candidates":by_name.get(filename,[])})
    result={"file_count":manifest["file_count"],"bytes":manifest["bytes"],
        "integrity_failures":integrity,"credential_pattern_locations":sensitive_locations,
        "python_file_count":len(python_files),
        "parse_errors":parse_errors,
        "import_classifications":dict(Counter(i["classification"] for i in import_records)),
        "imports":import_records,"path_portability_references":dynamic_paths,
        "script_references":script_refs,
        "unresolved_import_names":sorted(set(i["module"] for i in import_records
            if i["classification"]=="unresolved_or_external")),
        "missing_referenced_script_names":sorted(set(i["filename"] for i in script_refs
            if i["classification"]=="missing_or_external_reference"))}
    output=ROOT/"provenance/source_closure_static.json"
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("file_count","bytes","integrity_failures","credential_pattern_locations",
        "python_file_count","parse_errors","import_classifications","unresolved_import_names",
        "missing_referenced_script_names")},ensure_ascii=False,indent=2))
    if integrity or parse_errors or sensitive_locations:
        raise SystemExit(1)
if __name__=="__main__":
    main()

