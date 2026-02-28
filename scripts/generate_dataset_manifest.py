#!/usr/bin/env python3
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("GEMMPALI_HDD_ROOT", "/mnt/ripped_media/GemmPali/datasets/raw"))
OUT = Path(os.environ.get("GEMMPALI_MANIFEST_OUT", "/mnt/ripped_media/GemmPali/datasets/manifests/dataset_manifest_latest.json"))
HF = os.environ.get("HF_BIN", str(Path.home() / ".local/bin/hf"))

DATASETS = [
    ("vidore/colpali_train_set", "vidore-colpali-train-set"),
    ("vidore/docvqa_train", "vidore-docvqa-train"),
    ("AHS-uni/mpdocvqa-corpus", "mpdocvqa-corpus"),
    ("AHS-uni/mpdocvqa-qa", "mpdocvqa-qa"),
    ("AHS-uni/dude-corpus", "dude-corpus"),
    ("AHS-uni/dude-qa", "dude-qa"),
]


def du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            pass
    return total


def hf_info(repo_id: str):
    last_err = None
    for _ in range(3):
        try:
            r = subprocess.run([HF, "datasets", "info", repo_id], capture_output=True, text=True, check=True)
            info = json.loads(r.stdout)
            return {"sha": info.get("sha"), "last_modified": info.get("last_modified")}
        except Exception as e:
            last_err = e
    return {"sha": None, "last_modified": None, "error": str(last_err)}


def download_running(repo_id: str) -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", "hf download"], text=True)
        return repo_id in out
    except subprocess.CalledProcessError:
        return False


records = []
for repo_id, local_name in DATASETS:
    p = ROOT / local_name
    size = du_bytes(p)
    info = hf_info(repo_id)
    running = download_running(repo_id)
    status = "in_progress" if running else ("present" if p.exists() and size > 0 else "missing")
    records.append(
        {
            "repo_id": repo_id,
            "local_dir": str(p),
            "bytes": size,
            "status": status,
            "sha": info.get("sha"),
            "last_modified": info.get("last_modified"),
            "error": info.get("error"),
        }
    )

manifest = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "root": str(ROOT),
    "datasets": records,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(manifest, indent=2))
print(str(OUT))
print(json.dumps(manifest, indent=2))
