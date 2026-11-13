"""Every file the static demo (and the README images) refer to must be tracked by git.

A previous .gitignore pattern hid docs/data from the repository, so the demo would have loaded with no data on GitHub.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def tracked() -> set[str]:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return set(out.splitlines())


def referenced() -> set[str]:
    refs: set[str] = set()
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    for m in re.finditer(r'(?:src|href)="([^"#?]+)"', html):
        if not re.match(r"(https?:|mailto:|data:|//)", m.group(1)):
            refs.add(f"docs/{m.group(1)}")
    refs |= {f"docs/js/{p.name}" for p in (DOCS / "js").glob("*.js")}
    data = DOCS / "data"
    for name in ("index.json", "convergence.json", "training.json", "fid.json", "fashion_real.webp"):
        refs.add(f"docs/data/{name}")
    for run in json.loads((data / "index.json").read_text(encoding="utf-8")):
        refs |= {f"docs/data/runs/{run['id']}.json", f"docs/data/runs/{run['id']}.webp"}
    report = json.loads((data / "training.json").read_text(encoding="utf-8"))
    for kind, by in report["samplers"].items():
        refs.add(f"docs/data/grid_{kind}.png")
        refs |= {f"docs/data/grid_{kind}_{key.replace('-', '')}.png" for key in by}
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for m in re.finditer(r'(?:\]\(|src=")((?:docs/)[^)" ]+)', readme):
        refs.add(m.group(1))
    return refs


def test_every_demo_file_is_tracked():
    files = tracked()
    missing = sorted(r for r in referenced() if r not in files)
    assert not missing, f"referenced but not tracked by git: {missing[:10]}"


def test_no_weights_are_tracked():
    assert not [f for f in tracked() if f.endswith((".pt", ".pth", ".safetensors", ".ckpt", ".bin"))]
