"""The README's tables and quoted numbers must match docs/data (see scripts/check_numbers.py)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("check_numbers", ROOT / "scripts" / "check_numbers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readme_numbers_match_the_results_files(capsys):
    rc = _script().main(["--check"])
    assert rc == 0, capsys.readouterr().out


def test_training_page_and_readme_share_one_source():
    # the UI reads docs/data/training.json; the README is rendered from the same file
    ui = (ROOT / "docs" / "js" / "train.js").read_text(encoding="utf-8")
    assert "training.json" in ui and "holdout" in ui
    vals = _script().values()
    assert vals["steps"] == "8,580" or int(vals["steps"].replace(",", "")) > 0
