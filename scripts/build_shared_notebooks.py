"""Generate the Databricks shared-code notebooks from the tested lakehouse package.

Databricks notebooks share code with %run, so lakehouse/*.py is copied into
databricks/lakehouse_*.py with intra-package imports removed (00_config %runs them
in dependency order). CI runs this with --check to fail if they drift from the source.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ("# Databricks notebook source\n"
          "# Generated from lakehouse/{name}.py by scripts/build_shared_notebooks.py. Do not edit.\n")


def render(name: str) -> str:
    src = (ROOT / "lakehouse" / f"{name}.py").read_text()
    src = re.sub(r"^from lakehouse\.\w+ import .*\n", "", src, flags=re.M)
    return HEADER.format(name=name) + src


def render_reference() -> str:
    import json
    pipe = json.loads((ROOT / "reports" / "pipeline_metrics.json").read_text())
    pipe.pop("incremental_merges", None)
    model = json.loads((ROOT / "reports" / "model_metrics.json").read_text())
    scoring = json.loads((ROOT / "reports" / "scoring_metrics.json").read_text())
    return ("# Databricks notebook source\n# Generated from reports/*.json by "
            "scripts/build_shared_notebooks.py. Do not edit.\n"
            f"REF = {json.dumps(pipe)}\nREF_MODEL = {json.dumps(model)}\nREF_SCORING = {json.dumps(scoring)}\n")


def render_manifest() -> str:
    import csv
    import hashlib
    rows = sorted(csv.DictReader((ROOT / "data" / "raw_manifest.csv").open()), key=lambda r: r["file"])
    digest = hashlib.sha256("".join(r["sha256"] for r in rows).encode()).hexdigest()
    names = [r["file"] for r in rows]
    return ("# Databricks notebook source\n# Generated from data/raw_manifest.csv by "
            "scripts/build_shared_notebooks.py. Do not edit.\n"
            f"MANIFEST_FILES = {json_list(names)}\n"
            "# sha256 over the concatenated per-file sha256 values, sorted by file name\n"
            f'MANIFEST_DIGEST = "{digest}"\n')


def json_list(items):
    import json
    return json.dumps(items)


def outputs():
    yield "lakehouse_transforms.py", render("transforms")
    yield "lakehouse_model.py", render("model")
    yield "lakehouse_monitoring.py", render("monitoring")
    yield "reference_metrics.py", render_reference()
    yield "raw_manifest.py", render_manifest()


def main(check: bool) -> int:
    stale = []
    for fname, text in outputs():
        out = ROOT / "databricks" / fname
        if check:
            if not out.exists() or out.read_text() != text:
                stale.append(out.name)
        else:
            out.write_text(text)
    if stale:
        print("stale generated notebooks:", ", ".join(stale), "- run scripts/build_shared_notebooks.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
