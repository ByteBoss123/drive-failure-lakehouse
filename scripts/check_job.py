"""Validate the Databricks job definition: every task points at a notebook in databricks/,
dependencies refer to existing tasks, and there is no cycle."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
job = json.loads(Path(sys.argv[1]).read_text())
tasks = {t["task_key"]: t for t in job["tasks"]}
errors = []
for key, t in tasks.items():
    name = t["notebook_task"]["notebook_path"].rsplit("/", 1)[-1]
    if not (ROOT / "databricks" / f"{name}.py").exists():
        errors.append(f"{key}: notebook {name} not in databricks/")
    for dep in t.get("depends_on", []):
        if dep["task_key"] not in tasks:
            errors.append(f"{key}: unknown dependency {dep['task_key']}")

seen, order = set(), []


def visit(k, stack=()):
    if k in stack:
        errors.append(f"cycle at {k}")
        return
    if k in seen:
        return
    for dep in tasks[k].get("depends_on", []):
        if dep["task_key"] in tasks:
            visit(dep["task_key"], stack + (k,))
    seen.add(k)
    order.append(k)


for k in tasks:
    visit(k)
if errors:
    print("\n".join(errors))
    sys.exit(1)
print("job ok:", " -> ".join(order))
