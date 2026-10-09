"""A stand-in for the tvla binary. It records its arguments and writes small files."""

import json
import os
import sys
import time
from pathlib import Path

argv = sys.argv[1:]
log = os.environ.get("FAKE_TVLA_LOG")
if log:
    with open(log, "a") as fh:
        fh.write(json.dumps(argv) + "\n")


def value(flag):
    return argv[argv.index(flag) + 1]


if "--list-signals" in argv:
    print("0\tno\t$rootio.clk")
    print("1\tyes\ttop.clk")
    print("2\tyes\ttop.x")
    print("2 of 3 selectable signals are selected", file=sys.stderr)
    if os.environ.get("FAKE_TVLA_WARN"):
        print("warning: the rule +signal:nope matches no signal", file=sys.stderr)
elif "--stats-out" in argv:
    meta = json.loads(Path(value("--meta-json")).read_text())
    batch = meta["batch"]["id"]
    time.sleep(float(os.environ.get("FAKE_TVLA_SLEEP", "0")))
    if "--traces-out" in argv:  # like tvla: the traces file comes before the cache
        Path(value("--traces-out")).write_text(f"traces {batch}\n")
    if batch in os.environ.get("FAKE_TVLA_FAIL", "").split(","):
        print("boom", file=sys.stderr)
        sys.exit(3)
    Path(value("--stats-out")).write_text(f"stats {batch}\n")
elif "--merge-stats" in argv:
    i = argv.index("--merge-stats") + 1
    files = []
    while i < len(argv) and not argv[i].startswith("-"):
        files.append(argv[i])
        i += 1
    out = Path(value("--ttest-output-dir"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "merged.txt").write_text("".join(Path(f).read_text() for f in files))
else:
    sys.exit(4)
