#!/usr/bin/env python3
"""resumen.py - agrega results/*.events.jsonl y *.json en una tabla y un CSV."""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "results"


def read_text(path):
    try:
        return open(path).read().strip()
    except OSError:
        return ""


def peak(events, key):
    vals = [e[key] for e in events if e.get(key) is not None]
    return max(vals) if vals else None


rows = []
for ev_path in sorted(glob.glob(os.path.join(d, "*.events.jsonl"))):
    tag = os.path.basename(ev_path)[: -len(".events.jsonl")]
    events = [json.loads(line) for line in open(ev_path) if line.strip()]
    res_path = os.path.join(d, tag + ".json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else None
    exit_code = read_text(os.path.join(d, tag + ".exit"))
    oom = read_text(os.path.join(d, tag + ".oom"))

    if res is None:
        status = "killed"  # el proceso murio sin escribir el .json (SIGKILL/OOM)
        last = [e["stage"] for e in events][-1] if events else "none"
    else:
        status = res["status"]
        last = res.get("failed_after_stage") or "end"

    rows.append({
        "tag": tag,
        "group": tag.rsplit("_r", 1)[0],
        "status": status,
        "last_stage": last,
        "exit": exit_code,
        "oom_killed": oom,
        "val_acc": (res or {}).get("final_val_acc"),
        "train_s": (res or {}).get("train_time_s"),
        "cg_limit_mb": (res or {}).get("cg_limit_mb"),
        "cg_peak_mb": peak(events, "cg_peak_sampled_mb"),
        "cg_anon_peak_mb": peak(events, "cg_anon_peak_sampled_mb"),
        "cg_peak_kernel_mb": peak(events, "cg_peak_kernel_mb"),
        "rss_peak_mb": peak(events, "rss_peak_mb"),
        "cuda_max_alloc_mb": peak(events, "cuda_max_alloc_mb"),
    })

if not rows:
    sys.exit(f"Sin resultados en {d}/")

cols = list(rows[0].keys())
with open(os.path.join(d, "resumen.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

show = ["tag", "status", "last_stage", "exit", "oom_killed", "val_acc", "train_s",
        "cg_peak_mb", "cg_anon_peak_mb", "rss_peak_mb", "cuda_max_alloc_mb"]
widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in show}
print("  ".join(c.ljust(widths[c]) for c in show))
for r in rows:
    print("  ".join(str(r[c]).ljust(widths[c]) for c in show))

print("\nPor configuracion (ok / total):")
g = defaultdict(lambda: [0, 0])
for r in rows:
    g[r["group"]][1] += 1
    g[r["group"]][0] += r["status"] == "ok"
for k, (ok, n) in g.items():
    print(f"  {k:<18} {ok}/{n}")
