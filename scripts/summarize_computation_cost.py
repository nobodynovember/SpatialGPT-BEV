#!/usr/bin/env python3
"""Generate the reviewer-ready computation cost table and response."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


MODULES = ["bev_build", "dcll_skg_update", "synchronize", "align", "backtrack"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_events(path):
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not events:
        return []
    session = events[-1].get("session_id")
    return [event for event in events if event.get("session_id") == session]


def build_report(events):
    times = defaultdict(lambda: defaultdict(float))
    api = defaultdict(list)
    for event in events:
        module = event.get("module")
        key = (event.get("scan"), event.get("viewpoint"), event.get("navigation_step"))
        if event.get("event") == "module" and event.get("success", True):
            times[module][key] += float(event.get("wall_clock_seconds", 0))
        elif event.get("event") == "gpt_api":
            api[module].append(event)
    viewpoints = len({key for module in MODULES for key in times[module] if key[1] is not None})
    rows = []
    for module in MODULES:
        values = list(times[module].values())
        calls = api[module]
        avg = lambda xs: statistics.fmean(xs) if xs else 0.0
        rows.append((module, len(values), len(values) / viewpoints if viewpoints else 0,
                     avg(values), sum(values) / viewpoints if viewpoints else 0,
                     len(calls), avg([float(x.get("wall_clock_seconds", 0)) for x in calls]),
                     avg([int(x.get("prompt_tokens", 0)) for x in calls]),
                     avg([int(x.get("completion_tokens", 0)) for x in calls]),
                     sum(int(x.get("total_tokens", 0)) for x in calls)))
    lines = ["# Computational Cost", "", f"Statistics cover **{viewpoints} viewpoints** from the latest run.", "",
             "| Major module | Invocations | Calls/viewpoint | Latency/call (s) | Amortized s/viewpoint | GPT calls | GPT latency/call (s) | Prompt tokens/call | Completion tokens/call | Total tokens |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]} | {r[6]:.3f} | {r[7]:.1f} | {r[8]:.1f} | {r[9]} |")
    total_seconds = sum(sum(times[module].values()) for module in MODULES)
    total_calls = sum(len(api[module]) for module in MODULES)
    total_tokens = sum(int(call.get("total_tokens", 0)) for module in MODULES for call in api[module])
    lines += ["", "Reviewer response:", "",
              "We report computational cost at the viewpoint level for the five major modules: BEV construction, DCLL/SKG update, synchronization, alignment, and backtracking. Conditional latency is measured only when a module is invoked, while amortized latency divides its total wall-clock time by all " + str(viewpoints) + " evaluated viewpoints. Across this run, these modules used " + f"{total_seconds:.3f}" + " seconds in total, including " + str(total_calls) + " GPT API calls and " + str(total_tokens) + " tokens. The table reports module-level API latency, call count, and prompt/completion token usage.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=str(PROJECT_ROOT / "computation_cost.log"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "computation_cost_response.md"))
    args = parser.parse_args()
    events = load_events(args.log)
    if not events:
        raise SystemExit(f"No valid computation-cost events found in {args.log}")
    report = build_report(events)
    Path(args.output).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
