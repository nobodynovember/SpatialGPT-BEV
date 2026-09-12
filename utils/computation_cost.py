"""Process-safe JSONL tracing for per-viewpoint computation cost."""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import time
import uuid
from datetime import datetime, timezone


_phase = contextvars.ContextVar("computation_cost_phase", default="unclassified")
_metadata = contextvars.ContextVar("computation_cost_metadata", default={})


def _log_path():
    return os.environ.get("COMPUTATION_COST_LOG", "computation_cost.log")


def log_event(event, **fields):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": os.environ.get("COMPUTATION_COST_SESSION", "unknown"),
        "pid": os.getpid(), "event": event, **_metadata.get(), **fields,
    }
    line = json.dumps(record, sort_keys=True, default=str) + "\n"
    path = _log_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def start_session(log_path, reset=True):
    os.environ["COMPUTATION_COST_LOG"] = os.path.abspath(log_path)
    os.environ["COMPUTATION_COST_SESSION"] = uuid.uuid4().hex
    if reset:
        with open(os.environ["COMPUTATION_COST_LOG"], "w", encoding="utf-8"):
            pass
    log_event("session_start")


@contextlib.contextmanager
def computation_phase(module, **metadata):
    phase_token = _phase.set(module)
    metadata_token = _metadata.set({**_metadata.get(), **metadata})
    try:
        yield
    finally:
        _metadata.reset(metadata_token)
        _phase.reset(phase_token)


@contextlib.contextmanager
def timed_module(module, **metadata):
    start = time.perf_counter()
    ok = False
    with computation_phase(module, **metadata):
        try:
            yield
            ok = True
        finally:
            log_event("module", module=module,
                      wall_clock_seconds=time.perf_counter() - start,
                      success=ok)


def record_gpt_call(wall_clock_seconds, usage, model, image_count,
                    success=True, error=None):
    def value(*names):
        for name in names:
            item = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None) if usage is not None else None
            if item is not None:
                try:
                    return int(item)
                except (TypeError, ValueError):
                    pass
        return 0

    prompt_tokens = value("prompt_tokens", "input_tokens")
    completion_tokens = value("completion_tokens", "output_tokens")
    total_tokens = value("total_tokens") or prompt_tokens + completion_tokens
    log_event("gpt_api", module=_phase.get(), model=model,
              image_count=image_count, wall_clock_seconds=wall_clock_seconds,
              prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
              total_tokens=total_tokens, success=success, error=error)
