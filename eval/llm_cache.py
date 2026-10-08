"""Record/replay cache for LLM calls so the billed eval legs reproduce offline.

Modes:
  off     - no caching; call the model live (default; original behavior)
  record  - call the model live AND persist each (model, prompt) -> output
  replay  - read outputs from the cache; never call the model, no API key needed

The cache keys on (model, output-type, exact prompt), so it is tied to the
inputs: regenerate it whenever the gold set or prompts change. Downstream
metrics (kappa, bootstrap CIs, reliability) are seeded and deterministic, so a
replay run reproduces the reported numbers exactly with zero API cost.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


class _Result:
    """Minimal stand-in for a pydantic-ai run result (exposes `.output`)."""

    __slots__ = ("output",)

    def __init__(self, output: BaseModel) -> None:
        self.output = output


class LLMCache:
    """A content-addressed cache of LLM outputs, used to record then replay."""

    def __init__(self, path: Path, mode: str = "off") -> None:
        if mode not in {"off", "record", "replay"}:
            raise ValueError(f"unknown cache mode: {mode!r}")
        self.path = Path(path)
        self.mode = mode
        self._data: dict[str, dict] = {}
        if mode == "replay":
            if not self.path.exists():
                raise SystemExit(
                    f"--replay needs a cache at {self.path}, but it is missing.\n"
                    "Record one first with: python -m eval.harness --record"
                )
            self._data = json.loads(self.path.read_text())

    @staticmethod
    def _key(model: str, output_type: type, prompt: str) -> str:
        h = hashlib.sha256()
        for part in (model, output_type.__name__, prompt):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def run(self, agent, prompt: str, *, model: str, output_type: type[M]) -> _Result:
        """Return a result with `.output` of type `output_type`.

        replay: reconstruct from cache (no API call). record: call live + store.
        off: call live, store nothing.
        """
        key = self._key(model, output_type, prompt)
        if self.mode == "replay":
            entry = self._data.get(key)
            if entry is None:
                raise SystemExit(
                    f"replay cache miss (model={model}, type={output_type.__name__}).\n"
                    "The cache is stale vs the current inputs — re-record with --record."
                )
            return _Result(output_type.model_validate(entry["output"]))

        result = agent.run_sync(prompt)
        if self.mode == "record":
            self._data[key] = {
                "model": model,
                "output_type": output_type.__name__,
                "prompt": prompt,
                "output": result.output.model_dump(mode="json"),
            }
        return _Result(result.output)

    def save(self) -> None:
        """Persist the cache (record mode only)."""
        if self.mode != "record":
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True) + "\n")

    def __len__(self) -> int:
        return len(self._data)
