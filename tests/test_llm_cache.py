"""Tests for the record/replay LLM cache (eval/llm_cache.py)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from eval.llm_cache import LLMCache


class _Out(BaseModel):
    x: int
    s: str


class _FakeResult:
    def __init__(self, output):
        self.output = output


class _FakeAgent:
    """Stand-in for a pydantic-ai Agent; counts live calls."""

    def __init__(self):
        self.calls = 0

    def run_sync(self, prompt):
        self.calls += 1
        return _FakeResult(_Out(x=len(prompt), s=prompt.upper()))


def test_record_then_replay_round_trip(tmp_path):
    path = tmp_path / "cache.json"

    rec = LLMCache(path, mode="record")
    agent = _FakeAgent()
    r = rec.run(agent, "hello", model="groq:test", output_type=_Out)
    assert r.output.x == 5 and r.output.s == "HELLO"
    assert agent.calls == 1
    rec.save()
    assert path.exists()

    # Replay: identical output, and the model is never called.
    rep = LLMCache(path, mode="replay")
    agent2 = _FakeAgent()
    r2 = rep.run(agent2, "hello", model="groq:test", output_type=_Out)
    assert r2.output.x == 5 and r2.output.s == "HELLO"
    assert agent2.calls == 0


def test_replay_miss_raises(tmp_path):
    path = tmp_path / "cache.json"
    LLMCache(path, mode="record").save()  # empty cache on disk

    rep = LLMCache(path, mode="replay")
    with pytest.raises(SystemExit):
        rep.run(_FakeAgent(), "not-recorded", model="m", output_type=_Out)


def test_replay_missing_cache_file_raises(tmp_path):
    with pytest.raises(SystemExit):
        LLMCache(tmp_path / "does_not_exist.json", mode="replay")


def test_off_mode_is_live_passthrough_without_persisting(tmp_path):
    path = tmp_path / "cache.json"
    off = LLMCache(path, mode="off")
    agent = _FakeAgent()
    r = off.run(agent, "hi", model="m", output_type=_Out)
    assert r.output.x == 2
    assert agent.calls == 1
    off.save()
    assert not path.exists()  # off mode writes nothing


def test_key_distinguishes_model_and_prompt(tmp_path):
    rec = LLMCache(tmp_path / "c.json", mode="record")
    k1 = rec._key("groq:a", _Out, "p")
    k2 = rec._key("groq:b", _Out, "p")
    k3 = rec._key("groq:a", _Out, "q")
    assert len({k1, k2, k3}) == 3
