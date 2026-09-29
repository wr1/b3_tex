"""Tests for optional FE stage timers (backends._profile)."""

from __future__ import annotations

from b3_tex.backends._profile import StageTimer, profiling_enabled


def test_profiling_enabled_from_env(monkeypatch):
    monkeypatch.delenv("B3_TEX_PROFILE", raising=False)
    assert profiling_enabled(None) is False
    assert profiling_enabled({}) is False
    assert profiling_enabled({"profile": False}) is False
    assert profiling_enabled({"profile": True}) is True

    monkeypatch.setenv("B3_TEX_PROFILE", "1")
    assert profiling_enabled(None) is True
    assert profiling_enabled({"profile": False}) is True  # env wins

    monkeypatch.setenv("B3_TEX_PROFILE", "true")
    assert profiling_enabled({}) is True
    monkeypatch.setenv("B3_TEX_PROFILE", "yes")
    assert profiling_enabled({}) is True
    monkeypatch.setenv("B3_TEX_PROFILE", "0")
    assert profiling_enabled({"profile": True}) is True
    assert profiling_enabled({}) is False


def test_stage_timer_disabled_is_noop():
    t = StageTimer(enabled=False)
    with t.stage("mesh"):
        pass
    assert t.stages == {}
    assert t.total() == 0.0
    assert t.as_dict() == {}


def test_stage_timer_accumulates():
    t = StageTimer(enabled=True)
    with t.stage("a"):
        pass
    with t.stage("b"):
        pass
    with t.stage("a"):
        pass  # accumulate
    d = t.as_dict()
    assert "a" in d and "b" in d and "total_s" in d
    assert d["a"] >= 0.0
    assert d["total_s"] >= d["a"]
    assert t.stages["a"] == d["a"] or abs(t.stages["a"] - d["a"]) < 1e-5
