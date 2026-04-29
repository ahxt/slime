"""Tests for Sample.policy_name (Step 1 of multi-policy fork).

The field is the smallest possible change to the existing Sample dataclass:
optional str, default None, preserves all legacy behavior.

Run with:
    python -m pytest tests/test_sample_policy_name.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from slime.utils.types import Sample


# ── Field presence ──────────────────────────────────────────────────────────


def test_field_in_dataclass():
    assert "policy_name" in Sample.__dataclass_fields__


def test_default_is_none():
    """Default None preserves legacy single-policy behavior — manager's _split_by_policy
    routes None-tagged samples through the bit-for-bit identical legacy buffer path."""
    assert Sample().policy_name is None


def test_set_via_constructor():
    s = Sample(prompt="x", policy_name="solver")
    assert s.policy_name == "solver"


def test_set_via_attribute():
    s = Sample(prompt="x")
    s.policy_name = "rewriter"
    assert s.policy_name == "rewriter"


# ── Round-trip via to_dict / from_dict ──────────────────────────────────────


def test_to_dict_includes_policy_name():
    s = Sample(prompt="x", policy_name="selector")
    d = s.to_dict()
    assert d["policy_name"] == "selector"


def test_to_dict_includes_default_none():
    s = Sample(prompt="x")
    d = s.to_dict()
    assert d["policy_name"] is None


def test_from_dict_recovers_policy_name():
    s = Sample(prompt="x", policy_name="solver")
    s2 = Sample.from_dict(s.to_dict())
    assert s2.policy_name == "solver"


def test_from_dict_default_none():
    s = Sample(prompt="x")
    s2 = Sample.from_dict(s.to_dict())
    assert s2.policy_name is None


def test_from_dict_missing_key_treated_as_none():
    """Older serialized samples (pre-fork) won't have the key; should default to None
    so legacy checkpoints/logs still load."""
    legacy_dict = {
        "prompt": "x",
        "tokens": [],
        "weight_versions": [],
        "metadata": {},
        "status": "pending",
        "spec_info": {},
        "prefix_cache_info": {},
    }
    s = Sample.from_dict(legacy_dict)
    assert s.policy_name is None


# ── Back-compat: existing Sample usage unaffected ───────────────────────────


def test_existing_fields_still_default_correctly():
    s = Sample()
    # Spot-check a sample of existing fields
    assert s.group_index is None
    assert s.index is None
    assert s.prompt == ""
    assert s.tokens == []
    assert s.response == ""
    assert s.reward is None
    assert s.status == Sample.Status.PENDING
    assert s.session_id is None
    assert s.non_generation_time == 0.0


def test_existing_round_trip_unaffected():
    """A Sample without policy_name set should round-trip exactly as before."""
    s = Sample(
        prompt="hello",
        response="world",
        response_length=2,
        reward=0.5,
        status=Sample.Status.COMPLETED,
        session_id="abc",
    )
    s2 = Sample.from_dict(s.to_dict())
    assert s2.prompt == "hello"
    assert s2.response == "world"
    assert s2.response_length == 2
    assert s2.reward == 0.5
    assert s2.status == Sample.Status.COMPLETED
    assert s2.session_id == "abc"
    assert s2.policy_name is None


# ── Type contract ───────────────────────────────────────────────────────────


def test_policy_name_accepts_strings():
    """No validation at the dataclass level — but typed annotation says str | None."""
    for name in ["solver", "rewriter", "selector", "proposer-1", "x"]:
        s = Sample(policy_name=name)
        assert s.policy_name == name


def test_falsy_string_treated_as_set():
    """Empty string is technically a string, not None — manager validation should
    catch this if ever set, but the dataclass itself preserves it."""
    s = Sample(policy_name="")
    assert s.policy_name == ""
    assert s.policy_name is not None


# ── Integration with the multi-policy buffer split contract ─────────────────


def test_split_by_policy_logic_simulation():
    """Simulate what RolloutManager._split_by_policy will do (Step 2 in plan.md).
    Confirms the contract: None → __shared__; "<name>" → that bucket."""

    def split_by_policy(samples: list[Sample]) -> dict[str, list[Sample]]:
        """Mirrors the reference implementation in plan.md Step 2."""
        if not any(s.policy_name for s in samples):
            return {"__shared__": samples}
        out: dict[str, list[Sample]] = {}
        for s in samples:
            out.setdefault(s.policy_name or "__shared__", []).append(s)
        return out

    # Case 1: all None → single shared bucket (legacy path)
    samples = [Sample(prompt=f"p{i}") for i in range(4)]
    buckets = split_by_policy(samples)
    assert list(buckets.keys()) == ["__shared__"]
    assert len(buckets["__shared__"]) == 4

    # Case 2: all tagged → multiple buckets
    samples = [Sample(policy_name="solver")] * 3 + [Sample(policy_name="rewriter")] * 2
    buckets = split_by_policy(samples)
    assert sorted(buckets.keys()) == ["rewriter", "solver"]
    assert len(buckets["solver"]) == 3
    assert len(buckets["rewriter"]) == 2

    # Case 3: mixed (some tagged, some None) — None goes to __shared__
    samples = [Sample(policy_name="solver"), Sample(policy_name=None), Sample(policy_name="solver")]
    buckets = split_by_policy(samples)
    assert sorted(buckets.keys()) == ["__shared__", "solver"]
    assert len(buckets["solver"]) == 2
    assert len(buckets["__shared__"]) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
