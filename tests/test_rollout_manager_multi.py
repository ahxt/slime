"""Tests for the multi-policy methods on RolloutManager (Step 4).

Covers:
  - _get_server(name)
  - register_policy(...)
  - get_engines_and_lock(policy_name=None) — both branches
  - set_train_parallel_config(config, policy_name=None) — both branches

Mocks Ray so the methods can be exercised as plain Python. RolloutManager itself
is decorated with @ray.remote, so we instantiate via __new__ + manual __init__
of just the multi-policy state (skipping the heavy Ray actor setup, sglang
servers, etc.) to test only the new logic in isolation.

Run with:
    python -m pytest tests/test_rollout_manager_multi.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
from argparse import Namespace
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# slime/ray/rollout.py imports ray, sglang_router, etc. at module level.
# Skip the whole file when those are unavailable.
if importlib.util.find_spec("ray") is None:
    pytest.skip("ray not installed; skipping rollout manager tests", allow_module_level=True)


# ────────────────────────────────────────────────────────────────────────────
# Helpers — build a RolloutManager-like instance without Ray
# ────────────────────────────────────────────────────────────────────────────


class _FakeServer:
    """Stand-in for slime.ray.rollout.RolloutServer with the fields we touch."""

    def __init__(
        self,
        name: str,
        update_weights: bool = True,
        engines=None,
        num_new_engines: int = 0,
        engine_gpu_counts=None,
        engine_gpu_offsets=None,
    ):
        self.model_name = name
        self.update_weights = update_weights
        self.engines = engines or [f"engine-{name}-0"]
        self.num_new_engines = num_new_engines
        self.engine_gpu_counts = engine_gpu_counts or [8]
        self.engine_gpu_offsets = engine_gpu_offsets or [0]


def _make_manager(servers: dict[str, _FakeServer]) -> "RolloutManager":
    """Build a RolloutManager-shaped object with only the state the multi-policy
    methods touch. Skips Ray actor instantiation and __init__'s heavy setup."""
    from slime.ray.rollout import RolloutManager
    # __new__ on the underlying class (the .options(...).remote(...) factory wraps
    # the actual class; we want the raw Python class for in-process testing).
    cls = RolloutManager.__ray_actor_class__ if hasattr(RolloutManager, "__ray_actor_class__") else RolloutManager
    mgr = cls.__new__(cls)
    mgr.servers = servers
    mgr.rollout_engine_lock = MagicMock(name="rollout_engine_lock")
    mgr.train_parallel_config = {}
    mgr._policy_to_server = {}
    mgr._policy_args = {}
    mgr._policy_train_parallel_config = {}
    return mgr


# ────────────────────────────────────────────────────────────────────────────
# _get_server
# ────────────────────────────────────────────────────────────────────────────


class TestGetServer:
    def test_returns_server_by_name(self):
        servers = {"solver": _FakeServer("solver"), "rewriter": _FakeServer("rewriter")}
        mgr = _make_manager(servers)
        assert mgr._get_server("solver") is servers["solver"]
        assert mgr._get_server("rewriter") is servers["rewriter"]

    def test_unknown_name_raises(self):
        mgr = _make_manager({"solver": _FakeServer("solver")})
        with pytest.raises(ValueError, match="unknown sglang server"):
            mgr._get_server("ghost")

    def test_legacy_get_updatable_server_unchanged(self):
        """The legacy first-wins method must keep working — Step 4 only adds new
        methods, doesn't modify existing ones."""
        servers = {
            "ref": _FakeServer("ref", update_weights=False),
            "solver": _FakeServer("solver", update_weights=True),
        }
        mgr = _make_manager(servers)
        result = mgr._get_updatable_server()
        assert result is servers["solver"]


# ────────────────────────────────────────────────────────────────────────────
# register_policy
# ────────────────────────────────────────────────────────────────────────────


class TestRegisterPolicy:
    def test_basic_registration(self):
        mgr = _make_manager({"solver": _FakeServer("solver")})
        ns = Namespace(policy_name="solver", lr=1e-6)
        mgr.register_policy("solver", "solver", ns)
        assert mgr._policy_to_server["solver"] == "solver"
        assert mgr._policy_args["solver"] is ns

    def test_with_train_parallel_config(self):
        mgr = _make_manager({"solver": _FakeServer("solver")})
        cfg = {"dp_size": 2, "tp_size": 4}
        mgr.register_policy("solver", "solver", Namespace(), train_parallel_config=cfg)
        assert mgr._policy_train_parallel_config["solver"] == cfg

    def test_train_parallel_config_optional(self):
        """When omitted, _policy_train_parallel_config stays empty for that policy
        (Step 4's design: train_actor fills it later via set_train_parallel_config)."""
        mgr = _make_manager({"solver": _FakeServer("solver")})
        mgr.register_policy("solver", "solver", Namespace())
        assert "solver" not in mgr._policy_train_parallel_config

    def test_duplicate_server_rejected(self):
        servers = {"X": _FakeServer("X"), "Y": _FakeServer("Y")}
        mgr = _make_manager(servers)
        mgr.register_policy("a", "X", Namespace())
        with pytest.raises(ValueError, match="already bound to another policy"):
            mgr.register_policy("b", "X", Namespace())

    def test_unknown_server_rejected(self):
        mgr = _make_manager({"solver": _FakeServer("solver")})
        with pytest.raises(ValueError, match="unknown sglang server"):
            mgr.register_policy("ghost", "ghost-server", Namespace())

    def test_frozen_server_rejected(self):
        """Frozen mirrors (update_weights=false) cannot be a policy's training target."""
        mgr = _make_manager({"ref": _FakeServer("ref", update_weights=False)})
        with pytest.raises(ValueError, match="update_weights=false"):
            mgr.register_policy("ref-trainer", "ref", Namespace())

    def test_two_policies_two_servers(self):
        servers = {"solver": _FakeServer("solver"), "rewriter": _FakeServer("rewriter")}
        mgr = _make_manager(servers)
        mgr.register_policy("solver", "solver", Namespace())
        mgr.register_policy("rewriter", "rewriter", Namespace())
        assert mgr._policy_to_server == {"solver": "solver", "rewriter": "rewriter"}


# ────────────────────────────────────────────────────────────────────────────
# get_engines_and_lock — both branches
# ────────────────────────────────────────────────────────────────────────────


class TestGetEnginesAndLock:
    def test_legacy_fallback_with_none(self):
        """policy_name=None → routes through _get_updatable_server (the existing
        first-wins path used by single-policy train.py and PPO)."""
        servers = {
            "ref": _FakeServer("ref", update_weights=False),
            "solver": _FakeServer(
                "solver",
                update_weights=True,
                engines=["e1"],
                num_new_engines=1,
                engine_gpu_counts=[8],
                engine_gpu_offsets=[0],
            ),
        }
        mgr = _make_manager(servers)
        engines, lock, num_new, gpu_counts, gpu_offsets = mgr.get_engines_and_lock(
            policy_name=None
        )
        # _get_updatable_server returns the first updatable → "solver"
        assert engines == ["e1"]
        assert num_new == 1
        assert gpu_counts == [8]
        assert gpu_offsets == [0]
        assert lock is mgr.rollout_engine_lock

    def test_named_routes_to_right_server(self):
        servers = {
            "solver": _FakeServer("solver", engines=["solver-e0"], engine_gpu_counts=[4]),
            "rewriter": _FakeServer("rewriter", engines=["rewriter-e0"], engine_gpu_counts=[4]),
        }
        mgr = _make_manager(servers)
        mgr.register_policy("solver", "solver", Namespace())
        mgr.register_policy("rewriter", "rewriter", Namespace())

        engines, _, _, gpu_counts, _ = mgr.get_engines_and_lock(policy_name="solver")
        assert engines == ["solver-e0"]

        engines, _, _, gpu_counts, _ = mgr.get_engines_and_lock(policy_name="rewriter")
        assert engines == ["rewriter-e0"]

    def test_colocated_offsets_are_policy_relative(self):
        servers = {
            "rewriter": _FakeServer(
                "rewriter",
                engines=["rewriter-e0"],
                engine_gpu_counts=[2],
                engine_gpu_offsets=[2],
            ),
        }
        mgr = _make_manager(servers)
        mgr.register_policy(
            "rewriter",
            "rewriter",
            Namespace(
                colocate=True,
                actor_gpu_offset=2,
                actor_num_nodes=1,
                actor_num_gpus_per_node=2,
            ),
        )

        _, _, _, _, gpu_offsets = mgr.get_engines_and_lock(policy_name="rewriter")
        assert gpu_offsets == [0]

    def test_unknown_policy_raises(self):
        mgr = _make_manager({"solver": _FakeServer("solver")})
        with pytest.raises(ValueError, match="no registered sglang server"):
            mgr.get_engines_and_lock(policy_name="ghost")

    def test_returns_5_tuple_shape(self):
        """Same shape as legacy get_updatable_engines_and_lock so callers can swap."""
        mgr = _make_manager({"solver": _FakeServer("solver")})
        mgr.register_policy("solver", "solver", Namespace())
        result = mgr.get_engines_and_lock(policy_name="solver")
        assert len(result) == 5

    def test_legacy_method_unchanged(self):
        """get_updatable_engines_and_lock must keep its 0-arg signature for legacy callers."""
        servers = {"solver": _FakeServer("solver", num_new_engines=2)}
        mgr = _make_manager(servers)
        engines, _, num_new, _, _ = mgr.get_updatable_engines_and_lock()
        assert engines == ["engine-solver-0"]
        assert num_new == 2

    def test_clear_num_new_engines_routes_by_policy(self):
        servers = {
            "solver": _FakeServer("solver", num_new_engines=1),
            "rewriter": _FakeServer("rewriter", num_new_engines=2),
        }
        mgr = _make_manager(servers)
        mgr.register_policy("solver", "solver", Namespace())
        mgr.register_policy("rewriter", "rewriter", Namespace())

        mgr.clear_updatable_num_new_engines(policy_name="rewriter")
        assert servers["solver"].num_new_engines == 1
        assert servers["rewriter"].num_new_engines == 0


# ────────────────────────────────────────────────────────────────────────────
# set_train_parallel_config — extended signature, legacy behavior preserved
# ────────────────────────────────────────────────────────────────────────────


class TestSetTrainParallelConfig:
    def test_legacy_single_arg(self):
        """Passing one positional arg (legacy) writes only the global field."""
        mgr = _make_manager({})
        cfg = {"dp_size": 4, "tp_size": 2}
        mgr.set_train_parallel_config(cfg)
        assert mgr.train_parallel_config == cfg
        # Per-policy dict stays empty
        assert mgr._policy_train_parallel_config == {}

    def test_with_policy_name_writes_both(self):
        """Multi-policy: both the global field and the per-policy dict get written."""
        mgr = _make_manager({})
        cfg = {"dp_size": 2, "tp_size": 4}
        mgr.set_train_parallel_config(cfg, policy_name="solver")
        assert mgr.train_parallel_config == cfg
        assert mgr._policy_train_parallel_config["solver"] == cfg

    def test_three_policies_three_configs(self):
        mgr = _make_manager({})
        for name, dp in [("solver", 2), ("rewriter", 4), ("selector", 1)]:
            mgr.set_train_parallel_config({"dp_size": dp}, policy_name=name)
        assert mgr._policy_train_parallel_config["solver"]["dp_size"] == 2
        assert mgr._policy_train_parallel_config["rewriter"]["dp_size"] == 4
        assert mgr._policy_train_parallel_config["selector"]["dp_size"] == 1


# ────────────────────────────────────────────────────────────────────────────
# State init — multi-policy dicts default empty
# ────────────────────────────────────────────────────────────────────────────


class TestStateDictInit:
    def test_empty_after_construction(self):
        mgr = _make_manager({"solver": _FakeServer("solver")})
        assert mgr._policy_to_server == {}
        assert mgr._policy_args == {}
        assert mgr._policy_train_parallel_config == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
