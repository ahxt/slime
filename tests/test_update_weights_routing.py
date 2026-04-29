"""Tests for Step 5: per-policy update_weights routing in MegatronTrainRayActor.

Asserts the one-line swap in actor.py:548 — get_engines_and_lock.remote is
called with policy_name=getattr(self.args, "policy_name", None). Legacy
train.py builds args without the attribute → None → manager falls back to
_get_updatable_server (the existing single-policy code path).

Run with:
    python -m pytest tests/test_update_weights_routing.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# slime/backends/megatron_utils/actor.py imports torch, megatron, ray, transformers.
# Skip the file when any of those are missing.
for _mod in ("ray", "torch", "megatron", "transformers"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(f"{_mod} not installed; skipping update_weights routing tests",
                    allow_module_level=True)


# ────────────────────────────────────────────────────────────────────────────
# Helpers — build a MegatronTrainRayActor-like instance without Ray
# ────────────────────────────────────────────────────────────────────────────


def _make_actor(args):
    """Construct a MegatronTrainRayActor skipping the heavy __init__."""
    from slime.backends.megatron_utils.actor import MegatronTrainRayActor
    cls = (MegatronTrainRayActor.__ray_actor_class__
           if hasattr(MegatronTrainRayActor, "__ray_actor_class__")
           else MegatronTrainRayActor)
    actor = cls.__new__(cls)
    actor.args = args
    actor.rollout_manager = MagicMock(name="rollout_manager")
    # update_weights returns the 5-tuple (engines, lock, num_new, gpu_counts, gpu_offsets).
    # num_new=0 short-circuits the connect_rollout_engines / weight_updater path.
    actor.rollout_manager.get_engines_and_lock.remote = MagicMock(
        return_value=([], MagicMock(name="lock"), 0, [], [])
    )
    actor.weight_updater = MagicMock(name="weight_updater")
    return actor


def _legacy_args(**overrides):
    """args namespace as built by single-policy train.py — no policy_name attribute."""
    defaults = dict(
        debug_train_only=False,
        debug_rollout_only=False,
        use_fault_tolerance=False,
        offload_train=False,
        check_weight_update_equal=False,
        keep_old_actor=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _multi_policy_args(policy_name, **overrides):
    """args namespace as built by train_multi_policy.py via config_to_namespace —
    has policy_name set to the cfg.name."""
    ns = _legacy_args(**overrides)
    ns.policy_name = policy_name
    return ns


# ────────────────────────────────────────────────────────────────────────────
# Routing — the actual one-line change
# ────────────────────────────────────────────────────────────────────────────


class TestUpdateWeightsRouting:
    def test_legacy_args_routes_with_none(self):
        """Single-policy train.py: args has no policy_name attribute →
        getattr default returns None → get_engines_and_lock(policy_name=None)
        → manager falls back to _get_updatable_server (legacy path)."""
        actor = _make_actor(_legacy_args())

        with patch("slime.backends.megatron_utils.actor.ray.get",
                   side_effect=lambda x: x.return_value if isinstance(x, MagicMock) else x):
            actor.update_weights()

        actor.rollout_manager.get_engines_and_lock.remote.assert_called_once_with(
            policy_name=None
        )
        # Legacy method must NOT be called (we replaced it)
        actor.rollout_manager.get_updatable_engines_and_lock.remote.assert_not_called()

    def test_multi_policy_args_routes_with_name(self):
        """train_multi_policy.py: args.policy_name == cfg.name →
        get_engines_and_lock(policy_name='solver') → manager routes via
        _policy_to_server['solver']."""
        actor = _make_actor(_multi_policy_args("solver"))

        with patch("slime.backends.megatron_utils.actor.ray.get",
                   side_effect=lambda x: x.return_value if isinstance(x, MagicMock) else x):
            actor.update_weights()

        actor.rollout_manager.get_engines_and_lock.remote.assert_called_once_with(
            policy_name="solver"
        )

    def test_multi_policy_three_distinct_routes(self):
        """SPIRAL/multi-agent: each policy's update_weights routes independently."""
        for name in ("solver", "rewriter", "selector"):
            actor = _make_actor(_multi_policy_args(name))
            with patch("slime.backends.megatron_utils.actor.ray.get",
                       side_effect=lambda x: x.return_value if isinstance(x, MagicMock) else x):
                actor.update_weights()
            actor.rollout_manager.get_engines_and_lock.remote.assert_called_once_with(
                policy_name=name
            )

    def test_explicit_none_attribute_routes_with_none(self):
        """Edge case: args explicitly has policy_name=None (e.g. driver sets it
        for a non-trainable model). Routes the same as a missing attribute."""
        ns = _legacy_args()
        ns.policy_name = None
        actor = _make_actor(ns)

        with patch("slime.backends.megatron_utils.actor.ray.get",
                   side_effect=lambda x: x.return_value if isinstance(x, MagicMock) else x):
            actor.update_weights()

        actor.rollout_manager.get_engines_and_lock.remote.assert_called_once_with(
            policy_name=None
        )


# ────────────────────────────────────────────────────────────────────────────
# Short-circuit branches stay intact
# ────────────────────────────────────────────────────────────────────────────


class TestUpdateWeightsShortCircuits:
    def test_debug_train_only_skips(self):
        """No rollout_manager call when debug_train_only is set."""
        actor = _make_actor(_legacy_args(debug_train_only=True))
        actor.update_weights()
        actor.rollout_manager.get_engines_and_lock.remote.assert_not_called()

    def test_debug_rollout_only_skips(self):
        actor = _make_actor(_legacy_args(debug_rollout_only=True))
        actor.update_weights()
        actor.rollout_manager.get_engines_and_lock.remote.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
