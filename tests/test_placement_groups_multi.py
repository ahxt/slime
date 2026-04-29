"""Tests for create_placement_groups_multi (Step 7).

Mocks _create_placement_group so the slice-carving math runs without Ray.
The pure-Python equivalent (derive_policy_slices in policy_config.py) is also
covered by tests/test_multi_policy_config.py — these tests focus on the actual
slime.ray.placement_group entry point that train_multi_policy.py calls.

Run with:
    python -m pytest tests/test_placement_groups_multi.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
from argparse import Namespace
from unittest.mock import patch

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# slime.ray.placement_group imports ray + slime.ray.actor_group at module level.
# Skip the whole file when ray isn't installed.
if importlib.util.find_spec("ray") is None:
    pytest.skip("ray not installed; skipping placement_group tests", allow_module_level=True)

from slime.ray.placement_group import (
    create_placement_groups_multi,
    create_rollout_manager_multi,
)
from slime.utils.policy_config import PolicyConfig


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _policy(name, megatron_num_nodes=1, sglang_num_nodes=1, num_gpus_per_node=8):
    return PolicyConfig(
        name=name,
        role="actor",
        hf_checkpoint="/x",
        sglang_server=name,
        buffer_mode="split",
        num_gpus_per_node=num_gpus_per_node,
        megatron_num_nodes=megatron_num_nodes,
        sglang_num_nodes=sglang_num_nodes,
        sglang={
            "update_weights": True,
            "num_gpus_per_engine": num_gpus_per_node,
            "server_groups": [
                {"worker_type": "regular", "num_gpus": sglang_num_nodes * num_gpus_per_node}
            ],
        },
    )


class _MockPG:
    """Stand-in for ray's PlacementGroup — only its identity matters in tests."""
    def __init__(self, total):
        self.total = total

    def __repr__(self):
        return f"MockPG(total={self.total})"


def _mock_create_placement_group(total):
    """Replacement for _create_placement_group that returns predictable indices."""
    return _MockPG(total), list(range(total)), list(range(total))


# ────────────────────────────────────────────────────────────────────────────
# create_placement_groups_multi — slice carving
# ────────────────────────────────────────────────────────────────────────────


class TestCreatePlacementGroupsMulti:
    def test_three_policies_colocate(self):
        cfgs = [_policy("solver"), _policy("rewriter"), _policy("selector")]
        args = Namespace(colocate=True)

        with patch(
            "slime.ray.placement_group._create_placement_group",
            side_effect=_mock_create_placement_group,
        ) as m:
            result = create_placement_groups_multi(args, cfgs)

        # Total requested = max(24, 24) = 24
        m.assert_called_once_with(24)

        assert set(result.keys()) == {"solver", "rewriter", "selector", "rollout"}
        assert result["solver"][1] == list(range(0, 8))
        assert result["rewriter"][1] == list(range(8, 16))
        assert result["selector"][1] == list(range(16, 24))
        # rollout shares the whole pool when colocated
        assert result["rollout"][1] == list(range(24))

    def test_three_policies_no_colocate(self):
        cfgs = [_policy("solver"), _policy("rewriter"), _policy("selector")]
        args = Namespace(colocate=False)

        with patch(
            "slime.ray.placement_group._create_placement_group",
            side_effect=_mock_create_placement_group,
        ) as m:
            result = create_placement_groups_multi(args, cfgs)

        # Total = 24 + 24 = 48
        m.assert_called_once_with(48)

        assert result["solver"][1] == list(range(0, 8))
        assert result["rewriter"][1] == list(range(8, 16))
        assert result["selector"][1] == list(range(16, 24))
        # rollout slice is contiguous AFTER the actor slices
        assert result["rollout"][1] == list(range(24, 48))

    def test_two_policies_disjoint(self):
        cfgs = [
            _policy("a", megatron_num_nodes=1, sglang_num_nodes=1, num_gpus_per_node=4),
            _policy("b", megatron_num_nodes=1, sglang_num_nodes=1, num_gpus_per_node=4),
        ]
        args = Namespace(colocate=False)

        with patch(
            "slime.ray.placement_group._create_placement_group",
            side_effect=_mock_create_placement_group,
        ):
            result = create_placement_groups_multi(args, cfgs)

        assert result["a"][1] == list(range(0, 4))
        assert result["b"][1] == list(range(4, 8))
        assert result["rollout"][1] == list(range(8, 16))

    def test_heterogeneous_actor_sizes(self):
        # Policy a: 1 × 4 = 4 actor GPUs; policy b: 2 × 4 = 8 actor GPUs
        cfgs = [
            _policy("a", megatron_num_nodes=1, sglang_num_nodes=1, num_gpus_per_node=4),
            _policy("b", megatron_num_nodes=2, sglang_num_nodes=1, num_gpus_per_node=4),
        ]
        args = Namespace(colocate=False)

        with patch(
            "slime.ray.placement_group._create_placement_group",
            side_effect=_mock_create_placement_group,
        ) as m:
            result = create_placement_groups_multi(args, cfgs)

        # actor=12, rollout=8, total=20
        m.assert_called_once_with(20)
        assert result["a"][1] == list(range(0, 4))
        assert result["b"][1] == list(range(4, 12))
        assert result["rollout"][1] == list(range(12, 20))

    def test_single_policy(self):
        cfgs = [_policy("solo")]
        args = Namespace(colocate=True)

        with patch(
            "slime.ray.placement_group._create_placement_group",
            side_effect=_mock_create_placement_group,
        ) as m:
            result = create_placement_groups_multi(args, cfgs)

        m.assert_called_once_with(8)
        assert result["solo"][1] == list(range(0, 8))
        assert result["rollout"][1] == list(range(0, 8))

    def test_pg_object_shared_across_slices(self):
        """All slices must reference the same placement-group object — Ray actors
        bind to bundles within one PG, not multiple PGs."""
        cfgs = [_policy("a"), _policy("b")]
        args = Namespace(colocate=True)

        with patch(
            "slime.ray.placement_group._create_placement_group",
            side_effect=_mock_create_placement_group,
        ):
            result = create_placement_groups_multi(args, cfgs)

        pg = result["a"][0]
        assert result["b"][0] is pg
        assert result["rollout"][0] is pg

    def test_returns_lists_not_iterators(self):
        """The result tuples must contain plain lists so they're indexable and reusable."""
        cfgs = [_policy("a"), _policy("b")]
        args = Namespace(colocate=True)

        with patch(
            "slime.ray.placement_group._create_placement_group",
            side_effect=_mock_create_placement_group,
        ):
            result = create_placement_groups_multi(args, cfgs)

        for key in ["a", "b", "rollout"]:
            _, idxs, gpus = result[key]
            assert isinstance(idxs, list)
            assert isinstance(gpus, list)


# ────────────────────────────────────────────────────────────────────────────
# create_rollout_manager_multi — stub raises (Step 4 ships the real version)
# ────────────────────────────────────────────────────────────────────────────


class TestCreateRolloutManagerMultiStub:
    def test_stub_raises_pointing_at_step_4(self):
        with pytest.raises(NotImplementedError, match="Step 4"):
            create_rollout_manager_multi(
                args=Namespace(),
                pg=_MockPG(8),
                sglang_config=None,
            )

    def test_stub_message_mentions_train_py(self):
        """The error message should point users at train.py for single-policy in
        the meantime, so they don't get stuck."""
        with pytest.raises(NotImplementedError, match="train.py"):
            create_rollout_manager_multi(args=Namespace(), pg=None, sglang_config=None)


# ────────────────────────────────────────────────────────────────────────────
# Back-compat: original create_placement_groups untouched
# ────────────────────────────────────────────────────────────────────────────


class TestLegacyCreatePlacementGroupsUnchanged:
    """Spot-check that the original create_placement_groups function still has
    the same signature and dispatches the way train.py expects."""

    def test_original_function_still_exists(self):
        from slime.ray.placement_group import create_placement_groups
        assert callable(create_placement_groups)

    def test_original_function_signature(self):
        import inspect
        from slime.ray.placement_group import create_placement_groups
        sig = inspect.signature(create_placement_groups)
        assert list(sig.parameters.keys()) == ["args"]

    def test_original_dispatches_on_use_critic(self):
        """train.py expects the result to have keys: actor, rollout, critic."""
        from slime.ray.placement_group import create_placement_groups
        args = Namespace(
            debug_train_only=False,
            debug_rollout_only=False,
            colocate=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=4,
            rollout_num_gpus=4,
            use_critic=False,
        )
        with patch(
            "slime.ray.placement_group._create_placement_group",
            side_effect=_mock_create_placement_group,
        ):
            result = create_placement_groups(args)
        assert "actor" in result
        assert "rollout" in result
        assert "critic" in result
        assert result["critic"] is None  # use_critic=False
        # Bit-for-bit shape check: all three reference the same PG (colocate)
        assert result["actor"][0] is result["rollout"][0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
