"""Tests for create_training_models_multi (Step 6 refactor).

Verifies the function that consolidates per-policy RayTrainGroup allocation,
rollout-manager registration, async_init, and start_rollout_id reconciliation.

Mocks: allocate_train_group (returns MagicMock train_group), ray.get (identity
function — async_init returns plain lists, not ObjectRefs, in tests), and the
rollout manager actor handle.

Run with:
    python -m pytest tests/test_create_training_models_multi.py -v
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if importlib.util.find_spec("ray") is None:
    pytest.skip("ray not installed; skipping placement_group tests", allow_module_level=True)

from slime.utils.policy_config import PolicyConfig, PolicyHandle


@pytest.fixture(autouse=True)
def _mock_ray_get():
    """Make ray.get act as identity inside placement_group so MagicMock returns
    from async_init / register_policy.remote / load.remote pass through cleanly."""
    with patch("slime.ray.placement_group.ray.get", side_effect=lambda x: x):
        yield


# ────────────────────────────────────────────────────────────────────────────
# Helpers — minimal fixtures
# ────────────────────────────────────────────────────────────────────────────


def _policy(name, sglang_server=None, megatron_num_nodes=1, num_gpus_per_node=8):
    return PolicyConfig(
        name=name,
        role="actor",
        hf_checkpoint=f"/ckpt/{name}",
        sglang_server=sglang_server or name,
        buffer_mode="split",
        num_gpus_per_node=num_gpus_per_node,
        megatron_num_nodes=megatron_num_nodes,
        sglang_num_nodes=1,
        sglang={
            "update_weights": True,
            "num_gpus_per_engine": num_gpus_per_node,
            "server_groups": [
                {"worker_type": "regular", "num_gpus": num_gpus_per_node},
            ],
        },
    )


def _base_args(**overrides):
    """Bare-minimum global args namespace. config_to_namespace copies these
    onto the per-policy namespace before overlaying PolicyConfig fields."""
    defaults = dict(
        kl_coef=0.0,
        use_kl_loss=False,
        use_opd=False,
        opd_type=None,
        rollout_global_dataset=False,
        start_rollout_id=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_train_group(start_rollout_ids):
    """Return a MagicMock RayTrainGroup whose async_init and set_rollout_manager
    behave as the real methods do for the purposes of this driver."""
    tg = MagicMock(name="train_group")
    tg.async_init = MagicMock(return_value=start_rollout_ids)
    tg.set_rollout_manager = MagicMock()
    return tg


def _make_rollout_manager():
    rm = MagicMock(name="rollout_manager")
    # register_policy.remote(...) returns an ObjectRef-like; ray.get on a
    # MagicMock is fine because ray.get returns the input when given a non-ref.
    rm.register_policy.remote = MagicMock(return_value=None)
    rm.load.remote = MagicMock(return_value=None)
    return rm


def _patch_allocate(start_ids_per_policy):
    """Patch allocate_train_group so each call returns a MagicMock train_group
    with the requested async_init return-value for that policy.

    `start_ids_per_policy` is a list, ordered by call order.
    """
    iter_ids = iter(start_ids_per_policy)

    def _fake_allocate(args, num_nodes, num_gpus_per_node, pg, role="actor"):
        return _make_train_group(next(iter_ids))

    return patch(
        "slime.ray.placement_group.allocate_train_group",
        side_effect=_fake_allocate,
    )


# ────────────────────────────────────────────────────────────────────────────
# Basic construction & invariants
# ────────────────────────────────────────────────────────────────────────────


class TestBasicConstruction:
    def test_returns_dict_keyed_by_policy_name(self):
        """N policies in → dict of N PolicyHandles keyed by policy name."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver"), _policy("rewriter"), _policy("selector")]
        pgs = {c.name: ("pg", [], []) for c in cfgs}
        rm = _make_rollout_manager()

        with _patch_allocate([[0], [0], [0]]):
            handles = create_training_models_multi(_base_args(), pgs, rm, cfgs)

        assert set(handles.keys()) == {"solver", "rewriter", "selector"}
        assert all(isinstance(h, PolicyHandle) for h in handles.values())

    def test_iteration_order_matches_config_order(self):
        """dict[name, PolicyHandle] iterates in the same order policies were declared."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("c"), _policy("a"), _policy("b")]
        pgs = {c.name: ("pg", [], []) for c in cfgs}
        rm = _make_rollout_manager()

        with _patch_allocate([[0], [0], [0]]):
            handles = create_training_models_multi(_base_args(), pgs, rm, cfgs)

        assert list(handles.keys()) == ["c", "a", "b"]

    def test_handle_args_has_policy_name(self):
        """Each handle's args namespace has policy_name set to cfg.name (used by
        update_weights routing and Sample.policy_name tagging)."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver"), _policy("rewriter")]
        pgs = {c.name: ("pg", [], []) for c in cfgs}
        rm = _make_rollout_manager()

        with _patch_allocate([[0], [0]]):
            handles = create_training_models_multi(_base_args(), pgs, rm, cfgs)

        assert handles["solver"].args.policy_name == "solver"
        assert handles["rewriter"].args.policy_name == "rewriter"

    def test_handle_args_inherits_global_fields(self):
        """Per-policy namespace pulls global args (kl_coef, etc.) before overlaying
        PolicyConfig fields."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver")]
        pgs = {"solver": ("pg", [], [])}
        rm = _make_rollout_manager()

        with _patch_allocate([[0]]):
            handles = create_training_models_multi(
                _base_args(kl_coef=0.5, use_kl_loss=True), pgs, rm, cfgs
            )

        assert handles["solver"].args.kl_coef == 0.5
        assert handles["solver"].args.use_kl_loss is True


# ────────────────────────────────────────────────────────────────────────────
# Rollout-manager registration
# ────────────────────────────────────────────────────────────────────────────


class TestRegistration:
    def test_each_policy_registered_with_manager(self):
        """register_policy.remote called once per policy with (name, sglang_server, args)."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver"), _policy("rewriter")]
        pgs = {c.name: ("pg", [], []) for c in cfgs}
        rm = _make_rollout_manager()

        with _patch_allocate([[0], [0]]):
            create_training_models_multi(_base_args(), pgs, rm, cfgs)

        assert rm.register_policy.remote.call_count == 2
        calls = rm.register_policy.remote.call_args_list
        # Each call: (name, sglang_server, args)
        names = [c.args[0] for c in calls]
        servers = [c.args[1] for c in calls]
        assert names == ["solver", "rewriter"]
        assert servers == ["solver", "rewriter"]
        # args is the per-policy Namespace, not the global one
        for call_args, name in zip(calls, ["solver", "rewriter"]):
            assert call_args.args[2].policy_name == name

    def test_register_uses_per_policy_namespace_not_global(self):
        """The Namespace handed to register_policy must be the per-policy one
        (not the bare global args), so RolloutManager._policy_args[name] holds the
        policy-specific values."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver")]
        pgs = {"solver": ("pg", [], [])}
        rm = _make_rollout_manager()
        global_args = _base_args()

        with _patch_allocate([[0]]):
            create_training_models_multi(global_args, pgs, rm, cfgs)

        registered_args = rm.register_policy.remote.call_args.args[2]
        assert registered_args is not global_args
        assert registered_args.policy_name == "solver"


# ────────────────────────────────────────────────────────────────────────────
# async_init invocation — kwargs parity with legacy create_training_models
# ────────────────────────────────────────────────────────────────────────────


class TestAsyncInit:
    def test_async_init_called_per_policy(self):
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver"), _policy("rewriter")]
        pgs = {c.name: ("pg", [], []) for c in cfgs}
        rm = _make_rollout_manager()

        with _patch_allocate([[0], [0]]):
            handles = create_training_models_multi(_base_args(), pgs, rm, cfgs)

        for h in handles.values():
            h.train_group.async_init.assert_called_once()

    def test_with_ref_derived_from_kl_coef(self):
        """legacy parity: with_ref = kl_coef != 0 or use_kl_loss"""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver")]
        pgs = {"solver": ("pg", [], [])}
        rm = _make_rollout_manager()

        with _patch_allocate([[0]]):
            handles = create_training_models_multi(_base_args(kl_coef=0.1), pgs, rm, cfgs)

        kwargs = handles["solver"].train_group.async_init.call_args.kwargs
        assert kwargs["with_ref"] is True

    def test_with_ref_false_when_no_kl(self):
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver")]
        pgs = {"solver": ("pg", [], [])}
        rm = _make_rollout_manager()

        with _patch_allocate([[0]]):
            handles = create_training_models_multi(
                _base_args(kl_coef=0.0, use_kl_loss=False), pgs, rm, cfgs
            )

        kwargs = handles["solver"].train_group.async_init.call_args.kwargs
        assert kwargs["with_ref"] is False

    def test_with_opd_teacher_only_when_megatron_opd(self):
        """legacy parity: with_opd_teacher = use_opd and opd_type == 'megatron'"""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver")]
        pgs = {"solver": ("pg", [], [])}
        rm = _make_rollout_manager()

        with _patch_allocate([[0]]):
            handles = create_training_models_multi(
                _base_args(use_opd=True, opd_type="megatron"), pgs, rm, cfgs
            )

        kwargs = handles["solver"].train_group.async_init.call_args.kwargs
        assert kwargs["with_opd_teacher"] is True

    def test_with_opd_teacher_false_when_sglang_opd(self):
        """OPD-sglang uses a separate frozen engine, not a co-resident megatron tag."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver")]
        pgs = {"solver": ("pg", [], [])}
        rm = _make_rollout_manager()

        with _patch_allocate([[0]]):
            handles = create_training_models_multi(
                _base_args(use_opd=True, opd_type="sglang"), pgs, rm, cfgs
            )

        kwargs = handles["solver"].train_group.async_init.call_args.kwargs
        assert kwargs["with_opd_teacher"] is False

    def test_with_opd_teacher_handles_missing_attrs(self):
        """getattr defaults: if base_args lacks use_opd / opd_type entirely,
        the call should not raise — it just becomes False."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver")]
        pgs = {"solver": ("pg", [], [])}
        rm = _make_rollout_manager()

        bare = Namespace(
            kl_coef=0.0,
            use_kl_loss=False,
            rollout_global_dataset=False,
            start_rollout_id=None,
        )

        with _patch_allocate([[0]]):
            handles = create_training_models_multi(bare, pgs, rm, cfgs)

        kwargs = handles["solver"].train_group.async_init.call_args.kwargs
        assert kwargs["with_opd_teacher"] is False

    def test_role_passed_through_to_async_init(self):
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver")]
        pgs = {"solver": ("pg", [], [])}
        rm = _make_rollout_manager()

        with _patch_allocate([[0]]):
            handles = create_training_models_multi(_base_args(), pgs, rm, cfgs)

        kwargs = handles["solver"].train_group.async_init.call_args.kwargs
        assert kwargs["role"] == "actor"

    def test_set_rollout_manager_called(self):
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver"), _policy("rewriter")]
        pgs = {c.name: ("pg", [], []) for c in cfgs}
        rm = _make_rollout_manager()

        with _patch_allocate([[0], [0]]):
            handles = create_training_models_multi(_base_args(), pgs, rm, cfgs)

        for h in handles.values():
            h.train_group.set_rollout_manager.assert_called_once_with(rm)


# ────────────────────────────────────────────────────────────────────────────
# start_rollout_id reconciliation
# ────────────────────────────────────────────────────────────────────────────


class TestStartRolloutIdReconciliation:
    def test_all_zero_no_warning(self, caplog):
        """Fresh start: every policy returns [0, 0, ...] within group; chosen=0; no warning."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver"), _policy("rewriter")]
        pgs = {c.name: ("pg", [], []) for c in cfgs}
        rm = _make_rollout_manager()
        args = _base_args()

        with caplog.at_level(logging.WARNING, logger="slime.ray.placement_group"):
            with _patch_allocate([[0, 0], [0, 0]]):
                create_training_models_multi(args, pgs, rm, cfgs)

        assert args.start_rollout_id == 0
        assert "diverged" not in caplog.text.lower()

    def test_all_agree_uses_that_value(self):
        """Every policy at rollout 7 → args.start_rollout_id == 7, no warning."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver"), _policy("rewriter")]
        pgs = {c.name: ("pg", [], []) for c in cfgs}
        rm = _make_rollout_manager()
        args = _base_args()

        with _patch_allocate([[7, 7], [7, 7]]):
            create_training_models_multi(args, pgs, rm, cfgs)

        assert args.start_rollout_id == 7

    def test_divergence_takes_min_and_warns(self, caplog):
        """solver=5, rewriter=7 → use 5, log a warning naming the divergence."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver"), _policy("rewriter")]
        pgs = {c.name: ("pg", [], []) for c in cfgs}
        rm = _make_rollout_manager()
        args = _base_args()

        with caplog.at_level(logging.WARNING, logger="slime.ray.placement_group"):
            with _patch_allocate([[5, 5], [7, 7]]):
                create_training_models_multi(args, pgs, rm, cfgs)

        assert args.start_rollout_id == 5
        assert "diverged" in caplog.text.lower()
        # Both policy names mentioned for debuggability
        assert "solver" in caplog.text and "rewriter" in caplog.text

    def test_within_group_disagreement_raises(self):
        """If within-group workers disagree, fail fast with the policy name in the error."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver")]
        pgs = {"solver": ("pg", [], [])}
        rm = _make_rollout_manager()

        with pytest.raises(RuntimeError, match=r"solver.*disagree"):
            with _patch_allocate([[3, 5]]):  # 2 workers, different ids
                create_training_models_multi(_base_args(), pgs, rm, cfgs)

    def test_user_supplied_start_rollout_id_preserved(self):
        """If user explicitly set --start-rollout-id, don't overwrite it."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver")]
        pgs = {"solver": ("pg", [], [])}
        rm = _make_rollout_manager()
        args = _base_args(start_rollout_id=42)

        with _patch_allocate([[10, 10]]):
            create_training_models_multi(args, pgs, rm, cfgs)

        assert args.start_rollout_id == 42  # not overwritten


# ────────────────────────────────────────────────────────────────────────────
# rollout_global_dataset parity with legacy create_training_models
# ────────────────────────────────────────────────────────────────────────────


class TestRolloutGlobalDataset:
    def test_load_called_when_flag_set(self):
        """Legacy parity (placement_group.py:178): if rollout_global_dataset is True,
        load the buffer from `start_rollout_id - 1`."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver")]
        pgs = {"solver": ("pg", [], [])}
        rm = _make_rollout_manager()
        args = _base_args(rollout_global_dataset=True)

        with _patch_allocate([[7]]):
            create_training_models_multi(args, pgs, rm, cfgs)

        rm.load.remote.assert_called_once_with(6)  # start_rollout_id - 1

    def test_load_not_called_when_flag_unset(self):
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver")]
        pgs = {"solver": ("pg", [], [])}
        rm = _make_rollout_manager()
        args = _base_args(rollout_global_dataset=False)

        with _patch_allocate([[7]]):
            create_training_models_multi(args, pgs, rm, cfgs)

        rm.load.remote.assert_not_called()


# ────────────────────────────────────────────────────────────────────────────
# allocate_train_group invocation — pg + sizing flow through correctly
# ────────────────────────────────────────────────────────────────────────────


class TestAllocateTrainGroup:
    def test_each_policy_gets_its_pg_slice(self):
        """allocate_train_group is called with pgs[cfg.name] — not a different policy's PG."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver"), _policy("rewriter")]
        pgs = {
            "solver": ("pg-solver", [0, 1, 2, 3, 4, 5, 6, 7], list(range(8))),
            "rewriter": ("pg-rewriter", [8, 9, 10, 11, 12, 13, 14, 15], list(range(8, 16))),
        }
        rm = _make_rollout_manager()
        recorded_pgs = []

        def _record_allocate(args, num_nodes, num_gpus_per_node, pg, role="actor"):
            recorded_pgs.append((args.policy_name, pg))
            return _make_train_group([0])

        with patch(
            "slime.ray.placement_group.allocate_train_group", side_effect=_record_allocate
        ):
            create_training_models_multi(_base_args(), pgs, rm, cfgs)

        recorded = dict(recorded_pgs)
        assert recorded["solver"] == pgs["solver"]
        assert recorded["rewriter"] == pgs["rewriter"]

    def test_sizing_flows_through(self):
        """megatron_num_nodes and num_gpus_per_node are forwarded as positional/kwarg."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver", megatron_num_nodes=2, num_gpus_per_node=4)]
        pgs = {"solver": ("pg", [], [])}
        rm = _make_rollout_manager()
        seen = {}

        def _record(args, num_nodes, num_gpus_per_node, pg, role="actor"):
            seen["nodes"] = num_nodes
            seen["per_node"] = num_gpus_per_node
            seen["role"] = role
            return _make_train_group([0])

        with patch("slime.ray.placement_group.allocate_train_group", side_effect=_record):
            create_training_models_multi(_base_args(), pgs, rm, cfgs)

        assert seen == {"nodes": 2, "per_node": 4, "role": "actor"}


# ────────────────────────────────────────────────────────────────────────────
# Ordering invariants
# ────────────────────────────────────────────────────────────────────────────


class TestOrdering:
    def test_register_happens_before_async_init(self):
        """Registration must occur before async_init so RolloutManager._policy_args
        is populated when the train group starts initializing (and may inspect it
        via set_rollout_manager / weight-sync setup)."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver")]
        pgs = {"solver": ("pg", [], [])}
        rm = _make_rollout_manager()
        events: list[str] = []

        rm.register_policy.remote = MagicMock(
            side_effect=lambda *a, **kw: events.append("register") or None
        )

        def _allocate(args, num_nodes, num_gpus_per_node, pg, role="actor"):
            tg = MagicMock(name="train_group")
            tg.async_init = MagicMock(
                side_effect=lambda *a, **kw: events.append("async_init") or [0]
            )
            tg.set_rollout_manager = MagicMock(
                side_effect=lambda *a, **kw: events.append("set_rm")
            )
            return tg

        with patch("slime.ray.placement_group.allocate_train_group", side_effect=_allocate):
            create_training_models_multi(_base_args(), pgs, rm, cfgs)

        # All registrations finish before any async_init starts
        first_init = events.index("async_init")
        assert all(events[i] == "register" for i in range(first_init))

    def test_load_called_after_all_async_init(self):
        """rollout_global_dataset load must come after start_rollout_id is finalized."""
        from slime.ray.placement_group import create_training_models_multi

        cfgs = [_policy("solver"), _policy("rewriter")]
        pgs = {c.name: ("pg", [], []) for c in cfgs}
        events: list[str] = []
        rm = _make_rollout_manager()
        rm.register_policy.remote = MagicMock(return_value=None)
        rm.load.remote = MagicMock(side_effect=lambda *a, **kw: events.append("load"))

        def _allocate(args, num_nodes, num_gpus_per_node, pg, role="actor"):
            tg = MagicMock(name="train_group")
            tg.async_init = MagicMock(
                side_effect=lambda *a, **kw: events.append("async_init") or [0]
            )
            tg.set_rollout_manager = MagicMock()
            return tg

        with patch("slime.ray.placement_group.allocate_train_group", side_effect=_allocate):
            create_training_models_multi(
                _base_args(rollout_global_dataset=True), pgs, rm, cfgs
            )

        assert events == ["async_init", "async_init", "load"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
