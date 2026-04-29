"""Tests for per-policy buffer split (Step 2).

Covers:
  - _split_by_policy(samples) — bucketing by Sample.policy_name
  - _post_process_rewards(samples, policy_args=None) — args fallback
  - _convert_samples_to_train_data(samples, policy_args=None) — args propagation
  - generate(rollout_id) — dispatch between legacy single-buffer and multi-policy

The existing single-policy code path (no register_policy ever called → empty
self._policy_to_server) must stay bit-for-bit identical: this is the legacy
regression bar.

Run with:
    python -m pytest tests/test_buffer_split.py -v
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

# slime/ray/rollout.py imports ray, torch, and sglang at module level.
if importlib.util.find_spec("ray") is None:
    pytest.skip("ray not installed; skipping buffer split tests", allow_module_level=True)


# ────────────────────────────────────────────────────────────────────────────
# Helpers — build a RolloutManager-like instance without Ray
# ────────────────────────────────────────────────────────────────────────────


def _make_manager():
    """Construct a RolloutManager skipping the heavy Ray actor __init__."""
    from slime.ray.rollout import RolloutManager
    cls = RolloutManager.__ray_actor_class__ if hasattr(RolloutManager, "__ray_actor_class__") else RolloutManager
    mgr = cls.__new__(cls)
    mgr.servers = {}
    mgr.rollout_engine_lock = MagicMock(name="rollout_engine_lock")
    mgr.train_parallel_config = {"dp_size": 2}
    mgr._policy_to_server = {}
    mgr._policy_args = {}
    mgr._policy_train_parallel_config = {}
    mgr.custom_reward_post_process_func = None
    mgr.custom_convert_samples_to_train_data_func = None
    # Manager-global args used by the legacy fallback path
    mgr.args = _default_args()
    return mgr


def _default_args(**overrides):
    defaults = dict(
        advantage_estimator="grpo",
        rewards_normalization=False,
        grpo_std_normalization=False,
        n_samples_per_prompt=4,
        rollout_batch_size=2,
        ci_test=False,
        use_fault_tolerance=False,
        debug_rollout_only=False,
        balance_data=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_sample(policy_name=None, reward=1.0):
    """Minimal Sample-like object with the attrs _split_by_policy / _post_process_rewards
    actually read. We avoid the real Sample dataclass here because it has many
    required fields we don't care about for routing tests."""
    from slime.utils.types import Sample
    s = Sample(prompt="p", index=0)
    s.policy_name = policy_name
    s.reward = reward
    return s


# ────────────────────────────────────────────────────────────────────────────
# _split_by_policy — pure bucketing
# ────────────────────────────────────────────────────────────────────────────


class TestSplitByPolicy:
    def test_all_none_returns_shared(self):
        """Legacy regression: when no sample has policy_name set, return the
        single __shared__ bucket containing all samples (no copy / wrap surprise)."""
        mgr = _make_manager()
        samples = [_make_sample() for _ in range(8)]
        out = mgr._split_by_policy(samples)
        assert list(out.keys()) == ["__shared__"]
        assert out["__shared__"] is samples or out["__shared__"] == samples
        assert len(out["__shared__"]) == 8

    def test_all_tagged_routes_by_name(self):
        mgr = _make_manager()
        samples = (
            [_make_sample(policy_name="solver") for _ in range(4)]
            + [_make_sample(policy_name="rewriter") for _ in range(4)]
        )
        out = mgr._split_by_policy(samples)
        assert set(out.keys()) == {"solver", "rewriter"}
        assert len(out["solver"]) == 4
        assert len(out["rewriter"]) == 4
        assert all(s.policy_name == "solver" for s in out["solver"])

    def test_three_policies_distinct_buckets(self):
        mgr = _make_manager()
        samples = (
            [_make_sample(policy_name="solver") for _ in range(8)]
            + [_make_sample(policy_name="rewriter") for _ in range(8)]
            + [_make_sample(policy_name="selector") for _ in range(8)]
        )
        out = mgr._split_by_policy(samples)
        assert set(out.keys()) == {"solver", "rewriter", "selector"}
        for k in out:
            assert len(out[k]) == 8

    def test_mixed_tagged_and_untagged(self):
        """Untagged samples land in __shared__ bucket; tagged go to their named bucket."""
        mgr = _make_manager()
        samples = (
            [_make_sample(policy_name="solver") for _ in range(4)]
            + [_make_sample(policy_name=None) for _ in range(4)]
        )
        out = mgr._split_by_policy(samples)
        assert set(out.keys()) == {"solver", "__shared__"}
        assert len(out["solver"]) == 4
        assert len(out["__shared__"]) == 4

    def test_empty_list(self):
        """Empty input → __shared__ bucket with empty list (no crash)."""
        mgr = _make_manager()
        out = mgr._split_by_policy([])
        assert out == {"__shared__": []}

    def test_single_sample_tagged(self):
        mgr = _make_manager()
        out = mgr._split_by_policy([_make_sample(policy_name="solver")])
        assert out == {"solver": [_make_sample(policy_name="solver")]} or list(out.keys()) == ["solver"]

    def test_preserves_order_within_bucket(self):
        """Order of samples within each bucket follows their order in the input."""
        mgr = _make_manager()
        samples = []
        for i in range(6):
            s = _make_sample(policy_name="solver" if i % 2 == 0 else "rewriter")
            s.index = i
            samples.append(s)
        out = mgr._split_by_policy(samples)
        assert [s.index for s in out["solver"]] == [0, 2, 4]
        assert [s.index for s in out["rewriter"]] == [1, 3, 5]


# ────────────────────────────────────────────────────────────────────────────
# _post_process_rewards — args fallback
# ────────────────────────────────────────────────────────────────────────────


class TestPostProcessRewardsArgsFallback:
    def test_legacy_no_policy_args_uses_self_args(self):
        """Single-policy callers (no policy_args kwarg) read self.args.
        Bit-for-bit regression check."""
        mgr = _make_manager()
        mgr.args = _default_args(advantage_estimator="reinforce", rewards_normalization=False)
        samples = [_make_sample(reward=1.0) for _ in range(4)]
        raw, rewards = mgr._post_process_rewards(samples)
        # No normalization → raw == rewards
        assert raw == [1.0] * 4
        assert rewards == [1.0] * 4

    def test_explicit_policy_args_overrides_self_args(self):
        """Multi-policy: when policy_args is given, that namespace is used for
        n_samples_per_prompt / advantage_estimator / etc., not self.args."""
        mgr = _make_manager()
        mgr.args = _default_args(advantage_estimator="reinforce")  # would skip group-norm
        # Per-policy args triggers grpo group-norm
        p_args = _default_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=4,
            rollout_batch_size=1,
        )
        rewards_raw = [1.0, 1.0, 0.0, 0.0]
        samples = [_make_sample(reward=r) for r in rewards_raw]
        raw, rewards = mgr._post_process_rewards(samples, policy_args=p_args)

        # Group-norm with mean=0.5: [0.5, 0.5, -0.5, -0.5]
        assert raw == rewards_raw
        assert rewards == pytest.approx([0.5, 0.5, -0.5, -0.5])

    def test_custom_hook_receives_resolved_args(self):
        """The custom_reward_post_process_func gets `policy_args` if provided,
        else self.args. Verifies the hook sees the right namespace."""
        mgr = _make_manager()
        seen = {}

        def hook(args, samples):
            seen["args"] = args
            return ([], [])

        mgr.custom_reward_post_process_func = hook
        sentinel = _default_args(advantage_estimator="custom")

        mgr._post_process_rewards([], policy_args=sentinel)
        assert seen["args"] is sentinel

        mgr._post_process_rewards([])
        assert seen["args"] is mgr.args

    def test_n_samples_per_prompt_drives_reshape(self):
        """Per-policy n_samples_per_prompt controls the group-norm grouping.
        With n=2, [1,0, 1,0] → groups of 2 → centered to [0.5,-0.5, 0.5,-0.5]."""
        mgr = _make_manager()
        p_args = _default_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=2,
            rollout_batch_size=2,
        )
        samples = [_make_sample(reward=r) for r in [1.0, 0.0, 1.0, 0.0]]
        _, rewards = mgr._post_process_rewards(samples, policy_args=p_args)
        assert rewards == pytest.approx([0.5, -0.5, 0.5, -0.5])


# ────────────────────────────────────────────────────────────────────────────
# _convert_samples_to_train_data — policy_args propagation
# ────────────────────────────────────────────────────────────────────────────


class TestConvertSamplesToTrainDataPropagation:
    def test_custom_hook_receives_resolved_args(self):
        """When custom_convert_samples_to_train_data_func is set, it's called
        with policy_args (if given) else self.args."""
        mgr = _make_manager()
        seen = {}

        def hook(args, samples):
            seen["args"] = args
            return {}

        mgr.custom_convert_samples_to_train_data_func = hook
        sentinel = _default_args(advantage_estimator="custom")

        mgr._convert_samples_to_train_data([], policy_args=sentinel)
        assert seen["args"] is sentinel

        mgr._convert_samples_to_train_data([])
        assert seen["args"] is mgr.args

    def test_policy_args_propagates_to_post_process_rewards(self):
        """The inner _post_process_rewards call must receive the same policy_args."""
        mgr = _make_manager()
        captured = {}

        original = mgr._post_process_rewards

        def spy(samples, policy_args=None):
            captured["policy_args"] = policy_args
            return ([], [])

        mgr._post_process_rewards = spy
        sentinel = _default_args()

        # Custom convert_samples hook short-circuits before _post_process_rewards
        mgr.custom_convert_samples_to_train_data_func = None
        # Empty samples avoids the loss-mask loop crashes
        try:
            mgr._convert_samples_to_train_data([], policy_args=sentinel)
        except Exception:
            pass  # IndexError on samples[0] is fine — we only care _post_process_rewards was called

        assert captured.get("policy_args") is sentinel


# ────────────────────────────────────────────────────────────────────────────
# generate() dispatch — legacy vs multi-policy
# ────────────────────────────────────────────────────────────────────────────


def _stub_pipeline(mgr):
    """Patch generate()'s side-effect calls so we can drive it with controlled inputs.
    Returns the patches as context managers, the caller does the `with` chain."""
    # _get_rollout_data normally runs the full rollout fn; we feed it samples directly.
    # _save_debug_rollout_data and _log_rollout_data are pure side-effects on disk/log.
    return [
        patch.object(mgr, "_save_debug_rollout_data", return_value=None),
        patch("slime.ray.rollout._log_rollout_data", return_value=None),
        patch.object(mgr, "_split_train_data_by_dp", side_effect=lambda data, dp: [("batch", dp)]),
        patch.object(mgr, "_convert_samples_to_train_data", side_effect=lambda samples, policy_args=None: {"samples": samples, "policy_args": policy_args}),
    ]


class TestGenerateDispatchLegacy:
    def test_legacy_path_when_no_policies_registered(self):
        """_policy_to_server empty (e.g. single-policy train.py) → returns a list,
        bit-for-bit shape match with pre-fork generate()."""
        mgr = _make_manager()
        mgr.health_monitoring_resume = MagicMock()
        samples = [_make_sample() for _ in range(4)]

        with patch.object(mgr, "_get_rollout_data", return_value=(samples, {})):
            with _stub_pipeline(mgr)[0], _stub_pipeline(mgr)[1]:
                with patch.object(
                    mgr, "_split_train_data_by_dp", return_value=["batch0", "batch1"]
                ) as split_mock, patch.object(
                    mgr, "_convert_samples_to_train_data", return_value={"k": "v"}
                ) as conv_mock:
                    result = mgr.generate(rollout_id=0)

        assert isinstance(result, list)
        assert result == ["batch0", "batch1"]
        # Conversion called WITHOUT policy_args (legacy path)
        conv_mock.assert_called_once_with(samples)
        # Split called with manager-global dp_size
        split_mock.assert_called_once_with({"k": "v"}, 2)


class TestGenerateDispatchMultiPolicy:
    def _mgr_with_two_policies(self):
        mgr = _make_manager()
        mgr.health_monitoring_resume = MagicMock()
        mgr._policy_to_server = {"solver": "solver", "rewriter": "rewriter"}
        mgr._policy_args = {
            "solver": _default_args(n_samples_per_prompt=4),
            "rewriter": _default_args(n_samples_per_prompt=2),
        }
        mgr._policy_train_parallel_config = {
            "solver": {"dp_size": 4},
            "rewriter": {"dp_size": 2},
        }
        return mgr

    def test_returns_dict_when_samples_tagged(self):
        mgr = self._mgr_with_two_policies()
        samples = (
            [_make_sample(policy_name="solver") for _ in range(4)]
            + [_make_sample(policy_name="rewriter") for _ in range(4)]
        )

        with patch.object(mgr, "_get_rollout_data", return_value=(samples, {})), \
             patch.object(mgr, "_save_debug_rollout_data"), \
             patch("slime.ray.rollout._log_rollout_data"), \
             patch.object(mgr, "_convert_samples_to_train_data", side_effect=lambda s, policy_args=None: {"samples": s, "policy_args": policy_args}), \
             patch.object(mgr, "_split_train_data_by_dp", side_effect=lambda data, dp: [(data, dp)]):
            result = mgr.generate(rollout_id=0)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"solver", "rewriter"}
        # Each policy got its own dp_size
        assert result["solver"][0][1] == 4  # solver dp_size
        assert result["rewriter"][0][1] == 2  # rewriter dp_size

    def test_passes_per_policy_args_to_convert(self):
        """Each bucket's _convert_samples_to_train_data call gets that policy's args."""
        mgr = self._mgr_with_two_policies()
        samples = (
            [_make_sample(policy_name="solver") for _ in range(4)]
            + [_make_sample(policy_name="rewriter") for _ in range(2)]
        )
        captured = []

        def fake_convert(s, policy_args=None):
            captured.append(policy_args)
            return {"k": s}

        with patch.object(mgr, "_get_rollout_data", return_value=(samples, {})), \
             patch.object(mgr, "_save_debug_rollout_data"), \
             patch("slime.ray.rollout._log_rollout_data"), \
             patch.object(mgr, "_convert_samples_to_train_data", side_effect=fake_convert), \
             patch.object(mgr, "_split_train_data_by_dp", side_effect=lambda data, dp: [data]):
            mgr.generate(rollout_id=0)

        assert len(captured) == 2
        # Both calls received per-policy args (the right namespace each time)
        assert captured[0] is mgr._policy_args["solver"] or captured[0] is mgr._policy_args["rewriter"]
        assert captured[1] is mgr._policy_args["solver"] or captured[1] is mgr._policy_args["rewriter"]
        assert captured[0] is not captured[1]

    def test_raises_when_multi_policy_but_samples_untagged(self):
        """Multi-policy mode active but rollout fn forgot to tag → fail fast with
        a clear message naming the registered policies."""
        mgr = self._mgr_with_two_policies()
        samples = [_make_sample(policy_name=None) for _ in range(4)]

        with patch.object(mgr, "_get_rollout_data", return_value=(samples, {})), \
             patch.object(mgr, "_save_debug_rollout_data"), \
             patch("slime.ray.rollout._log_rollout_data"):
            with pytest.raises(ValueError, match="Multi-policy mode active"):
                mgr.generate(rollout_id=0)

    def test_raises_on_unregistered_policy_name(self):
        """Sample tagged with a policy that wasn't register_policy'd → raise."""
        mgr = self._mgr_with_two_policies()
        samples = [_make_sample(policy_name="ghost") for _ in range(2)]

        with patch.object(mgr, "_get_rollout_data", return_value=(samples, {})), \
             patch.object(mgr, "_save_debug_rollout_data"), \
             patch("slime.ray.rollout._log_rollout_data"), \
             patch.object(mgr, "_convert_samples_to_train_data", return_value={}), \
             patch.object(mgr, "_split_train_data_by_dp", return_value=[]):
            with pytest.raises(ValueError, match="not a registered policy"):
                mgr.generate(rollout_id=0)

    def test_mixed_shared_and_tagged_warns_and_drops_shared(self, caplog):
        """v1 doesn't support broadcasting __shared__ samples to all policies;
        warn loudly and drop them. Tagged samples still route correctly."""
        import logging as _logging
        mgr = self._mgr_with_two_policies()
        samples = (
            [_make_sample(policy_name="solver") for _ in range(4)]
            + [_make_sample(policy_name=None) for _ in range(2)]
        )

        with patch.object(mgr, "_get_rollout_data", return_value=(samples, {})), \
             patch.object(mgr, "_save_debug_rollout_data"), \
             patch("slime.ray.rollout._log_rollout_data"), \
             patch.object(mgr, "_convert_samples_to_train_data", side_effect=lambda s, policy_args=None: {"samples": s}), \
             patch.object(mgr, "_split_train_data_by_dp", side_effect=lambda data, dp: [data]):
            with caplog.at_level(_logging.WARNING, logger="slime.ray.rollout"):
                result = mgr.generate(rollout_id=0)

        assert "solver" in result
        assert "__shared__" not in result
        assert "rewriter" not in result  # had no tagged samples
        assert "mixed shared+split" in caplog.text or "no policy_name" in caplog.text


# ────────────────────────────────────────────────────────────────────────────
# Legacy regression — bit-for-bit single-policy path
# ────────────────────────────────────────────────────────────────────────────


class TestLegacyRegression:
    def test_post_process_rewards_default_arg_unchanged(self):
        """No policy_args kwarg → identical to pre-fork behavior."""
        mgr = _make_manager()
        mgr.args = _default_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=2,
            rollout_batch_size=2,
        )
        samples = [_make_sample(reward=r) for r in [1.0, 0.0, 1.0, 0.0]]
        _, rewards = mgr._post_process_rewards(samples)
        # mean=0.5 per group of 2 → [0.5, -0.5, 0.5, -0.5]
        assert rewards == pytest.approx([0.5, -0.5, 0.5, -0.5])

    def test_split_by_policy_no_op_for_legacy_samples(self):
        """Samples produced by the legacy rollout fn never set policy_name →
        bucketing is a no-op (same list under __shared__ key)."""
        mgr = _make_manager()
        samples = [_make_sample() for _ in range(8)]
        out = mgr._split_by_policy(samples)
        assert out == {"__shared__": samples}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
