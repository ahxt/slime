"""Level-1 tests for the multi-policy config layer.

Pure-Python: no Ray, no GPUs, no slime internals beyond SglangConfig (which is
upstream and self-contained). Run with:

    python -m pytest tests/test_multi_policy_config.py -v

or directly:

    python tests/test_multi_policy_config.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile

import pytest
import yaml

# Make sure the worktree's slime/ is importable when running directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from slime.utils.policy_config import (
    PolicyConfig,
    _validate_shared_buffer_consistency,
    _validate_unique_names,
    _validate_unique_sglang_servers,
    build_sglang_config_from_policies,
    derive_cluster_sizing,
    derive_policy_slices,
    parse_policy_configs,
    validate_policy_config,
)


EXAMPLE_CONFIG = os.path.join(
    _REPO_ROOT, "examples", "multi_policy_multi_agent", "config.yaml"
)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _minimal_actor(**overrides) -> PolicyConfig:
    """Build a minimal valid PolicyConfig; tests override specific fields."""
    base = dict(
        name="solver",
        role="actor",
        hf_checkpoint="/x",
        sglang_server="solver",
        buffer_mode="split",
        num_gpus_per_node=8,
        megatron_num_nodes=1,
        sglang_num_nodes=1,
        sglang={
            "update_weights": True,
            "num_gpus_per_engine": 8,
            "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
        },
    )
    base.update(overrides)
    return PolicyConfig(**base)


def _write_yaml(data: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.dump(data, f)
    return path


# ────────────────────────────────────────────────────────────────────────────
# Parser end-to-end on the actual example
# ────────────────────────────────────────────────────────────────────────────


class TestExampleConfig:
    def test_parses_three_policies(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        assert [c.name for c in cfgs] == ["solver", "rewriter", "selector"]

    def test_each_is_actor(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        assert all(c.role == "actor" for c in cfgs)

    def test_megatron_fields_flattened(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        solver = cfgs[0]
        assert solver.tensor_model_parallel_size == 4
        assert solver.expert_model_parallel_size == 8
        assert solver.lr == 1.0e-6
        assert solver.optimizer_cpu_offload is True
        assert solver.advantage_estimator == "grpo"
        assert solver.n_samples_per_prompt == 8

    def test_sglang_kept_as_dict_with_default_model_path(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        for c in cfgs:
            assert isinstance(c.sglang, dict)
            assert c.sglang["model_path"] == c.hf_checkpoint
            assert c.sglang["update_weights"] is True

    def test_sglang_server_defaults_to_policy_name(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        for c in cfgs:
            assert c.sglang_server == c.name

    def test_buffer_mode_split(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        assert all(c.buffer_mode == "split" for c in cfgs)

    def test_placement_fields(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        for c in cfgs:
            assert c.num_gpus_per_node == 8
            assert c.megatron_num_nodes == 1
            assert c.sglang_num_nodes == 1


# ────────────────────────────────────────────────────────────────────────────
# Per-entry validator
# ────────────────────────────────────────────────────────────────────────────


class TestValidatePolicyConfig:
    def test_minimal_actor_passes(self):
        validate_policy_config(_minimal_actor())  # no raise

    def test_critic_role_rejected(self):
        cfg = _minimal_actor(role="critic")
        with pytest.raises(ValueError, match="only role='actor'"):
            validate_policy_config(cfg)

    def test_unknown_role_rejected(self):
        cfg = _minimal_actor(role="wizard")
        with pytest.raises(ValueError, match="only role='actor'"):
            validate_policy_config(cfg)

    def test_missing_sglang_server_rejected(self):
        cfg = _minimal_actor(sglang_server=None)
        with pytest.raises(ValueError, match="actor requires sglang_server"):
            validate_policy_config(cfg)

    def test_missing_hf_checkpoint_rejected(self):
        cfg = _minimal_actor(hf_checkpoint="")
        with pytest.raises(ValueError, match="hf_checkpoint required"):
            validate_policy_config(cfg)

    def test_bad_buffer_mode_rejected(self):
        cfg = _minimal_actor(buffer_mode="duplex")
        with pytest.raises(ValueError, match="buffer_mode must be"):
            validate_policy_config(cfg)

    def test_sglang_placement_mismatch_rejected(self):
        # 1 node × 8 gpus_per_node = 8, but server_groups sums to 4
        cfg = _minimal_actor(
            sglang={
                "update_weights": True,
                "num_gpus_per_engine": 4,
                "server_groups": [{"worker_type": "regular", "num_gpus": 4}],
            }
        )
        with pytest.raises(ValueError, match="must equal sum of sglang.server_groups"):
            validate_policy_config(cfg)

    def test_sglang_placement_match_two_groups(self):
        # 1 node × 8 gpus_per_node = 8 = 4 + 4
        cfg = _minimal_actor(
            sglang={
                "update_weights": True,
                "num_gpus_per_engine": 4,
                "server_groups": [
                    {"worker_type": "prefill", "num_gpus": 4},
                    {"worker_type": "decode", "num_gpus": 4},
                ],
            }
        )
        validate_policy_config(cfg)  # no raise


# ────────────────────────────────────────────────────────────────────────────
# Cross-policy validators
# ────────────────────────────────────────────────────────────────────────────


class TestCrossPolicyValidators:
    def test_unique_names(self):
        cfgs = [
            _minimal_actor(name="a", sglang_server="a"),
            _minimal_actor(name="a", sglang_server="b"),  # duplicate name
        ]
        with pytest.raises(ValueError, match="duplicate policy names"):
            _validate_unique_names(cfgs)

    def test_unique_sglang_servers(self):
        cfgs = [
            _minimal_actor(name="a", sglang_server="X"),
            _minimal_actor(name="b", sglang_server="X"),  # collision
        ]
        with pytest.raises(ValueError, match="cannot push to the same"):
            _validate_unique_sglang_servers(cfgs)

    def test_shared_buffer_estimator_must_match(self):
        cfgs = [
            _minimal_actor(name="a", sglang_server="a",
                           buffer_mode="shared", advantage_estimator="grpo"),
            _minimal_actor(name="b", sglang_server="b",
                           buffer_mode="shared", advantage_estimator="gspo"),
        ]
        with pytest.raises(ValueError, match="advantage_estimator"):
            _validate_shared_buffer_consistency(cfgs)

    def test_shared_buffer_n_samples_must_match(self):
        cfgs = [
            _minimal_actor(name="a", sglang_server="a",
                           buffer_mode="shared", n_samples_per_prompt=4),
            _minimal_actor(name="b", sglang_server="b",
                           buffer_mode="shared", n_samples_per_prompt=8),
        ]
        with pytest.raises(ValueError, match="n_samples_per_prompt"):
            _validate_shared_buffer_consistency(cfgs)

    def test_split_buffer_no_constraint(self):
        cfgs = [
            _minimal_actor(name="a", sglang_server="a",
                           buffer_mode="split", advantage_estimator="grpo",
                           n_samples_per_prompt=4),
            _minimal_actor(name="b", sglang_server="b",
                           buffer_mode="split", advantage_estimator="gspo",
                           n_samples_per_prompt=8),
        ]
        _validate_shared_buffer_consistency(cfgs)  # no raise


# ────────────────────────────────────────────────────────────────────────────
# parse_policy_configs error paths (full-file YAML)
# ────────────────────────────────────────────────────────────────────────────


class TestParserErrors:
    def test_missing_top_level_policies(self):
        path = _write_yaml({"foo": []})
        with pytest.raises(ValueError, match="top-level 'policies' list"):
            parse_policy_configs(path)

    def test_policies_not_list(self):
        path = _write_yaml({"policies": "not a list"})
        with pytest.raises(ValueError, match="top-level 'policies' list"):
            parse_policy_configs(path)

    def test_duplicate_names_at_parse_time(self):
        path = _write_yaml({
            "policies": [
                {
                    "name": "a", "role": "actor", "hf_checkpoint": "/x",
                    "num_gpus_per_node": 8, "megatron_num_nodes": 1, "sglang_num_nodes": 1,
                    "sglang": {"update_weights": True, "num_gpus_per_engine": 8,
                               "server_groups": [{"worker_type": "regular", "num_gpus": 8}]},
                },
                {
                    "name": "a", "role": "actor", "hf_checkpoint": "/y",
                    "num_gpus_per_node": 8, "megatron_num_nodes": 1, "sglang_num_nodes": 1,
                    "sglang": {"update_weights": True, "num_gpus_per_engine": 8,
                               "server_groups": [{"worker_type": "regular", "num_gpus": 8}]},
                },
            ]
        })
        with pytest.raises(ValueError, match="duplicate policy names"):
            parse_policy_configs(path)


# ────────────────────────────────────────────────────────────────────────────
# Cluster sizing
# ────────────────────────────────────────────────────────────────────────────


class TestClusterSizing:
    def test_example_config_sizes(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        actor, rollout, total = derive_cluster_sizing(cfgs, colocate=True)
        assert (actor, rollout, total) == (24, 24, 24)
        actor, rollout, total = derive_cluster_sizing(cfgs, colocate=False)
        assert (actor, rollout, total) == (24, 24, 48)

    def test_split_sizes(self):
        # actor: 1×4 + 2×4 = 12; rollout: 1×4 + 1×4 = 8
        cfgs = [
            _minimal_actor(
                name="a", sglang_server="a",
                num_gpus_per_node=4, megatron_num_nodes=1, sglang_num_nodes=1,
                sglang={"update_weights": True, "num_gpus_per_engine": 4,
                        "server_groups": [{"worker_type": "regular", "num_gpus": 4}]},
            ),
            _minimal_actor(
                name="b", sglang_server="b",
                num_gpus_per_node=4, megatron_num_nodes=2, sglang_num_nodes=1,
                sglang={"update_weights": True, "num_gpus_per_engine": 4,
                        "server_groups": [{"worker_type": "regular", "num_gpus": 4}]},
            ),
        ]
        for c in cfgs:
            validate_policy_config(c)
        actor, rollout, total = derive_cluster_sizing(cfgs, colocate=True)
        assert actor == 12 and rollout == 8 and total == 12
        actor, rollout, total = derive_cluster_sizing(cfgs, colocate=False)
        assert actor == 12 and rollout == 8 and total == 20


# ────────────────────────────────────────────────────────────────────────────
# build_sglang_config_from_policies projection
# ────────────────────────────────────────────────────────────────────────────


class TestBuildSglangConfig:
    def test_three_models_named_after_policies(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sglang_config = build_sglang_config_from_policies(cfgs)
        assert [m.name for m in sglang_config.models] == ["solver", "rewriter", "selector"]

    def test_each_model_is_updatable(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sglang_config = build_sglang_config_from_policies(cfgs)
        assert all(m.update_weights is True for m in sglang_config.models)

    def test_each_model_has_one_server_group_with_8_gpus(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sglang_config = build_sglang_config_from_policies(cfgs)
        for m in sglang_config.models:
            assert len(m.server_groups) == 1
            g = m.server_groups[0]
            assert g.num_gpus == 8
            assert g.worker_type == "regular"

    def test_server_args_folded_into_overrides(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sglang_config = build_sglang_config_from_policies(cfgs)
        for m in sglang_config.models:
            ov = m.server_groups[0].overrides
            # Server-args (mem_fraction_static, attention_backend, etc.) should be folded in
            assert ov["mem_fraction_static"] == 0.7
            assert ov["attention_backend"] == "flash"
            assert ov["chunked_prefill_size"] == 8192
            assert ov["max_running_requests"] == 256
            # Model-level fields must NOT have leaked into overrides
            assert "num_gpus_per_engine" not in ov
            assert "update_weights" not in ov
            assert "server_groups" not in ov
            assert "model_path" not in ov

    def test_per_group_overrides_win(self):
        cfg = _minimal_actor(
            sglang={
                "update_weights": True,
                "num_gpus_per_engine": 8,
                "mem_fraction_static": 0.7,        # model-level
                "server_groups": [{
                    "worker_type": "regular",
                    "num_gpus": 8,
                    "overrides": {"mem_fraction_static": 0.9},  # per-group override
                }],
            }
        )
        sglang_config = build_sglang_config_from_policies([cfg])
        assert sglang_config.models[0].server_groups[0].overrides["mem_fraction_static"] == 0.9


# ────────────────────────────────────────────────────────────────────────────
# Placement-slice math (pure-function version of create_placement_groups_multi)
# ────────────────────────────────────────────────────────────────────────────


class TestDerivePolicySlices:
    def test_three_policies_colocate(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        # Colocate: total=24, all three policies share the 24-GPU pool with the rollout
        slices = derive_policy_slices(cfgs, list(range(24)), colocate=True)
        assert slices["solver"] == list(range(0, 8))
        assert slices["rewriter"] == list(range(8, 16))
        assert slices["selector"] == list(range(16, 24))
        assert slices["rollout"] == list(range(24))   # rollout shares the whole pool

    def test_three_policies_no_colocate(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        # No colocate: total=48, actors get 0..23, rollout gets 24..47
        slices = derive_policy_slices(cfgs, list(range(48)), colocate=False)
        assert slices["solver"] == list(range(0, 8))
        assert slices["rewriter"] == list(range(8, 16))
        assert slices["selector"] == list(range(16, 24))
        assert slices["rollout"] == list(range(24, 48))

    def test_disjoint_actor_slices(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        slices = derive_policy_slices(cfgs, list(range(48)), colocate=False)
        actor_idxs = (
            set(slices["solver"]) | set(slices["rewriter"]) | set(slices["selector"])
        )
        assert len(actor_idxs) == 24  # no overlap between actor slices

    def test_two_policies_disjoint(self):
        cfgs = [
            _minimal_actor(name="a", sglang_server="a",
                           num_gpus_per_node=4, megatron_num_nodes=1, sglang_num_nodes=1,
                           sglang={"update_weights": True, "num_gpus_per_engine": 4,
                                   "server_groups": [{"worker_type": "regular", "num_gpus": 4}]}),
            _minimal_actor(name="b", sglang_server="b",
                           num_gpus_per_node=4, megatron_num_nodes=1, sglang_num_nodes=1,
                           sglang={"update_weights": True, "num_gpus_per_engine": 4,
                                   "server_groups": [{"worker_type": "regular", "num_gpus": 4}]}),
        ]
        for c in cfgs:
            validate_policy_config(c)
        slices = derive_policy_slices(cfgs, list(range(16)), colocate=False)
        assert slices["a"] == list(range(0, 4))
        assert slices["b"] == list(range(4, 8))
        assert slices["rollout"] == list(range(8, 16))

    def test_heterogeneous_actor_sizes(self):
        # Policy a: 1 node × 4 = 4 GPUs; policy b: 2 nodes × 4 = 8 GPUs
        cfgs = [
            _minimal_actor(name="a", sglang_server="a",
                           num_gpus_per_node=4, megatron_num_nodes=1, sglang_num_nodes=1,
                           sglang={"update_weights": True, "num_gpus_per_engine": 4,
                                   "server_groups": [{"worker_type": "regular", "num_gpus": 4}]}),
            _minimal_actor(name="b", sglang_server="b",
                           num_gpus_per_node=4, megatron_num_nodes=2, sglang_num_nodes=1,
                           sglang={"update_weights": True, "num_gpus_per_engine": 4,
                                   "server_groups": [{"worker_type": "regular", "num_gpus": 4}]}),
        ]
        for c in cfgs:
            validate_policy_config(c)
        # actor=12, rollout=8, total no-colocate=20
        actor, rollout, total = derive_cluster_sizing(cfgs, colocate=False)
        assert (actor, rollout, total) == (12, 8, 20)
        slices = derive_policy_slices(cfgs, list(range(20)), colocate=False)
        assert slices["a"] == list(range(0, 4))
        assert slices["b"] == list(range(4, 12))
        assert slices["rollout"] == list(range(12, 20))

    def test_wrong_idx_length_rejected(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        with pytest.raises(ValueError, match="total_idxs has"):
            derive_policy_slices(cfgs, list(range(10)), colocate=True)


# ────────────────────────────────────────────────────────────────────────────
# config_to_namespace (PolicyHandle's projection)
# ────────────────────────────────────────────────────────────────────────────


class TestConfigToNamespace:
    def _base_args(self, **kw):
        from argparse import Namespace
        defaults = dict(
            colocate=True,
            num_rollout=100,
            rollout_batch_size=32,
            offload_train=False,
            offload_rollout=False,
            check_weight_update_equal=False,
            use_fault_tolerance=False,
            save_interval=20,
            eval_interval=None,
            skip_eval_before_train=False,
            rollout_global_dataset=False,
            start_rollout_id=None,
        )
        defaults.update(kw)
        return Namespace(**defaults)

    def test_all_policy_fields_copied(self):
        from slime.utils.policy_config import config_to_namespace
        cfg = _minimal_actor(name="proposer", sglang_server="proposer",
                             lr=5e-6, kl_coef=0.001, advantage_estimator="grpo")
        ns = config_to_namespace(cfg, self._base_args())
        assert ns.name == "proposer"
        assert ns.lr == 5e-6
        assert ns.kl_coef == 0.001
        assert ns.advantage_estimator == "grpo"
        assert ns.tensor_model_parallel_size == cfg.tensor_model_parallel_size

    def test_policy_name_set(self):
        from slime.utils.policy_config import config_to_namespace
        cfg = _minimal_actor(name="proposer", sglang_server="proposer")
        ns = config_to_namespace(cfg, self._base_args())
        assert ns.policy_name == "proposer"

    def test_base_args_preserved(self):
        from slime.utils.policy_config import config_to_namespace
        cfg = _minimal_actor()
        ns = config_to_namespace(cfg, self._base_args(num_rollout=5000, save_interval=100))
        assert ns.num_rollout == 5000
        assert ns.save_interval == 100
        assert ns.colocate is True

    def test_policy_field_overrides_base_arg_with_same_name(self):
        """If base_args and PolicyConfig both have a field, PolicyConfig wins."""
        from slime.utils.policy_config import config_to_namespace
        cfg = _minimal_actor(lr=9.99)
        # base_args also has lr — but config_to_namespace overwrites it
        ns = config_to_namespace(cfg, self._base_args(lr=1e-3))
        assert ns.lr == 9.99


# ────────────────────────────────────────────────────────────────────────────
# Parser edge cases
# ────────────────────────────────────────────────────────────────────────────


class TestParserEdgeCases:
    def test_role_defaults_to_actor(self):
        path = _write_yaml({
            "policies": [{
                "name": "a",
                # role omitted — should default to "actor"
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 8,
                "megatron_num_nodes": 1,
                "sglang_num_nodes": 1,
                "sglang": {"update_weights": True, "num_gpus_per_engine": 8,
                           "server_groups": [{"worker_type": "regular", "num_gpus": 8}]},
            }]
        })
        cfgs = parse_policy_configs(path)
        assert cfgs[0].role == "actor"

    def test_buffer_mode_defaults_to_split(self):
        path = _write_yaml({
            "policies": [{
                "name": "a",
                "role": "actor",
                "hf_checkpoint": "/x",
                # buffer_mode omitted
                "num_gpus_per_node": 8,
                "megatron_num_nodes": 1,
                "sglang_num_nodes": 1,
                "sglang": {"update_weights": True, "num_gpus_per_engine": 8,
                           "server_groups": [{"worker_type": "regular", "num_gpus": 8}]},
            }]
        })
        cfgs = parse_policy_configs(path)
        assert cfgs[0].buffer_mode == "split"

    def test_ref_load_optional(self):
        path = _write_yaml({
            "policies": [{
                "name": "a",
                "role": "actor",
                "hf_checkpoint": "/x",
                "ref_load": "/ref",   # provided
                "num_gpus_per_node": 8,
                "megatron_num_nodes": 1,
                "sglang_num_nodes": 1,
                "sglang": {"update_weights": True, "num_gpus_per_engine": 8,
                           "server_groups": [{"worker_type": "regular", "num_gpus": 8}]},
            }]
        })
        assert parse_policy_configs(path)[0].ref_load == "/ref"

    def test_ref_load_default_none(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        # Example config has ref_load commented out
        assert all(c.ref_load is None for c in cfgs)

    def test_unknown_top_level_field_silently_dropped(self):
        # Top-level fields are picked by name in the parser; unknown ones are simply
        # not forwarded to PolicyConfig. Documenting actual behavior.
        path = _write_yaml({
            "policies": [{
                "name": "a",
                "role": "actor",
                "hf_checkpoint": "/x",
                "wizardly_field": True,    # silently ignored
                "num_gpus_per_node": 8,
                "megatron_num_nodes": 1,
                "sglang_num_nodes": 1,
                "sglang": {"update_weights": True, "num_gpus_per_engine": 8,
                           "server_groups": [{"worker_type": "regular", "num_gpus": 8}]},
            }]
        })
        cfgs = parse_policy_configs(path)
        assert len(cfgs) == 1
        assert not hasattr(cfgs[0], "wizardly_field")

    def test_unknown_megatron_field_rejected(self):
        # Fields inside megatron: are **-spread into PolicyConfig — unknowns raise TypeError.
        path = _write_yaml({
            "policies": [{
                "name": "a",
                "role": "actor",
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 8,
                "megatron_num_nodes": 1,
                "sglang_num_nodes": 1,
                "megatron": {"wizardly_field": True},   # not a PolicyConfig field
                "sglang": {"update_weights": True, "num_gpus_per_engine": 8,
                           "server_groups": [{"worker_type": "regular", "num_gpus": 8}]},
            }]
        })
        with pytest.raises(TypeError):
            parse_policy_configs(path)

    def test_megatron_block_optional(self):
        # If megatron: is missing, all megatron fields use PolicyConfig defaults
        path = _write_yaml({
            "policies": [{
                "name": "a",
                "role": "actor",
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 8,
                "megatron_num_nodes": 1,
                "sglang_num_nodes": 1,
                "sglang": {"update_weights": True, "num_gpus_per_engine": 8,
                           "server_groups": [{"worker_type": "regular", "num_gpus": 8}]},
            }]
        })
        cfgs = parse_policy_configs(path)
        # Defaults from PolicyConfig
        assert cfgs[0].tensor_model_parallel_size == 1
        assert cfgs[0].lr == 1.0e-6
        assert cfgs[0].advantage_estimator == "grpo"


# ────────────────────────────────────────────────────────────────────────────
# build_sglang_config_from_policies — split prefill+decode
# ────────────────────────────────────────────────────────────────────────────


class TestBuildSglangSplit:
    def test_prefill_decode_split(self):
        cfg = _minimal_actor(
            num_gpus_per_node=8,
            megatron_num_nodes=1,
            sglang_num_nodes=1,
            sglang={
                "update_weights": True,
                "num_gpus_per_engine": 4,
                "mem_fraction_static": 0.7,
                "server_groups": [
                    {"worker_type": "prefill", "num_gpus": 4, "num_gpus_per_engine": 4},
                    {"worker_type": "decode", "num_gpus": 4, "num_gpus_per_engine": 4},
                ],
            },
        )
        validate_policy_config(cfg)  # 1×8 == 4+4
        sg = build_sglang_config_from_policies([cfg])
        m = sg.models[0]
        assert len(m.server_groups) == 2
        assert [g.worker_type for g in m.server_groups] == ["prefill", "decode"]
        # Both groups inherit model-level mem_fraction_static
        for g in m.server_groups:
            assert g.overrides["mem_fraction_static"] == 0.7

    def test_no_sglang_block_rejected(self):
        cfg = PolicyConfig(name="a", role="actor", hf_checkpoint="/x", sglang_server="a",
                           num_gpus_per_node=8)
        # No `sglang` sub-block → build raises (validate_policy_config skips since cfg.sglang is None)
        with pytest.raises(ValueError, match="missing 'sglang' sub-block"):
            build_sglang_config_from_policies([cfg])


# ────────────────────────────────────────────────────────────────────────────
# Full integration smoke (parser → validate → sizing → slices → sglang config)
# ────────────────────────────────────────────────────────────────────────────


class TestEndToEndPipeline:
    def test_full_pipeline_on_example_yaml(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)

        # Cluster sizing
        actor, rollout, total = derive_cluster_sizing(cfgs, colocate=True)
        assert (actor, rollout, total) == (24, 24, 24)

        # Slicing
        slices = derive_policy_slices(cfgs, list(range(total)), colocate=True)
        assert set(slices.keys()) == {"solver", "rewriter", "selector", "rollout"}

        # Sglang projection
        sg = build_sglang_config_from_policies(cfgs)
        assert [m.name for m in sg.models] == ["solver", "rewriter", "selector"]

        # Cross-validation: each policy's slice size matches its actor footprint
        for c in cfgs:
            assert len(slices[c.name]) == c.megatron_num_nodes * c.num_gpus_per_node


# ────────────────────────────────────────────────────────────────────────────
# Determinism / idempotence / roundtrip
# ────────────────────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_parse_twice_equal(self):
        a = parse_policy_configs(EXAMPLE_CONFIG)
        b = parse_policy_configs(EXAMPLE_CONFIG)
        assert a == b

    def test_dataclass_asdict_roundtrip(self):
        cfg = _minimal_actor(name="proposer", lr=5e-6, kl_coef=0.001)
        as_dict = dataclasses.asdict(cfg)
        cfg2 = PolicyConfig(**as_dict)
        assert cfg == cfg2

    def test_validate_idempotent(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        for cfg in cfgs:
            validate_policy_config(cfg)
            validate_policy_config(cfg)  # second call must not raise

    def test_derive_cluster_sizing_idempotent(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        a = derive_cluster_sizing(cfgs, colocate=True)
        b = derive_cluster_sizing(cfgs, colocate=True)
        assert a == b


# ────────────────────────────────────────────────────────────────────────────
# Cross-policy invariants in the actual example config
# ────────────────────────────────────────────────────────────────────────────


class TestExampleInvariants:
    def test_all_three_share_base_model(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        ckpts = {c.hf_checkpoint for c in cfgs}
        assert len(ckpts) == 1, f"all three policies should branch from one base, got {ckpts}"

    def test_num_gpus_per_node_consistent(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        ngs = {c.num_gpus_per_node for c in cfgs}
        assert len(ngs) == 1, f"node-level GPU count should be uniform, got {ngs}"

    def test_save_dirs_per_policy_distinct(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        saves = [c.save for c in cfgs]
        assert len(saves) == len(set(saves)), f"save dirs must be distinct: {saves}"

    def test_n_samples_per_prompt_matches_num_parallel(self):
        """Bug 1 fix regression: rollout_with_multi_agents.py num_parallel=8 must match
        each policy's n_samples_per_prompt=8 in config.yaml so GRPO group-norm reshape
        stays on the fast path."""
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        for cfg in cfgs:
            assert cfg.n_samples_per_prompt == 8, (
                f"{cfg.name}: n_samples_per_prompt={cfg.n_samples_per_prompt}, "
                f"should be 8 to match MULTI_AGENT_CONFIGS['num_parallel']"
            )

    def test_selector_n_samples_not_one(self):
        """Bug 2 fix regression: selector.n_samples_per_prompt must be > 1 so GRPO
        group-norm produces non-zero advantage."""
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        selector = next(c for c in cfgs if c.name == "selector")
        assert selector.n_samples_per_prompt > 1, (
            "selector with n_samples_per_prompt=1 makes GRPO group-norm degenerate "
            "(within-group variance is 0). Bug 2 fix requires N parallel selectors."
        )


# ────────────────────────────────────────────────────────────────────────────
# Launcher ↔ config consistency (catches the GPU sizing bug we found earlier)
# ────────────────────────────────────────────────────────────────────────────


class TestLauncherConsistency:
    LAUNCHER_PATH = os.path.join(
        _REPO_ROOT, "examples", "multi_policy_multi_agent",
        "run-qwen3-30B-A3B-multi-policy-multi-agent.sh",
    )

    def test_launcher_num_gpus_matches_derived(self):
        """The launcher's NUM_GPUS must equal derive_cluster_sizing(colocate=True)
        because the launcher passes --colocate. If config.yaml changes (e.g. add a
        4th policy), the launcher must be updated to match."""
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        actor, rollout, total_colocate = derive_cluster_sizing(cfgs, colocate=True)

        with open(self.LAUNCHER_PATH) as f:
            launcher = f.read()

        # Find NUM_GPUS=<n> declaration
        import re
        m = re.search(r"^NUM_GPUS=(\d+)", launcher, re.MULTILINE)
        assert m, "launcher must declare NUM_GPUS=<int>"
        launcher_num_gpus = int(m.group(1))
        assert launcher_num_gpus == total_colocate, (
            f"launcher NUM_GPUS={launcher_num_gpus} but config-derived total_colocate"
            f"={total_colocate}. Update launcher to match config.yaml."
        )

    def test_launcher_uses_colocate(self):
        with open(self.LAUNCHER_PATH) as f:
            launcher = f.read()
        # NUM_GPUS=24 only matches the colocate sizing; without --colocate it'd be 48
        assert "--colocate" in launcher, "launcher passes --colocate to train_multi_policy.py"

    def test_launcher_passes_config_flag(self):
        with open(self.LAUNCHER_PATH) as f:
            launcher = f.read()
        assert "--config" in launcher
        assert "config.yaml" in launcher


# ────────────────────────────────────────────────────────────────────────────
# Static checks on the example's Python source (regression for bug fixes)
# Uses ast — no module imports, robust to missing deps (transformers, etc.)
# ────────────────────────────────────────────────────────────────────────────


class TestExampleSourceStatic:
    AGENT_SYSTEM = os.path.join(
        _REPO_ROOT, "examples", "multi_policy_multi_agent", "agent_system.py"
    )
    ROLLOUT_FN = os.path.join(
        _REPO_ROOT, "examples", "multi_policy_multi_agent", "rollout_with_multi_agents.py"
    )

    def test_agent_system_imports_get_model_url(self):
        """Edit 1 (URL routing): agent_system.py must import get_model_url from sglang_rollout."""
        with open(self.AGENT_SYSTEM) as f:
            src = f.read()
        assert "from slime.rollout.sglang_rollout import get_model_url" in src

    def test_agent_system_uses_get_model_url_for_routing(self):
        """Edit 1: generate_response routes to get_model_url(args, key) not the
        single hardcoded sglang_router_ip:port."""
        with open(self.AGENT_SYSTEM) as f:
            src = f.read()
        assert "get_model_url(args, key)" in src
        # The single-router hardcode must be gone
        assert "sglang_router_ip" not in src
        assert "sglang_router_port" not in src

    def test_agent_system_tags_policy_name(self):
        """Edit 2 (sample tagging): each Sample must get sample.policy_name = key
        before being appended to results_dict so the manager can route."""
        with open(self.AGENT_SYSTEM) as f:
            src = f.read()
        assert "sample.policy_name = key" in src

    def test_agent_system_has_select_worker(self):
        """Edit 3 (selector parallel): a select_worker function must exist
        alongside solver_worker and rewrite_worker."""
        import ast
        with open(self.AGENT_SYSTEM) as f:
            tree = ast.parse(f.read())
        fn_names = {
            n.name for n in ast.walk(tree)
            if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
        }
        assert "select_worker" in fn_names
        assert "solver_worker" in fn_names
        assert "rewrite_worker" in fn_names

    def test_agent_system_no_singleton_assert(self):
        """Edit 3: the old `assert len(args.results_dict["selector"]) == 1` blocks
        N parallel selectors. Must be removed."""
        with open(self.AGENT_SYSTEM) as f:
            src = f.read()
        # The exact assertion text shouldn't appear anywhere
        assert 'len(args.results_dict["selector"]) == 1' not in src

    def test_agent_system_uses_mean_selector_reward_gate(self):
        """Edit 3: with N selectors, the global reward shaping must use the mean,
        not selector[0].reward."""
        with open(self.AGENT_SYSTEM) as f:
            src = f.read()
        assert "mean_selector_reward" in src or "mean(s.reward" in src

    def test_rollout_fn_num_parallel_is_8(self):
        """Bug 1 fix: num_parallel must equal n_samples_per_prompt across all
        policies, which is 8 in the current config.yaml."""
        import ast
        with open(self.ROLLOUT_FN) as f:
            tree = ast.parse(f.read())
        # Find the MULTI_AGENT_CONFIGS dict assignment
        configs_dict = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "MULTI_AGENT_CONFIGS":
                        configs_dict = node.value
                        break
        assert configs_dict is not None, "MULTI_AGENT_CONFIGS not found"
        # Walk the dict literal to find num_parallel
        num_parallel = None
        for k, v in zip(configs_dict.keys, configs_dict.values):
            if isinstance(k, ast.Constant) and k.value == "num_parallel":
                num_parallel = v.value if isinstance(v, ast.Constant) else None
                break
        assert num_parallel == 8, f"num_parallel must be 8, got {num_parallel}"

    def test_rollout_fn_points_at_multi_policy_module(self):
        """The custom_multi_agent_function_path must point at the new multi_policy_multi_agent
        package, not the original multi_agent."""
        with open(self.ROLLOUT_FN) as f:
            src = f.read()
        assert "examples.multi_policy_multi_agent.agent_system.run_agent_system" in src


# ────────────────────────────────────────────────────────────────────────────
# SglangConfig projection — non-trivial types (lists, strings, bools)
# ────────────────────────────────────────────────────────────────────────────


class TestSglangProjectionTypes:
    def test_list_field_preserved(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sg = build_sglang_config_from_policies(cfgs)
        for m in sg.models:
            cuda_bs = m.server_groups[0].overrides["cuda_graph_bs"]
            assert isinstance(cuda_bs, list)
            assert cuda_bs[0] == 1
            assert 256 in cuda_bs

    def test_string_field_preserved(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sg = build_sglang_config_from_policies(cfgs)
        for m in sg.models:
            assert m.server_groups[0].overrides["log_level"] == "info"

    def test_int_field_preserved(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sg = build_sglang_config_from_policies(cfgs)
        for m in sg.models:
            ov = m.server_groups[0].overrides
            assert ov["chunked_prefill_size"] == 8192
            assert ov["max_running_requests"] == 256

    def test_float_field_preserved(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sg = build_sglang_config_from_policies(cfgs)
        for m in sg.models:
            assert m.server_groups[0].overrides["mem_fraction_static"] == 0.7

    def test_model_path_passed_through(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sg = build_sglang_config_from_policies(cfgs)
        for m in sg.models:
            assert m.model_path == "/root/Qwen3-30B-A3B"

    def test_num_gpus_per_engine_passed_through(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sg = build_sglang_config_from_policies(cfgs)
        for m in sg.models:
            assert m.num_gpus_per_engine == 8


# ────────────────────────────────────────────────────────────────────────────
# Edge cases: empty / null / single-policy
# ────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_policies_list(self):
        path = _write_yaml({"policies": []})
        cfgs = parse_policy_configs(path)
        assert cfgs == []

    def test_empty_policies_list_cluster_sizing(self):
        actor, rollout, total = derive_cluster_sizing([], colocate=True)
        assert (actor, rollout, total) == (0, 0, 0)

    def test_empty_yaml_file(self):
        path = _write_yaml({})
        with pytest.raises(ValueError, match="top-level 'policies' list"):
            parse_policy_configs(path)

    def test_single_policy_run_works(self):
        """N=1 should be a valid degenerate case (essentially equivalent to single-policy
        but going through the multi-policy code path)."""
        path = _write_yaml({
            "policies": [{
                "name": "actor",
                "role": "actor",
                "hf_checkpoint": "/m",
                "num_gpus_per_node": 8,
                "megatron_num_nodes": 1,
                "sglang_num_nodes": 1,
                "sglang": {"update_weights": True, "num_gpus_per_engine": 8,
                           "server_groups": [{"worker_type": "regular", "num_gpus": 8}]},
            }]
        })
        cfgs = parse_policy_configs(path)
        assert len(cfgs) == 1
        actor, rollout, total = derive_cluster_sizing(cfgs, colocate=True)
        assert (actor, rollout, total) == (8, 8, 8)


# ────────────────────────────────────────────────────────────────────────────
# PolicyHandle smoke (pure dataclass — no Ray)
# ────────────────────────────────────────────────────────────────────────────


class TestPolicyHandleDataclass:
    def test_construct_with_mock_train_group(self):
        """PolicyHandle is a dataclass; can be constructed without Ray for unit tests."""
        # Skip if ray is not installed (slime.ray.policy_registry imports ray)
        import importlib.util
        if importlib.util.find_spec("ray") is None:
            pytest.skip("ray not installed; skipping PolicyHandle import")

        from argparse import Namespace
        from slime.ray.policy_registry import PolicyHandle
        cfg = _minimal_actor()
        ns = Namespace(policy_name="solver", lr=1e-6)
        handle = PolicyHandle(config=cfg, args=ns, train_group=object())
        assert handle.config.name == "solver"
        assert handle.args.policy_name == "solver"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
