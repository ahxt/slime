"""Multi-policy slime driver.

Replaces train.py for runs with N>1 trainable Megatron actors. Each actor is
paired 1:1 with its own sglang engine. Per-policy buffers (split mode), per-policy
weight sync (serialized), per-policy checkpointing.

YAML entry point: --config <path>.yaml. See examples/multi_policy_multi_agent/config.yaml.

Architecture (also in plan.md):

                          ┌──────────────────────┐
                          │   RolloutManager     │  data source, rollout-fn dispatch,
                          │   (1 Ray actor)      │  per-policy buffer + reward post-process
                          └──────────┬───────────┘
                                     │ invokes rollout fn (which posts HTTP to engines below)
                         ┌───────────┼───────────────────────┐
                   ┌─────▼─────┐ ┌───▼───────┐ ┌─────────────▼─────┐
                   │  sglang   │ │  sglang   │ │      sglang       │
                   │  engine A │ │  engine B │ │      engine C     │
                   └─────▲─────┘ └───▲───────┘ └─────────────▲─────┘
                         │ weight    │ weight                │ weight
                         │ push      │ push                  │ push
                   ┌─────┴─────┐ ┌───┴───────┐ ┌─────────────┴─────┐
                   │ actor A   │ │ actor B   │ │     actor C       │
                   │ RayTrain  │ │ RayTrain  │ │     RayTrain      │
                   │  Group    │ │  Group    │ │      Group        │
                   └───────────┘ └───────────┘ └───────────────────┘

Runtime dependencies (Steps 1, 2, 4, 5, 7 in plan.md must land before this runs):
  - slime/utils/types.py:           Sample.policy_name field (Step 1)
  - slime/ray/rollout.py:           _split_by_policy + _post_process_rewards(samples, policy_args)  (Step 2)
                                    _get_server(name) + register_policy(...) +
                                    get_engines_and_lock(policy_name=...)                  (Step 4)
                                    create_rollout_manager_multi(args, pg, sglang_config)  (this file)
  - slime/backends/megatron_utils/actor.py:
                                    update_weights reads args.policy_name                  (Step 5)
  - slime/ray/placement_group.py:   create_placement_groups_multi(args, policy_configs)    (Step 7)
"""

from __future__ import annotations

import logging

import ray

from slime.ray.placement_group import (
    create_placement_groups_multi,
    create_rollout_manager_multi,
    create_training_models_multi,
)
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import (
    configure_logger,
    finish_tracking,
    init_tracking,
    update_tracking_open_metrics,
)
from slime.utils.misc import should_run_periodic_action
from slime.utils.policy_config import (
    build_sglang_config_from_policies,
    derive_cluster_sizing,
    parse_policy_configs,
)

logger = logging.getLogger(__name__)


def _set_multi_policy_global_defaults(args, policy_configs, actor_gpus: int, rollout_gpus: int) -> None:
    """Populate legacy global args that shared rollout code still reads.

    Per-policy Megatron actors get their own namespaces later. These globals are
    for the single RolloutManager/data-source side.
    """
    if not policy_configs:
        raise ValueError("multi-policy config must contain at least one policy")

    if args.hf_checkpoint is None:
        args.hf_checkpoint = policy_configs[0].hf_checkpoint

    num_gpus_per_node_values = {cfg.num_gpus_per_node for cfg in policy_configs}
    if len(num_gpus_per_node_values) != 1:
        raise ValueError(
            "all policies must use the same num_gpus_per_node in v1; got "
            f"{sorted(num_gpus_per_node_values)}"
        )
    num_gpus_per_node = policy_configs[0].num_gpus_per_node

    args.rollout_num_gpus = rollout_gpus
    args.megatron_total_gpus = actor_gpus
    args.num_gpus_per_node = num_gpus_per_node
    if actor_gpus % num_gpus_per_node == 0:
        args.actor_num_nodes = actor_gpus // num_gpus_per_node
        args.actor_num_gpus_per_node = num_gpus_per_node
    else:
        args.actor_num_nodes = 1
        args.actor_num_gpus_per_node = actor_gpus


def train(args):
    configure_logger()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Parse the YAML config and validate v1 restrictions
    # ──────────────────────────────────────────────────────────────────────────
    if not getattr(args, "config", None):
        raise ValueError("train_multi_policy.py requires --config <path>.yaml")

    policy_configs = parse_policy_configs(args.config)
    n_trainable = len(policy_configs)

    if args.check_weight_update_equal and n_trainable > 1:
        raise ValueError(
            "--check-weight-update-equal not supported with multiple policies "
            "(uses _get_updatable_server which assumes a single trainable server)"
        )
    if args.use_fault_tolerance and n_trainable > 1:
        raise ValueError(
            "--use-fault-tolerance not supported with multiple policies in v1"
        )

    # Derive cluster sizing from config and surface it on args for downstream code.
    actor_gpus, rollout_gpus, total_gpus = derive_cluster_sizing(
        policy_configs, colocate=args.colocate
    )
    _set_multi_policy_global_defaults(args, policy_configs, actor_gpus, rollout_gpus)
    logger.info(
        f"cluster sizing (colocate={args.colocate}): "
        f"actor_gpus={actor_gpus}, rollout_gpus={rollout_gpus}, total={total_gpus}"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Allocate placement groups: per-policy actor slices + rollout slice
    # ──────────────────────────────────────────────────────────────────────────
    pgs = create_placement_groups_multi(args, policy_configs)
    init_tracking(args)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Build a SglangConfig from per-policy sglang sub-blocks and start manager
    # ──────────────────────────────────────────────────────────────────────────
    sglang_config = build_sglang_config_from_policies(policy_configs)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager_multi(
        args, pgs["rollout"], sglang_config
    )

    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Build the N RayTrainGroups, register each with the manager, async_init,
    #    and reconcile start_rollout_ids across policies. Returns a dict of
    #    PolicyHandles keyed by policy name. Mirrors create_training_models for
    #    the multi-policy path.
    # ──────────────────────────────────────────────────────────────────────────
    handles = create_training_models_multi(args, pgs, rollout_manager, policy_configs)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Initial weight push (each actor → its paired engine)
    # ──────────────────────────────────────────────────────────────────────────
    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())
    for h in handles.values():
        h.train_group.update_weights()
    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # Eval-only path (no training rollouts)
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Train loop
    # ──────────────────────────────────────────────────────────────────────────
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if (
            args.eval_interval is not None
            and rollout_id == args.start_rollout_id
            and not args.skip_eval_before_train
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))

        # Generate: returns dict[policy_name | "__shared__", list[batch_per_dp]]
        rollout_data = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        # Train each policy independently — no DAG, no external_data (PPO out of scope).
        # Wait for each to finish before starting the next; concurrent train across
        # policies that share GPU slots would conflict.
        for h in handles.values():
            data = rollout_data.get(h.config.name) or rollout_data.get("__shared__")
            if data is None:
                logger.warning(f"policy {h.config.name} got no rollout data")
                continue
            ray.get(h.train_group.async_train(rollout_id, data))

        # Save (per-policy checkpoint dirs from PolicyConfig.save)
        if should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            for h in handles.values():
                if h.config.save:
                    h.train_group.save_model(
                        rollout_id, force_sync=rollout_id == args.num_rollout - 1
                    )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        # Memory hygiene
        if not args.offload_train:
            for h in handles.values():
                h.train_group.clear_memory()

        # Weight update (serialized — sglang engines on shared GPUs cannot accept
        # two simultaneous broadcasts).
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        for h in handles.values():
            h.train_group.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        # Eval (routes through one trainable policy by convention)
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Teardown
    # ──────────────────────────────────────────────────────────────────────────
    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
