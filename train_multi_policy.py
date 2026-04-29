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
)
from slime.ray.policy_registry import PolicyRegistry
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
    args.rollout_num_gpus = rollout_gpus
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
    # 4. Build the policy registry — N RayTrainGroups, registered against manager
    # ──────────────────────────────────────────────────────────────────────────
    registry = PolicyRegistry(policy_configs, args, pgs, rollout_manager)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Init each policy (loads ckpts, builds weight_updater, etc.) and reconcile
    # ──────────────────────────────────────────────────────────────────────────
    start_rollout_id = _reconcile_start_rollout_ids(registry, rollout_manager)

    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_id

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Initial weight push (each actor → its paired engine)
    # ──────────────────────────────────────────────────────────────────────────
    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())
    for p in registry.all():
        p.train_group.update_weights()
    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # Eval-only path (no training rollouts)
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Train loop
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
        for p in registry.all():
            data = rollout_data.get(p.config.name) or rollout_data.get("__shared__")
            if data is None:
                logger.warning(f"policy {p.config.name} got no rollout data")
                continue
            ray.get(p.train_group.async_train(rollout_id, data))

        # Save (per-policy checkpoint dirs from PolicyConfig.save)
        if should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            for p in registry.all():
                if p.config.save:
                    p.train_group.save_model(
                        rollout_id, force_sync=rollout_id == args.num_rollout - 1
                    )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        # Memory hygiene
        if not args.offload_train:
            for p in registry.all():
                p.train_group.clear_memory()

        # Weight update (serialized — sglang engines on shared GPUs cannot accept
        # two simultaneous broadcasts).
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        for p in registry.all():
            p.train_group.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        # Eval (routes through one trainable policy by convention)
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    # ──────────────────────────────────────────────────────────────────────────
    # 8. Teardown
    # ──────────────────────────────────────────────────────────────────────────
    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


def _reconcile_start_rollout_ids(registry: PolicyRegistry, rollout_manager) -> int:
    """All policies start fresh at 0, or all resume at the same rollout_id.

    Each policy's async_init returns the per-worker start_rollout_ids for its train
    group. Within a group they must agree (Megatron's existing within-group assertion).
    Across policies, divergence triggers a warning and we fall back to the min.

    Also calls set_rollout_manager on each train group so it can post weight updates.
    """
    starts: dict[str, int] = {}
    for p in registry.all():
        ids = ray.get(
            p.train_group.async_init(
                p.args,
                role=p.config.role,
                with_ref=p.args.kl_coef != 0 or p.args.use_kl_loss,
            )
        )
        if len(set(ids)) != 1:
            raise RuntimeError(
                f"{p.config.name}: workers disagree on start_rollout_id: {ids}"
            )
        starts[p.config.name] = ids[0]
        p.train_group.set_rollout_manager(rollout_manager)

    chosen = min(starts.values())
    if any(s != chosen for s in starts.values()):
        logger.warning(
            f"start_rollout_ids diverged across policies: {starts}; using min={chosen}. "
            f"User is responsible for ensuring this is intended."
        )
    return chosen


if __name__ == "__main__":
    args = parse_args()
    train(args)
