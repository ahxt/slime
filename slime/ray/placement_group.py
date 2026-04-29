import logging
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .actor_group import RayTrainGroup
from .rollout import RolloutManager

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


def create_placement_groups(args):
    """Create placement groups for actor, critic, and rollout engines."""

    num_gpus = 0
    if args.debug_train_only:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    elif args.debug_rollout_only:
        num_gpus = args.rollout_num_gpus
        rollout_offset = 0
    elif args.colocate:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    else:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)
    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]

    result = {
        "actor": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }

    result["critic"] = result["actor"] if args.use_critic else None

    return result


def allocate_train_group(args, num_nodes, num_gpus_per_node, pg, role="actor"):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,
        role=role,
    )


def create_training_models(args, pgs, rollout_manager):
    actor_args = args
    if args.megatron_config_path is not None:
        from slime.utils.arguments import parse_megatron_role_args

        actor_args = parse_megatron_role_args(args, args.megatron_config_path, role="actor")

    actor_model = allocate_train_group(
        args=actor_args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
    )

    critic_model = None
    if args.use_critic:
        from slime.utils.arguments import parse_megatron_role_args

        critic_args = (
            parse_megatron_role_args(args, args.megatron_config_path, role="critic")
            if args.megatron_config_path is not None
            else args
        )
        critic_model = allocate_train_group(
            args=critic_args,
            num_nodes=args.critic_num_nodes,
            num_gpus_per_node=args.critic_num_gpus_per_node,
            pg=pgs["critic"],
            role="critic",
        )
        critic_start_rollout_ids = ray.get(critic_model.async_init(critic_model.args, role="critic", with_ref=False))

    actor_start_rollout_ids = ray.get(
        actor_model.async_init(
            actor_args,
            role="actor",
            with_ref=actor_args.kl_coef != 0 or actor_args.use_kl_loss,
            with_opd_teacher=actor_args.use_opd and actor_args.opd_type == "megatron",
        )
    )
    # TODO how to decide rollout start id when critic is involved? For now we just require user to specify it via args.
    if args.use_critic:
        start_rollout_ids = critic_start_rollout_ids
    else:
        start_rollout_ids = actor_start_rollout_ids

    assert len(set(start_rollout_ids)) == 1

    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    actor_model.set_rollout_manager(rollout_manager)
    if args.use_critic:
        critic_model.set_rollout_manager(rollout_manager)

    if args.rollout_global_dataset:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    return actor_model, critic_model


def create_rollout_manager(args, pg):
    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0,
    ).remote(args, pg)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    return rollout_manager, num_rollout_per_epoch


# ════════════════════════════════════════════════════════════════════════════
# Multi-policy additions (Step 7 — multi-actor placement groups)
#
# Pure additions — the existing create_placement_groups / create_rollout_manager
# above are unchanged. The legacy single-policy train.py path stays bit-for-bit
# identical.
#
# create_rollout_manager_multi is a stub that points at Step 4 (RolloutManager
# changes) where the real implementation lands. Step 7 ships placement groups
# only.
# ════════════════════════════════════════════════════════════════════════════


def create_placement_groups_multi(args, policy_configs):
    """Carve a single global placement group into per-policy actor slices + a
    rollout slice. Pure additive entry point used by train_multi_policy.py.

    Cluster sizes derived from policy_configs (config.yaml):
      actor_gpus   = sum(c.megatron_num_nodes * c.num_gpus_per_node for c in cfgs)
      rollout_gpus = sum(c.sglang_num_nodes   * c.num_gpus_per_node for c in cfgs)
      total = max(actor_gpus, rollout_gpus)  if args.colocate
            = actor_gpus + rollout_gpus      otherwise

    Returns dict keyed by policy name + "rollout":
      result["<policy_name>"] = (pg, bundle_indices, gpu_ids)  for that policy's actor
      result["rollout"]       = (pg, bundle_indices, gpu_ids)  for sglang engines

    With --colocate, the rollout slice is the entire pool (sglang shares GPUs
    with actors via fractional Ray resources). Without --colocate, rollout gets
    the contiguous range right after the last actor slice.
    """
    actor_gpus = sum(c.megatron_num_nodes * c.num_gpus_per_node for c in policy_configs)
    rollout_gpus = sum(c.sglang_num_nodes * c.num_gpus_per_node for c in policy_configs)

    if args.colocate:
        total = max(actor_gpus, rollout_gpus)
    else:
        total = actor_gpus + rollout_gpus

    logger.info(
        f"create_placement_groups_multi: {len(policy_configs)} policies, "
        f"actor_gpus={actor_gpus}, rollout_gpus={rollout_gpus}, "
        f"total={total}, colocate={args.colocate}"
    )

    pg, idxs, gpus = _create_placement_group(total)

    cursor = 0
    result: dict[str, tuple] = {}
    for c in policy_configs:
        ss = c.megatron_num_nodes * c.num_gpus_per_node
        result[c.name] = (
            pg,
            list(idxs[cursor : cursor + ss]),
            list(gpus[cursor : cursor + ss]),
        )
        cursor += ss

    if args.colocate:
        result["rollout"] = (pg, list(idxs), list(gpus))
    else:
        result["rollout"] = (pg, list(idxs[cursor:]), list(gpus[cursor:]))

    return result


def create_rollout_manager_multi(args, pg, sglang_config):
    """Multi-policy rollout manager — STUB until Step 4 lands.

    The real implementation requires modifications to RolloutManager:
      - RolloutManager.register_policy(name, server_name, args)
      - RolloutManager._get_server(name)
      - RolloutManager.get_engines_and_lock(policy_name=...)
      - per-policy buffer split (Step 2)

    Step 7 only ships the placement-group side. Calling this raises so callers
    fail fast with a clear pointer to Step 4. Run via the legacy train.py for
    single-policy in the meantime.
    """
    raise NotImplementedError(
        "create_rollout_manager_multi requires Step 4 (RolloutManager per-policy "
        "registration + name-based engine routing). Step 7 only ships placement "
        "groups. See plan.md for status. Use train.py for single-policy runs."
    )
