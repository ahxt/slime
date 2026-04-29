"""PolicyRegistry — driver-side container of all trainable policies in a multi-policy run.

Holds N PolicyHandles, each wrapping a Megatron RayTrainGroup paired 1:1 with one
sglang server. The driver (train_multi_policy.py) walks this registry per rollout_id.

NOT a Ray actor — just a dict-of-handles in the driver process.
"""

from __future__ import annotations

import dataclasses
import logging
from argparse import Namespace
from typing import Any

import ray

from slime.utils.policy_config import PolicyConfig, config_to_namespace

# Re-export for back-compat with anyone importing config_to_namespace from this module
__all__ = ["PolicyHandle", "PolicyRegistry", "config_to_namespace"]

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PolicyHandle:
    """One trainable Megatron actor + its 1:1-paired sglang engine handle."""

    config: PolicyConfig
    args: Namespace  # PolicyConfig projected onto a Namespace for downstream Megatron code
    train_group: Any  # RayTrainGroup


class PolicyRegistry:
    def __init__(
        self,
        configs: list[PolicyConfig],
        base_args: Namespace,
        pgs: dict,
        rollout_manager,
    ):
        # Local import to avoid circulars and to defer slime.ray loading.
        from slime.ray.placement_group import allocate_train_group

        self._policies: dict[str, PolicyHandle] = {}
        for cfg in configs:
            args_p = config_to_namespace(cfg, base_args)
            train_group = allocate_train_group(
                args=args_p,
                num_nodes=cfg.megatron_num_nodes,
                num_gpus_per_node=cfg.num_gpus_per_node,
                pg=pgs[cfg.name],
                role=cfg.role,
            )
            handle = PolicyHandle(config=cfg, args=args_p, train_group=train_group)
            self._policies[cfg.name] = handle

            # Bind this policy to its sglang server on the manager side.
            # See Step 4 in plan.md: RolloutManager.register_policy(name, server_name, args).
            ray.get(
                rollout_manager.register_policy.remote(
                    cfg.name, cfg.sglang_server, args_p
                )
            )

    # ── lookups ──
    def all(self) -> list[PolicyHandle]:
        return list(self._policies.values())

    def names(self) -> list[str]:
        return list(self._policies.keys())

    def get(self, name: str) -> PolicyHandle:
        return self._policies[name]
