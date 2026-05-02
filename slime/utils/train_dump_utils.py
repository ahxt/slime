import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_debug_train_data(args, *, rollout_id, rollout_data):
    if (path_template := args.save_debug_train_data) is not None:
        rank = torch.distributed.get_rank()
        policy_name = getattr(args, "policy_name", None) or "default"
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank, policy_name=policy_name))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                policy_name=policy_name,
                rollout_data=rollout_data,
            ),
            path,
        )


def _serialize_value(v):
    """Recursively serialize a single value to a torch.save-friendly form.

    - Tensor → CPU-detached tensor.
    - PackedSeqParams (Megatron dataclass) → plain dict so loaders don't need Megatron.
    - dict / list / tuple → recurse.
    - Anything else (None, str, int, float) → pass through.
    """
    if torch.is_tensor(v):
        return v.detach().cpu()
    if hasattr(v, "cu_seqlens_q"):  # PackedSeqParams
        return {
            "cu_seqlens_q": _serialize_value(v.cu_seqlens_q),
            "cu_seqlens_kv": _serialize_value(v.cu_seqlens_kv),
            "max_seqlen_q": v.max_seqlen_q,
            "max_seqlen_kv": v.max_seqlen_kv,
            "qkv_format": v.qkv_format,
        }
    if isinstance(v, dict):
        return {k: _serialize_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_serialize_value(x) for x in v)
    return v


def _serialize_packed_batch(batch: dict) -> dict:
    """Convert a get_batch() output dict to a torch.save-friendly form.

    Recurses into nested dicts/lists/tuples so multimodal_train_inputs
    (dict of feature tensors) doesn't pass through with live GPU references.
    """
    return {k: _serialize_value(v) for k, v in batch.items()}


def stash_packed_batch(args, batch: dict, *, step_id: int, microbatch_idx: int) -> None:
    """Append a labeled packed batch to the per-rollout accumulator, if dumping
    is enabled. Called from inside train_one_step's forward_step closure
    once per micro-batch on PP rank 0.

    Each stashed entry carries (step_id, microbatch_idx) so the saved file is
    self-describing: a flat list across multiple train_one_step calls can be
    grouped without external bookkeeping.
    """
    if getattr(args, "save_debug_packed_data", None) is None:
        return
    if not hasattr(args, "_packed_batches_dump") or args._packed_batches_dump is None:
        args._packed_batches_dump = []
    args._packed_batches_dump.append(
        {
            "step_id": step_id,
            "microbatch_idx": microbatch_idx,
            "batch": _serialize_packed_batch(batch),
        }
    )


def save_debug_packed_data(args, *, rollout_id) -> None:
    """Save accumulated packed batches for this rollout.

    Mirrors save_debug_train_data: once per rollout per rank, single file.
    Path template placeholders: {rollout_id}, {rank}, {policy_name}.
    Contents: list of per-microbatch dicts (tokens stream, packed_seq_params,
    advantages, log_probs, loss_masks, ...) — exactly what Megatron's forward
    consumes. Resets the accumulator after writing.
    """
    if (path_template := getattr(args, "save_debug_packed_data", None)) is None:
        return
    batches = getattr(args, "_packed_batches_dump", None) or []
    rank = torch.distributed.get_rank()
    policy_name = getattr(args, "policy_name", None) or "default"
    path = Path(path_template.format(rollout_id=rollout_id, rank=rank, policy_name=policy_name))
    logger.info(f"Save debug packed data to {path} ({len(batches)} micro-batches)")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            rollout_id=rollout_id,
            rank=rank,
            policy_name=policy_name,
            packed_batches=batches,
        ),
        path,
    )
    args._packed_batches_dump = []
