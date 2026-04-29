#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-0.8B.sh"

# Per-policy fields (parallel, recompute, batching, optimizer, loss, paths)
# all live in config.yaml — one block per actor.
# Globals (rollout cadence, wandb, sglang infra, attention defaults) stay as CLI args.

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.multi_policy_multi_agent.rollout_with_multi_agents.generate_with_multi_agents
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --global-batch-size 256
   --disable-rollout-trim-samples
   --rollout-max-context-len 16384
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --balance-data
)

# Cluster sizing — derived from config.yaml:
#   actor_gpus   = sum(policies[i].megatron_num_nodes * num_gpus_per_node)
#                = 3 × 1 × 2 = 6
#   rollout_gpus = sum(policies[i].sglang_num_nodes   * num_gpus_per_node)
#                = 3 × 1 × 2 = 6
#   total (colocate) = max(actor_gpus, rollout_gpus) = 6
#   total (separate) = actor_gpus + rollout_gpus     = 12
NUM_GPUS=6

TRAIN_ARGS=(
   --config "${SCRIPT_DIR}/config.yaml"
   --save-interval 20
)
# Note: train_multi_policy.py derives args.rollout_num_gpus from config.yaml
# (sum of sglang_num_nodes × num_gpus_per_node across policies), so no
# --rollout-num-gpus flag is needed.
# config.yaml is the single source of truth: one entry per policy holding
# Megatron training, orchestration (buffer_mode, GPU placement), and the
# 1:1-paired sglang engine sub-block.

# Eval routes through one trainable policy (first in megatron_config: solver).
EVAL_ARGS=(
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

# PERF and OPTIMIZER are per-policy in config.yaml.
# Each policy declares its own: tensor_model_parallel_size, sequence_parallel,
# recompute_*, use_dynamic_batch_size, max_tokens_per_gpu, optimizer, lr,
# lr_decay_style, weight_decay, adam_beta*, optimizer_cpu_offload, etc.

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3.5-0.8B-multi-policy-multi-agent
)

# sglang server args are per-policy in config.yaml (sglang sub-block).
# Each policy declares its own: model_path, num_gpus_per_engine, mem_fraction_static,
# cuda_graph_bs, chunked_prefill_size, max_running_requests, attention_backend, server_groups, etc.

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_multi_policy.py \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${TRAIN_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}
