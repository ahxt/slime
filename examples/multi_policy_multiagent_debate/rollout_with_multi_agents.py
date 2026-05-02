import random

from transformers import AutoTokenizer

from slime.utils.misc import load_function
from slime.utils.types import Sample

MULTI_AGENT_CONFIGS = {
    "custom_multi_agent_function_path": "examples.multi_policy_multiagent_debate.agent_system.run_agent_system",
    # num_parallel = paper's `agents`. Counterfactual reward needs ≥ 3 to
    # avoid the degenerate summarize-1-response case (see plan §1).
    "num_parallel": 3,
    # rounds = paper's `rounds`. Round 0 = propose; rounds 1..R-1 = summarize
    # + update. With 3 rounds we get one full iterated debate cycle.
    "rounds": 3,
}


async def generate_with_multi_agents(args, sample: Sample, sampling_params, evaluation=False) -> list[Sample]:
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    max_context_length = args.rollout_max_context_len if not evaluation else args.eval_max_context_len

    args.sampling_params = sampling_params
    args.rollout_max_context_len = max_context_length
    args.tokenizer = tokenizer

    for key, value in MULTI_AGENT_CONFIGS.items():
        setattr(args, key, value)

    custom_multi_agent_func = load_function(args.custom_multi_agent_function_path)
    samples = await custom_multi_agent_func(args, sample)

    random.shuffle(samples)
    return samples
