"""Two-agent multi-policy example: solver + summarizer.

Pipeline (N = num_parallel = n_samples_per_prompt per role):
  1. Solver:     N parallel solvers each produce a candidate solution.
  2. Summarizer: N parallel summarizers each see ALL N solver candidates and
                 synthesize one final answer. Each summarizer's response is
                 graded directly by the verifiable reward (RLVR — its own
                 boxed answer is what's scored), so we don't need index-
                 extraction logic like the selector example does.

Both policies train on their own buffers (split-buffer mode).
"""

import asyncio
import itertools
import re
import time
import traceback
from copy import deepcopy

from slime.rollout.rm_hub import batched_async_rm
from slime.rollout.sglang_rollout import get_model_url
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .prompts import SOLVER_PROMPT_TEMPLATE, generate_summarize_template

# Unique-index source for inner samples. run_agent_system spawns num_parallel
# deep-copies of an outer sample, all of which would share the outer's index;
# get_data_iterator's uniqueness assertion then trips on the duplicates. Using
# a high offset keeps the inner indices well clear of slime's data_source counter.
_INNER_SAMPLE_ID = itertools.count(start=1_000_000_000)


# Match the chat-control tokens slime's apply_chat_template emits for Qwen-style
# models. We strip these from the solver's chat-formatted prompt before
# embedding it as plain "problem text" inside the summarizer's template.
_CHAT_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>(?:user|assistant|system)?\s*")


def _strip_chat_tokens(text: str) -> str:
    """Strip Qwen/sglang chat-control tokens (<|im_start|>user, etc.) so the
    inner problem text can be embedded inside another prompt without nesting
    chat structures (which confuses downstream models)."""
    return _CHAT_TOKEN_RE.sub("", text).strip()


def _wrap_user_turn(tokenizer, user_content: str) -> str:
    """Render `user_content` as a single-turn user message via the tokenizer's
    chat template. Falls back to a manual wrap if the tokenizer has no
    chat_template (e.g. unit-test stubs)."""
    if getattr(tokenizer, "chat_template", None) is None:
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


async def generate_response(args, prompt, key):
    """Call the policy's paired sglang engine with `prompt`. Tags the resulting
    Sample with policy_name=key so the manager routes it to the right buffer."""
    try:
        sampling_params = args.sampling_params
        tokenizer = args.tokenizer
        max_context_length = args.rollout_max_context_len
        sample = deepcopy(args.sample)

        # Multi-policy: route to the sglang engine paired with this role.
        url = get_model_url(args, key)

        sample.prompt = prompt
        prompt_token_ids = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        sample.tokens = prompt_token_ids
        prompt_length = len(prompt_token_ids)
        current_sampling_params = deepcopy(sampling_params)
        current_sampling_params["max_new_tokens"] = min(
            sampling_params["max_new_tokens"], max_context_length - prompt_length
        )

        if current_sampling_params["max_new_tokens"] <= 0:
            return None

        payload = {"input_ids": prompt_token_ids, "sampling_params": current_sampling_params, "return_logprob": True}
        output = await post(url, payload)

        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_response_tokens = []
            new_response_log_probs = []

        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response = output["text"]
        # Save sglang's per-token logprobs so train-side can compute
        # train_rollout_logprob_abs_diff (and tis_* if --use-tis is on).
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "stop":
                sample.status = Sample.Status.COMPLETED

        # Multi-policy buffer routing tag + unique index per inner sample (the
        # deepcopy from args.sample inherits the outer prompt's index, which
        # would collide across the num_parallel siblings from this call).
        sample.policy_name = key
        sample.index = next(_INNER_SAMPLE_ID)
        args.results_dict[key].append(sample)

        final = output["text"].replace("<|user|>", "")
        if "</think>" in final:
            contents = final.split("</think>")
            if len(contents) == 2 and contents[1] != "":
                sample.reason_content = contents[0].strip()
                sample.response_content = contents[1].strip()
                return sample.response_content
        sample.reason_content = None
        sample.response_content = None
        return None
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


class Agent:
    async def run(self, args, prompt, max_retries: int = 1, key: str = None) -> str:
        for _ in range(max_retries):
            try:
                response = await generate_response(args, prompt, key=key)
                return response
            except Exception as e:
                print(f"Error querying LLM: {e}")
                time.sleep(1)
        return None


class SolverAgent(Agent):
    async def generate_initial_solution(self, args, problem_statement) -> str:
        prompt = SOLVER_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
        return await self.run(args, prompt, max_retries=3, key="solver")


class SummarizerAgent(Agent):
    async def summarize(self, args, problem_statement, candidate_solutions: list[str]) -> str:
        template = generate_summarize_template(len(candidate_solutions))
        # problem_statement comes in raw (no chat tokens) — see run_agent_system.
        # Candidate solutions are model outputs which may carry stray
        # <|im_end|> tokens; strip those too.
        format_params = {"problem_statement": problem_statement}
        for i, solution in enumerate(candidate_solutions):
            format_params[f"solution{i+1}"] = _strip_chat_tokens(solution)
        body = template.format(**format_params)
        # Wrap as a proper user turn so the model receives an unambiguous
        # chat-formatted message instead of bare text with nested tokens.
        prompt = _wrap_user_turn(args.tokenizer, body)
        return await self.run(args, prompt, max_retries=3, key="summarizer")


async def solver_worker(args, problem_statement, worker_id):
    try:
        solver = SolverAgent()
        return await solver.generate_initial_solution(args, problem_statement)
    except Exception as e:
        print(f"[Solver-{worker_id}] exception: {e}\n{traceback.format_exc()}")
        return None


async def summarize_worker(args, problem_statement, candidate_solutions, worker_id):
    """Run a single summarizer. Multiple summarizers run in parallel so the
    summarizer policy gets `num_parallel` trajectories per problem (needed for
    GRPO group-norm)."""
    try:
        summarizer = SummarizerAgent()
        return await summarizer.summarize(args, problem_statement, candidate_solutions)
    except Exception as e:
        print(f"[Summarizer-{worker_id}] exception: {e}\n{traceback.format_exc()}")
        return None


def _pad_role_buffer(args, role: str, target_count: int, donor_role: str | None = None):
    """Top up `args.results_dict[role]` to exactly `target_count` samples.

    The multi-policy training path requires each policy's buffer per rollout
    to equal global_batch_size (slime's get_data_iterator asserts on this:
    num_steps_per_rollout = num_local_samples // num_local_gbs). When a phase
    early-exits or some workers fail, the role's buffer is short and the
    next training step trips that assertion.

    Pad with zero-reward placeholder samples cloned from an existing role
    sample (or from `donor_role` when this role has none yet — typically
    we donate a solver sample so the summarizer pad has valid tokens).
    Each placeholder gets a fresh unique index so the inside-rollout
    uniqueness invariants hold.
    """
    samples = args.results_dict[role]
    if len(samples) >= target_count:
        del samples[target_count:]   # also trim if somehow longer than expected
        return
    donor_pool = samples if samples else (args.results_dict.get(donor_role) or [])
    if not donor_pool:
        return  # nothing to clone from; let the assertion fire downstream
    while len(samples) < target_count:
        placeholder = deepcopy(donor_pool[0])
        placeholder.policy_name = role
        placeholder.index = next(_INNER_SAMPLE_ID)
        placeholder.reward = 0.0
        placeholder.response_content = None
        placeholder.reason_content = None
        samples.append(placeholder)


async def run_agent_system(args, sample):
    """Run num_parallel solver pipelines + num_parallel summarizer pipelines.

    Returns a flat list of samples tagged with policy_name in {"solver",
    "summarizer"} so the rollout manager's split-buffer routing fans them
    out correctly.
    """
    args = deepcopy(args)
    args.sample = sample
    args.results_dict = {"solver": [], "summarizer": []}

    # The solver's prompt was already chat-templated by slime upstream and
    # contains <|im_start|>user / <|im_end|> tokens. The solver passes that
    # straight through to sglang (which expects chat format), but we cannot
    # embed the chat-formatted string inside the summarizer's template —
    # that produces nested chat structures and breaks the summarizer.
    # Keep the chat-formatted version for the solver and a clean raw
    # version for use inside the summarizer template.
    solver_prompt = sample.prompt
    raw_problem = _strip_chat_tokens(sample.prompt)

    n = args.num_parallel

    # Phase 1 — solvers (in parallel)
    tasks = [solver_worker(args, solver_prompt, wid) for wid in range(n)]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Score each solver directly (verifiable reward / RLVR on its own response).
    rewards = await batched_async_rm(args, args.results_dict["solver"])
    for s, r in zip(args.results_dict["solver"], rewards, strict=False):
        s.reward = r

    candidate_solutions = [s.response_content for s in args.results_dict["solver"] if s.response_content is not None]

    def reward_adjustment(samples, weight):
        for s in samples:
            s.reward = s.reward * weight
        return samples

    if len(candidate_solutions) == 0:
        # No usable solver output; pad both roles so the per-policy buffers
        # stay at exactly num_parallel samples (split-buffer routing requires
        # each rollout to contribute the same count per role).
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        _pad_role_buffer(args, "solver", n)
        _pad_role_buffer(args, "summarizer", n, donor_role="solver")
        return args.results_dict["solver"] + args.results_dict["summarizer"]

    # Phase 2 — summarizers (in parallel; each synthesizes from all solver candidates)
    tasks = [summarize_worker(args, raw_problem, candidate_solutions, wid) for wid in range(n)]
    await asyncio.gather(*tasks, return_exceptions=True)

    # No summarizer output at all → anti-train guard: don't penalize correct
    # solvers when the summarizer phase failed entirely.
    if not args.results_dict["summarizer"]:
        _pad_role_buffer(args, "solver", n)
        _pad_role_buffer(args, "summarizer", n, donor_role="solver")
        return args.results_dict["solver"] + args.results_dict["summarizer"]

    # Score each summarizer directly — its synthesized boxed answer is what's
    # graded by the RM (deepscaler reads sample.response, not response_content),
    # so no index lookup is required. We DON'T filter on response_content here:
    # unlike the selector example (where selector reward depends on parsing
    # `Judgment: IDX` out of response_content), the summarizer's RM grade is
    # independent of whether `</think>` appears. Filtering would wrongly drop
    # valid samples that emitted `Answer: \boxed{...}` outside a think block.
    summarizer_rewards = await batched_async_rm(args, args.results_dict["summarizer"])
    for s, r in zip(args.results_dict["summarizer"], summarizer_rewards, strict=False):
        s.reward = r

    # Group reward shaping: if the summarizer phase produced mostly correct
    # final answers, bonus both roles; otherwise penalize. Mean over ALL
    # summarizer samples since each one is a valid RM datapoint.
    mean_summarizer_reward = sum(s.reward for s in args.results_dict["summarizer"]) / len(args.results_dict["summarizer"])
    weight = args.correct_reward_weight if mean_summarizer_reward > 0.5 else args.incorrect_reward_weight
    reward_adjustment(args.results_dict["solver"], weight)
    reward_adjustment(args.results_dict["summarizer"], weight)

    # Final guard: pad each role to exactly num_parallel so the per-rollout
    # per-role buffer count is invariant across run_agent_system calls.
    _pad_role_buffer(args, "solver", n)
    _pad_role_buffer(args, "summarizer", n, donor_role="solver")

    return args.results_dict["solver"] + args.results_dict["summarizer"]
