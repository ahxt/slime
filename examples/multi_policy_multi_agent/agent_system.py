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

from .prompts import SOLVER_PROMPT_TEMPLATE, generate_rewriter_template, generate_select_template

# Unique-index source for inner samples spawned inside run_agent_system; without
# this, all num_parallel siblings inherit the outer prompt's index and trip
# get_data_iterator's uniqueness assertion.
_INNER_SAMPLE_ID = itertools.count(start=1_000_000_000)


# Strip Qwen/sglang chat-control tokens so chat-formatted text can be embedded
# inside another prompt body without nesting chat structures.
_CHAT_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>(?:user|assistant|system)?\s*")


def _strip_chat_tokens(text: str) -> str:
    return _CHAT_TOKEN_RE.sub("", text).strip()


def _wrap_user_turn(tokenizer, user_content: str) -> str:
    """Render `user_content` as a single-turn user message via chat template."""
    if getattr(tokenizer, "chat_template", None) is None:
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


async def generate_response(args, prompt, key):
    try:
        sampling_params = args.sampling_params
        tokenizer = args.tokenizer
        max_context_length = args.rollout_max_context_len
        sample = deepcopy(args.sample)

        # Multi-policy: route to the sglang engine paired with this role.
        # `key` ∈ {"solver", "rewriter", "selector"} matches a name in --sglang-config.
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

        # Extract new response tokens
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            # abort
            new_response_tokens = []
            new_response_log_probs = []

        # Update sample with tokens directly - avoiding re-tokenization
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
            # case "abort":
            #     sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED

        # Multi-policy: tag the sample so the manager routes it to the right
        # policy's buffer. Also assign a unique index — the deepcopy from
        # args.sample inherits the outer prompt's index, which would collide
        # across the num_parallel siblings from this call.
        sample.policy_name = key
        sample.index = next(_INNER_SAMPLE_ID)
        args.results_dict[key].append(sample)

        final = output["text"].replace("<|user|>", "")
        if "</think>" in final:
            contents = final.split("</think>")
            if len(contents) == 2 and contents[1] != "":
                reason_content = contents[0].strip()
                response_content = contents[1].strip()
                sample.reason_content = reason_content
                sample.response_content = response_content
                return response_content
        sample.reason_content = None
        sample.response_content = None
        return None
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


class Agent:
    """A base class for our AI agents."""

    def __init__(self):
        pass

    async def run(self, args, prompt, max_retries: int = 1, key: str = None) -> str:
        """Runs the agent by sending a prompt to the LLM."""
        for _i in range(max_retries):
            try:
                response = await generate_response(args, prompt, key=key)
                return response
            except Exception as e:
                print(f"Error querying LLM: {e}")
                time.sleep(1)
        print(f"Failed to query LLM after {max_retries} retries")
        return None


class SolverAgent(Agent):
    """The agent responsible for generating and improving solutions."""

    def __init__(self):
        super().__init__()

    async def generate_initial_solution(self, args, problem_statement) -> str:
        """Generates the first solution attempt."""
        prompt = SOLVER_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
        return await self.run(args, prompt, max_retries=3, key="solver")


class RewriterAgent(Agent):
    """The agent responsible for rewriting solutions."""

    def __init__(self):
        super().__init__()

    async def rewrite(self, args, problem_statement, previous_solutions: list[str]) -> str:
        """Generates the rewrited solution."""

        # Build the prompt template dynamically.
        template = generate_rewriter_template(len(previous_solutions))

        # Populate the template arguments.
        # problem_statement comes in raw (no chat tokens). Solutions are model
        # outputs which may carry stray <|im_end|> — strip those too.
        format_params = {"problem_statement": problem_statement}
        for i, solution in enumerate(previous_solutions):
            format_params[f"solution{i+1}"] = _strip_chat_tokens(solution)

        body = template.format(**format_params)
        # Wrap as a proper user turn so the model sees an unambiguous chat msg.
        prompt = _wrap_user_turn(args.tokenizer, body)
        return await self.run(args, prompt, max_retries=1, key="rewriter")


class SelectorAgent(Agent):
    """The agent responsible for selecting solutions."""

    def __init__(self):
        super().__init__()

    async def select(self, args, problem_statement, candidate_solutions: list[str]) -> str:
        """Generates the rewrited solution."""

        # Build the prompt template dynamically.
        template = generate_select_template(len(candidate_solutions))

        # Populate the template arguments.
        # problem_statement is raw (no chat tokens); solutions are stripped too.
        format_params = {"problem_statement": problem_statement}
        for i, solution in enumerate(candidate_solutions):
            format_params[f"solution{i+1}"] = _strip_chat_tokens(solution)

        body = template.format(**format_params)
        prompt = _wrap_user_turn(args.tokenizer, body)
        return await self.run(args, prompt, max_retries=10, key="selector")

    def extract_selected_solution_idx(self, response: str, candidate_solutions: list[str]) -> int:
        """Extracts the selected solution ID from the response.
        Accepts variants the model emits: "Judgment: 1", "Judgment: IDX 1",
        "Judgment: Solution 1", "Judgment: #1"."""
        PATTERN = re.compile(r"Judgment:\s*(?:IDX|Solution)?\s*#?(\d+)", re.IGNORECASE)
        matched = PATTERN.findall(response)
        if not matched:
            return None
        try:
            selected_id = int(matched[0]) - 1
            if selected_id < len(candidate_solutions) and selected_id >= 0:
                return selected_id
            else:
                return None
        except Exception as e:
            print(f"extract_selected_solution_idx error: {e}")
            return None


async def rewrite_worker(args, previous_solutions, problem_statement, worker_id):
    rewriter = RewriterAgent()
    new_solution = await rewriter.rewrite(args, problem_statement, previous_solutions)
    return new_solution


async def solver_worker(args, problem_statement, worker_id):
    """
    Run a single solver pipeline.
    """

    try:
        solver = SolverAgent()
        current_solution = await solver.generate_initial_solution(args, problem_statement)
        return current_solution

    except Exception as e:
        print(f"[Worker-{worker_id}] exception: {e}")
        print(f"[Worker-{worker_id}] traceback: {traceback.format_exc()}")
        return None


async def select_worker(args, problem_statement, candidate_solutions, worker_id):
    """Run a single selector. Multiple selectors run in parallel so the selector
    policy gets `num_parallel` trajectories per problem (needed for GRPO group-norm)."""
    try:
        selector = SelectorAgent()
        response = await selector.select(args, problem_statement, candidate_solutions)
        return response
    except Exception as e:
        print(f"[Selector-{worker_id}] exception: {e}")
        print(f"[Selector-{worker_id}] traceback: {traceback.format_exc()}")
        return None


def _pad_role_buffer(args, role: str, target_count: int, donor_role: str | None = None):
    """Top up `args.results_dict[role]` to exactly `target_count` samples.

    Multi-policy split-buffer routing requires each policy's buffer per
    rollout to equal global_batch_size. When a phase early-exits or some
    workers fail, the role's buffer ends up short and the next training
    step trips:
        assert len(set(sum(micro_batch_indices, []))) == num_local_samples
    in slime/backends/megatron_utils/data.py.

    Pad with zero-reward placeholders cloned from an existing role sample
    (or from `donor_role` when this role has none yet — typically donate
    a solver sample so selector pad has valid tokens). Each placeholder
    gets a fresh unique index from _INNER_SAMPLE_ID.
    """
    samples = args.results_dict[role]
    if len(samples) >= target_count:
        del samples[target_count:]
        return
    donor_pool = samples if samples else (args.results_dict.get(donor_role) or [])
    if not donor_pool:
        return
    while len(samples) < target_count:
        placeholder = deepcopy(donor_pool[0])
        placeholder.policy_name = role
        placeholder.index = next(_INNER_SAMPLE_ID)
        placeholder.reward = 0.0
        placeholder.response_content = None
        placeholder.reason_content = None
        samples.append(placeholder)


async def run_agent_system(args, sample):
    """
    Run `num_parallel` pipelines concurrently.
    """

    args = deepcopy(args)  # Deep copy args because rollout_with_multi_agents mutates them.
    args.sample = sample
    args.results_dict = {"solver": [], "rewriter": [], "selector": []}

    # Solver gets the chat-formatted prompt straight through (sglang expects
    # chat format). Rewriter / selector embed the problem inside their own
    # template — they need the RAW text, not the chat-tagged version, or we
    # end up with nested <|im_start|>user...<|im_end|> structures that
    # confuse the model.
    solver_prompt = sample.prompt
    raw_problem = _strip_chat_tokens(sample.prompt)
    tasks = [solver_worker(args, solver_prompt, worker_id) for worker_id in range(args.num_parallel)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    rewards = await batched_async_rm(args, args.results_dict["solver"])
    for sample, reward in zip(args.results_dict["solver"], rewards, strict=False):
        sample.reward = reward

    previous_solutions = [item for item in results if isinstance(item, str)]

    def reward_adjustment(samples, reward_weight):
        for sample in samples:
            sample.reward = sample.reward * reward_weight
        return samples

    n = args.num_parallel

    if len(previous_solutions) == 0:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        # Pad all three roles so each policy's per-rollout buffer count is the
        # same regardless of which phase early-exits. Slime's get_data_iterator
        # asserts num_local_samples == num_local_gbs and trips otherwise.
        _pad_role_buffer(args, "solver", n)
        _pad_role_buffer(args, "rewriter", n, donor_role="solver")
        _pad_role_buffer(args, "selector", n, donor_role="solver")
        return args.results_dict["solver"] + args.results_dict["rewriter"] + args.results_dict["selector"]

    # Rewriting — feed the raw (un-chat-templated) problem to the rewriter
    # template so the inner solver chat tokens don't leak through.
    tasks = [
        rewrite_worker(args, previous_solutions, raw_problem, worker_id)
        for worker_id in range(args.num_parallel)
    ]
    rewrited_solutions_raw = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out failed tasks and keep only valid rewritten solutions.
    rewrited_solutions = []
    for _i, result in enumerate(rewrited_solutions_raw):
        if isinstance(result, str):
            rewrited_solutions.append(result)

    rewards = await batched_async_rm(args, args.results_dict["rewriter"])
    for sample, reward in zip(args.results_dict["rewriter"], rewards, strict=False):
        sample.reward = reward

    if len(rewrited_solutions) == 0:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        reward_adjustment(args.results_dict["rewriter"], args.incorrect_reward_weight)
        _pad_role_buffer(args, "solver", n)
        _pad_role_buffer(args, "rewriter", n, donor_role="solver")
        _pad_role_buffer(args, "selector", n, donor_role="solver")
        return args.results_dict["solver"] + args.results_dict["rewriter"] + args.results_dict["selector"]

    # Selection — run num_parallel selectors so the selector policy gets enough
    # trajectories per problem for GRPO group-norm (matches n_samples_per_prompt).
    # Selector also receives the raw problem (no chat tokens) — its template
    # will chat-template the whole thing as a user turn.
    selector = SelectorAgent()  # used for extract_selected_solution_idx
    tasks = [
        select_worker(args, raw_problem, rewrited_solutions, worker_id)
        for worker_id in range(args.num_parallel)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    if len(args.results_dict["selector"]) == 0:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        reward_adjustment(args.results_dict["rewriter"], args.incorrect_reward_weight)
        _pad_role_buffer(args, "solver", n)
        _pad_role_buffer(args, "rewriter", n, donor_role="solver")
        _pad_role_buffer(args, "selector", n, donor_role="solver")
        return args.results_dict["solver"] + args.results_dict["rewriter"] + args.results_dict["selector"]

    # Assign each selector trajectory its own reward based on which rewriter solution
    # it picked. Use sample.response_content (set by generate_response) so the order
    # of args.results_dict["selector"] doesn't have to match task order.
    # Track parsing success — selectors that fail to emit a parseable judgment
    # must NOT contribute to mean_selector_reward, otherwise a parse failure
    # (e.g. "Judgment: IDX.AUTHOR") penalizes correct solvers/rewriters
    # (anti-train).
    parsed_selector_rewards: list[float] = []
    for sel_sample in args.results_dict["selector"]:
        response = sel_sample.response_content
        if response is None:
            sel_sample.reward = 0
            continue
        selected_solution_idx = selector.extract_selected_solution_idx(response, rewrited_solutions)
        if selected_solution_idx is None:
            sel_sample.reward = 0
            continue
        selected_solution = rewrited_solutions[selected_solution_idx]
        sel_sample.reward = 0  # default if no match
        for r_sample in args.results_dict["rewriter"]:
            if r_sample.response_content is not None and selected_solution in r_sample.response_content:
                sel_sample.reward = r_sample.reward
                break
        parsed_selector_rewards.append(sel_sample.reward)

    # Global reward shaping: if the average parsed selector reward suggests
    # success, bonus all roles; otherwise penalize. If every selector failed
    # to parse, we have no judgment signal — leave the raw rewards untouched
    # so we don't anti-train correct solvers/rewriters when the selector
    # is broken in some prompt-specific way.
    if not parsed_selector_rewards:
        _pad_role_buffer(args, "solver", n)
        _pad_role_buffer(args, "rewriter", n, donor_role="solver")
        _pad_role_buffer(args, "selector", n, donor_role="solver")
        return args.results_dict["solver"] + args.results_dict["rewriter"] + args.results_dict["selector"]

    mean_selector_reward = sum(parsed_selector_rewards) / len(parsed_selector_rewards)
    weight = args.correct_reward_weight if mean_selector_reward > 0.5 else args.incorrect_reward_weight
    reward_adjustment(args.results_dict["solver"], weight)
    reward_adjustment(args.results_dict["rewriter"], weight)
    reward_adjustment(args.results_dict["selector"], weight)

    # Final guard: ensure each role's buffer is exactly num_parallel.
    _pad_role_buffer(args, "solver", n)
    _pad_role_buffer(args, "rewriter", n, donor_role="solver")
    _pad_role_buffer(args, "selector", n, donor_role="solver")

    return args.results_dict["solver"] + args.results_dict["rewriter"] + args.results_dict["selector"]
