"""Two-agent multi-policy example: solver + selector.

Solver generates N candidate solutions per problem; selector picks one of them.
Both policies are trained on their own buffers (split-buffer mode).

Per-prompt request count:
  num_parallel solver calls + num_parallel selector calls.

Reward shaping:
- Each solver sample's reward comes from the rule-based RM (correct/incorrect
  answer match against the gold label).
- Each selector sample's reward = the reward of the solution it selected.
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

from .prompts import SOLVER_PROMPT_TEMPLATE, generate_select_template

# Unique-index source for inner samples. run_agent_system spawns num_parallel
# deep-copies of an outer sample, all of which would share the outer's index;
# get_data_iterator's uniqueness assertion then trips on the duplicates. Using
# a high offset keeps the inner indices well clear of slime's data_source counter.
_INNER_SAMPLE_ID = itertools.count(start=1_000_000_000)


# Match the chat-control tokens slime's apply_chat_template emits for Qwen-style
# models. We strip these from the solver's chat-formatted prompt before
# embedding it as plain "problem text" inside the selector's template.
_CHAT_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>(?:user|assistant|system)?\s*")


def _strip_chat_tokens(text: str) -> str:
    """Strip Qwen/sglang chat-control tokens (<|im_start|>user, etc.) so the
    inner problem text can be embedded inside another prompt without nesting
    chat structures (which confuses the selector model)."""
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
        else:
            new_response_tokens = []

        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response = output["text"]

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


class SelectorAgent(Agent):
    async def select(self, args, problem_statement, candidate_solutions: list[str]) -> str:
        template = generate_select_template(len(candidate_solutions))
        # problem_statement comes in raw (no chat tokens) — see run_agent_system.
        # The candidate solutions are model outputs which may contain stray
        # <|im_end|> tokens; strip those too so the selector's prompt has clean
        # text inside its candidate sections.
        format_params = {"problem_statement": problem_statement}
        for i, solution in enumerate(candidate_solutions):
            format_params[f"solution{i+1}"] = _strip_chat_tokens(solution)
        body = template.format(**format_params)
        # Wrap the whole selector prompt as a proper user turn so the model
        # receives an unambiguous chat-formatted message instead of the prior
        # bare-text-with-nested-tokens mess.
        prompt = _wrap_user_turn(args.tokenizer, body)
        return await self.run(args, prompt, max_retries=10, key="selector")

    def extract_selected_solution_idx(self, response: str, candidate_solutions: list[str]) -> int | None:
        # The prompt asks for "Judgment: IDX" but the model frequently echoes
        # the literal "IDX" token (e.g. "Judgment: IDX 1" or "Judgment: IDX1").
        # Allow optional "IDX" or "Solution" filler between the colon and digit.
        PATTERN = re.compile(r"Judgment:\s*(?:IDX|Solution)?\s*#?(\d+)", re.IGNORECASE)
        matched = PATTERN.findall(response)
        if not matched:
            return None
        try:
            selected_id = int(matched[0]) - 1
            if 0 <= selected_id < len(candidate_solutions):
                return selected_id
            return None
        except Exception as e:
            print(f"extract_selected_solution_idx error: {e}")
            return None


async def solver_worker(args, problem_statement, worker_id):
    try:
        solver = SolverAgent()
        return await solver.generate_initial_solution(args, problem_statement)
    except Exception as e:
        print(f"[Solver-{worker_id}] exception: {e}\n{traceback.format_exc()}")
        return None


async def select_worker(args, problem_statement, candidate_solutions, worker_id):
    """Run a single selector. Multiple selectors run in parallel so the selector
    policy gets `num_parallel` trajectories per problem (needed for GRPO group-norm)."""
    try:
        selector = SelectorAgent()
        return await selector.select(args, problem_statement, candidate_solutions)
    except Exception as e:
        print(f"[Selector-{worker_id}] exception: {e}\n{traceback.format_exc()}")
        return None


async def run_agent_system(args, sample):
    """Run num_parallel solver pipelines + num_parallel selector pipelines.

    Returns a flat list of samples tagged with policy_name in {"solver", "selector"}
    so the rollout manager's split-buffer routing fans them out correctly.
    """
    args = deepcopy(args)
    args.sample = sample
    args.results_dict = {"solver": [], "selector": []}

    # The solver's prompt was already chat-templated by slime upstream and
    # contains <|im_start|>user / <|im_end|> tokens. The solver passes that
    # straight through to sglang (which expects chat format), but we cannot
    # embed the chat-formatted string inside the selector's "### Problem"
    # field — that produces nested chat structures and breaks the selector
    # model. Keep the chat-formatted version for the solver and a clean
    # raw version for use inside the selector template.
    solver_prompt = sample.prompt
    raw_problem = _strip_chat_tokens(sample.prompt)

    # Phase 1 — solvers (in parallel)
    tasks = [solver_worker(args, solver_prompt, wid) for wid in range(args.num_parallel)]
    await asyncio.gather(*tasks, return_exceptions=True)

    rewards = await batched_async_rm(args, args.results_dict["solver"])
    for s, r in zip(args.results_dict["solver"], rewards, strict=False):
        s.reward = r

    candidate_solutions = [s.response_content for s in args.results_dict["solver"] if s.response_content is not None]

    def reward_adjustment(samples, weight):
        for s in samples:
            s.reward = s.reward * weight
        return samples

    if len(candidate_solutions) == 0:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        return args.results_dict["solver"]

    # Phase 2 — selectors (in parallel; each picks one of the solver candidates)
    tasks = [select_worker(args, raw_problem, candidate_solutions, wid) for wid in range(args.num_parallel)]
    await asyncio.gather(*tasks, return_exceptions=True)

    if len(args.results_dict["selector"]) == 0:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        return args.results_dict["solver"]

    # Reward propagation: each selector sample's reward = reward of the solution
    # it picked. Match by response_content text since task order may not align.
    # Track parsing success separately — selectors that fail to emit a parseable
    # "Judgment: N" must NOT contribute to mean_selector_reward, otherwise a
    # parse failure (selector-side bug) penalizes correct solvers (anti-train).
    selector = SelectorAgent()
    parsed_selector_rewards: list[float] = []
    for sel_sample in args.results_dict["selector"]:
        if sel_sample.response_content is None:
            sel_sample.reward = 0
            continue
        idx = selector.extract_selected_solution_idx(sel_sample.response_content, candidate_solutions)
        if idx is None:
            sel_sample.reward = 0
            continue
        picked_solution = candidate_solutions[idx]
        sel_sample.reward = 0
        for solver_s in args.results_dict["solver"]:
            if solver_s.response_content == picked_solution:
                sel_sample.reward = solver_s.reward
                break
        parsed_selector_rewards.append(sel_sample.reward)

    # If every selector failed to parse, we have no judgment signal. Keep the
    # raw solver rewards untouched — anything else (e.g. multiplying by
    # incorrect_reward_weight) anti-trains correct solvers when the selector
    # is broken in some prompt-specific way (#2 from the audit).
    if not parsed_selector_rewards:
        return args.results_dict["solver"] + args.results_dict["selector"]

    # Group-level reward shaping: bonus all roles when the average parsed
    # selector reward suggests the agents found a correct answer.
    mean_selector_reward = sum(parsed_selector_rewards) / len(parsed_selector_rewards)
    weight = args.correct_reward_weight if mean_selector_reward > 0.5 else args.incorrect_reward_weight
    reward_adjustment(args.results_dict["solver"], weight)
    reward_adjustment(args.results_dict["selector"], weight)

    return args.results_dict["solver"] + args.results_dict["selector"]
