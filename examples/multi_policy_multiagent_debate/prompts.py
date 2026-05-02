## Prompt templates for the multi-agent debate example.
##
## Three templates, derived from the paper's `original_gen.py`:
##   - GENERATOR_INITIAL_TEMPLATE: round-0 independent proposal.
##   - generate_summarize_template(N): summarize N other agents' responses.
##   - GENERATOR_UPDATE_TEMPLATE: produce updated answer using a summary.
##
## Brace escape: every literal `{ANSWER}` placeholder uses `{{{{ANSWER}}}}` in
## the f-string so it survives one f-string evaluation and one `.format()`
## call without becoming a substitution slot. Lessons from the
## solver_summarizer KeyError fix.


# Round 0 — initial independent proposal. The problem_statement comes in
# already chat-templated by slime, so we pass it through as-is.
GENERATOR_INITIAL_TEMPLATE = """{problem_statement}"""


def generate_summarize_template(num_other_agents: int) -> str:
    """Build a summarize-subroutine prompt that summarizes `num_other_agents`
    other-agent responses on the same math problem. The output is consumed
    by the critic (A^C) in the next debate round. The summarize step is a
    SUBROUTINE in the paper's Algorithm 1 (A^S role) — its samples are not
    added to results_dict and not trained on."""
    sections = []
    for i in range(num_other_agents):
        sections.append(f"#### Agent {i+1} response\n{{solution{i+1}}}\n\n---")
    sections_text = "\n".join(sections)

    return f"""Here are responses from {num_other_agents} other agents on the same math problem:

{sections_text}

Write a concise summary of these solutions that calls out:
- Where the agents AGREE on the final answer or method.
- Where they DISAGREE (different answers, different approaches).
- Which approach is most defensible, and why.

Do NOT solve the problem yourself. Only summarize the other agents' reasoning.
"""


# Round k>=1 — agent updates its previous answer using:
#   - its own prior-round response (so it can iterate on its own reasoning,
#     matching the paper's chat-history-based debate structure where each
#     agent sees its own previous turns)
#   - a summary of OTHER agents' prior-round responses
#
# Regular (non-f) string passed through ONE .format() call → `{ANSWER}` is
# escaped as `{{ANSWER}}` (two braces). The f-string template in
# `generate_summarize_template` uses four braces — different escape rules.
GENERATOR_UPDATE_TEMPLATE = """Original problem:

{problem_statement}

---

Your previous response was:

{prior_response}

---

Here is a summary of solutions from other agents on this problem:

{summary}

---

Use the other agents' reasoning above as additional advice. Re-examine your
previous response and provide an updated bullet-point answer, ending with
EXACTLY this format on its own line:

Answer: \\boxed{{ANSWER}}

where ANSWER is the final numeric or symbolic answer.
"""
