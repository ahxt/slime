## Prompt templates for the solver + summarizer example.


SOLVER_PROMPT_TEMPLATE = """{problem_statement}"""


def generate_summarize_template(num_solutions: int) -> str:
    """Build a summarizer prompt that synthesizes `num_solutions` candidate
    solutions into one final answer. The summarizer's response is graded
    directly by the verifiable reward (RLVR — deepscaler boxed-answer
    matcher), so it MUST end with the standard `Answer: \\boxed{...}`
    format that deepscaler/dapo-math expect."""
    solution_sections = []
    for i in range(num_solutions):
        solution_sections.append(f"#### Solution {i+1}\n{{solution{i+1}}}\n\n---")

    solutions_text = "\n".join(solution_sections)

    return f"""You will be given a challenging math problem followed by {num_solutions} candidate solutions.
Your task is to synthesize them into one clean, correct final solution.

You are provided with two documents:
1.  The problem you need to solve.
2.  {num_solutions} "Candidate Solutions" produced by independent solvers.

Synthesis Process:
1. Initial Screening
- Group candidates by their final answers.
- Identify mathematical contradictions and eliminate clearly wrong reasoning.

2. Synthesis
- Choose the most defensible approach (or combine the best parts of several).
- Re-derive the final answer cleanly, fixing any small errors you spot.
- Keep the explanation short but complete.

End your response with EXACTLY this format on its own line:
Answer: \\boxed{{ANSWER}}
where ANSWER is the final numeric or symbolic answer.

### Problem

{{problem_statement}}

---

### Candidate Solutions
{solutions_text}
"""
