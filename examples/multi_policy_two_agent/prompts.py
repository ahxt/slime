## Prompt templates for the two-agent (solver + selector) example.


SOLVER_PROMPT_TEMPLATE = """{problem_statement}"""


def generate_select_template(num_solutions: int) -> str:
    """Dynamically generate select templates based on the number of solutions."""
    solution_sections = []
    for i in range(num_solutions):
        solution_sections.append(f"#### Solution {i+1}\n{{solution{i+1}}}\n\n---")

    solutions_text = "\n".join(solution_sections)

    return f"""You will be given a challenging math problem followed by {num_solutions} solutions.
Your task is to systematically analyze these solutions to identify the most mathematically sound approach.

You are provided with two documents:
1.  The problem you need to solve.
2.  Your {num_solutions} "Candidate Solutions".

Evaluation Process:
1. Initial Screening
- Group solutions by their final answers
- Identify and explain mathematical contradictions between different answers
- Eliminate solutions with clear mathematical errors

2. Detailed Analysis
For remaining solutions, evaluate:
- Mathematical precision and accuracy
- Logical progression of steps
- Completeness of mathematical reasoning

3. Solution Comparison
Compare viable solutions based on:
- Efficiency of approach
- Clarity of mathematical reasoning
- Robustness of solution

End your evaluation with exactly:
Judgment: IDX
where IDX is the index 1-{num_solutions} of the best solution

### Problem

{{problem_statement}}

---

### Candidate Solutions
{solutions_text}
"""
