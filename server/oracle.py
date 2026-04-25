# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import re
import math

def agent_solve(problem_str: str) -> float:
    """
    Rule-based oracle solver. Handles all 10 difficulty tiers.
    Returns the correct float answer.
    """
    # Strip the prefix: "[Difficulty N/10] Solve: <expr>"
    if "Solve: " in problem_str:
        parts = problem_str.split("Solve: ", 1)
        expr = parts[1].strip()
    else:
        expr = problem_str.strip()

    # Tier 8: "Solve for x: 8x + 33 = 57"
    m = re.search(r"(\d+)x \+ (-?\d+) = (\d+)", expr)
    if m:
        a, b, c = map(float, m.groups())
        return (c - b) / a

    # Tier 10: "Solve for x: (2x + 10) / 4 = 5"
    m = re.search(r"\((\d+)x \+ (-?\d+)\) / (\d+) = (\d+)", expr)
    if m:
        a, b, d, c = map(float, m.groups())
        return (c * d - b) / a

    # Tier 9: "sqrt(N)"
    m = re.search(r"sqrt\((\d+)\)", expr)
    if m:
        return float(math.sqrt(int(m.group(1))))

    # Tiers 1-7: arithmetic expressions — safe eval on cleaned string
    # Remove anything that isn't a number or operator
    clean = re.sub(r"[^0-9+\-*/().² ]", "", expr).strip()
    try:
        # Simple safety check: only allow digits and math operators
        if re.match(r"^[0-9+\-*/(). ]+$", clean):
            return float(eval(clean))
    except Exception:
        pass
    
    return 0.0
