"""GA-LLAMEA: Standalone GA-LLAMEA package for BLADE integration.

This package provides GA-LLAMEA (LLaMEA with Discounted Thompson Sampling)
as a standalone module that integrates with BLADE with zero code changes required.

Example Usage:
    from iohblade.llm import LLM
    from iohblade.solution import Solution
    from iohblade.problems import MA_BBOB
    from ga_llamea import GA_LLaMEA
    
    llm = LLM(model="gemini-2.0-flash")
    method = GA_LLaMEA(
        llm=llm,
        budget=100,
        solution_class=Solution,
    )
    problem = MA_BBOB(function_id=1, dimension=5, instance=1)
    best = method(problem)
"""

from .ds_ts import DiscountedThompsonSampler, ArmState
from .ga_llamea import GA_LLaMEA, calculate_reward
from .interfaces import LLMProtocol, SolutionProtocol, ProblemProtocol

__version__ = "1.0.0"
__all__ = [
    "GA_LLaMEA",
    "DiscountedThompsonSampler",
    "ArmState",
    "calculate_reward",
    "LLMProtocol",
    "SolutionProtocol", 
    "ProblemProtocol",
]
