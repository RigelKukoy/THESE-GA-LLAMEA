"""Protocol-based interfaces for BLADE compatibility.

These protocols define the expected interfaces that BLADE classes must satisfy.
Using Protocols allows GA-LLAMEA to work with any implementation that provides
these methods/attributes, requiring zero changes to BLADE code.
"""

from typing import Protocol, Dict, Any, runtime_checkable


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM interface.
    
    BLADE's LLM class satisfies this protocol automatically.
    """
    
    def query(self, prompt: str) -> str:
        """Query the LLM with a prompt and return the response."""
        ...


@runtime_checkable
class SolutionProtocol(Protocol):
    """Protocol for Solution interface.
    
    BLADE's Solution class satisfies this protocol automatically.
    """
    
    code: str
    fitness: float
    error: str
    name: str
    metadata: Dict[str, Any]


@runtime_checkable
class ProblemProtocol(Protocol):
    """Protocol for Problem interface.
    
    BLADE's Problem class satisfies this protocol automatically.
    """
    
    llm_call_counter: int
    task_prompt: str
    format_prompt: str
    
    def evaluate(self, solution: Any) -> None:
        """Evaluate a solution and update its fitness."""
        ...


# Type alias for flexibility
LLM = LLMProtocol
Solution = SolutionProtocol
Problem = ProblemProtocol
