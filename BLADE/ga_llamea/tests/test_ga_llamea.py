"""Unit tests for GA-LLAMEA standalone package."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ga_llamea import GA_LLaMEA, DiscountedThompsonSampler, calculate_reward


class TestDiscountedThompsonSampler:
    """Test D-TS bandit implementation."""

    def test_initialization(self):
        """Test bandit initialization."""
        bandit = DiscountedThompsonSampler(
            arm_names=["mutation", "crossover", "random_new"],
            discount=0.9,
        )

        assert len(bandit.arms) == 3
        assert "mutation" in bandit.arms
        assert "crossover" in bandit.arms
        assert "random_new" in bandit.arms
        assert bandit.total_pulls == 0

    def test_arm_selection(self):
        """Test arm selection returns valid arm."""
        bandit = DiscountedThompsonSampler(
            arm_names=["mutation", "crossover", "random_new"]
        )

        arm, theta = bandit.select_arm()
        assert arm in ["mutation", "crossover", "random_new"]
        assert isinstance(theta, float)
        assert bandit.total_pulls == 1

    def test_update(self):
        """Test bandit update with reward."""
        bandit = DiscountedThompsonSampler(
            arm_names=["mutation", "crossover", "random_new"]
        )

        # Select and update
        arm, _ = bandit.select_arm()
        initial_count = bandit.arms[arm].discounted_count

        bandit.update(arm, reward=0.5)

        # Check arm was updated
        assert bandit.arms[arm].discounted_count > initial_count
        assert bandit.arms[arm].pulls == 1

    def test_discount_application(self):
        """Test discount is applied correctly."""
        bandit = DiscountedThompsonSampler(
            arm_names=["mutation", "crossover", "random_new"], discount=0.9
        )

        # Make first update
        bandit.update("mutation", reward=1.0)
        first_count = bandit.arms["mutation"].discounted_count

        # Make second update (should discount first)
        bandit.update("mutation", reward=1.0)
        second_count = bandit.arms["mutation"].discounted_count

        # Second count should be: (first_count * 0.9) + 1.0
        expected = (first_count * 0.9) + 1.0
        assert abs(second_count - expected) < 1e-6

    def test_state_snapshot(self):
        """Test state snapshot returns correct structure."""
        bandit = DiscountedThompsonSampler(
            arm_names=["mutation", "crossover", "random_new"]
        )

        snapshot = bandit.get_state_snapshot()

        assert len(snapshot) == 3
        for arm_name in ["mutation", "crossover", "random_new"]:
            assert arm_name in snapshot
            assert "count" in snapshot[arm_name]
            assert "mean" in snapshot[arm_name]
            assert "var" in snapshot[arm_name]
            assert "std" in snapshot[arm_name]
            assert "theta" in snapshot[arm_name]
            assert "pulls" in snapshot[arm_name]


class TestGALLaMEA:
    """Test GA-LLAMEA method implementation."""

    def test_initialization(self):
        """Test GA-LLAMEA initialization."""
        llm = Mock()
        method = GA_LLaMEA(
            llm=llm,
            budget=100,
            n_parents=4,
            n_offspring=16,
            elitism=True,
        )

        assert method.budget == 100
        assert method.n_parents == 4
        assert method.n_offspring == 16
        assert method.elitism is True
        assert method.bandit is not None
        assert len(method.bandit.arms) == 3

    def test_to_dict(self):
        """Test to_dict returns correct structure."""
        llm = Mock()
        method = GA_LLaMEA(llm=llm, budget=100)

        config = method.to_dict()

        assert "method_name" in config
        assert "budget" in config
        assert config["budget"] == 100
        assert "n_parents" in config
        assert "n_offspring" in config
        assert "elitism" in config

    def test_create_solution_default(self):
        """Test default solution creation."""
        llm = Mock()
        method = GA_LLaMEA(llm=llm, budget=100)

        sol = method._create_solution(code="test code", fitness=0.5)
        assert sol.code == "test code"
        assert sol.fitness == 0.5
        assert hasattr(sol, 'metadata')
        assert hasattr(sol, 'error')

    def test_create_solution_custom_class(self):
        """Test custom solution class."""
        llm = Mock()
        
        class CustomSolution:
            def __init__(self, code=""):
                self.code = code
                self.fitness = 0.0
                self.error = ""
                self.name = ""
                self.metadata = {}
        
        method = GA_LLaMEA(llm=llm, budget=100, solution_class=CustomSolution)
        
        sol = method._create_solution(code="custom", fitness=0.8)
        assert isinstance(sol, CustomSolution)
        assert sol.code == "custom"
        assert sol.fitness == 0.8

    def test_extract_code(self):
        """Test code extraction from LLM response."""
        llm = Mock()
        method = GA_LLaMEA(llm=llm, budget=100)

        # Test markdown code block
        response = """
Here is the algorithm:

```python
def optimize(x):
    return x ** 2
```
"""
        code = method._extract_code(response)
        assert code is not None
        assert "def optimize" in code

        # Test without code block
        response_no_block = """
def optimize(x):
    return x ** 2
"""
        code = method._extract_code(response_no_block)
        assert code is not None
        assert "def optimize" in code


def test_calculate_reward():
    """Test reward calculation function."""
    # Valid improvement
    reward = calculate_reward(parent_score=0.5, child_score=0.7, is_valid=True)
    assert reward == pytest.approx(0.2, abs=1e-6)

    # Valid but worse (no negative rewards)
    reward = calculate_reward(parent_score=0.7, child_score=0.5, is_valid=True)
    assert reward == 0.0

    # Invalid code
    reward = calculate_reward(parent_score=0.5, child_score=0.7, is_valid=False)
    assert reward == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
