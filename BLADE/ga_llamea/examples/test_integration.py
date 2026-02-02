"""Test Integration Script for GA-LLAMEA.

This script verifies that GA-LLAMEA integrates correctly with BLADE
by testing imports, class instantiation, and basic functionality.

Usage:
    cd BLADE
    uv run python ga_llamea/examples/test_integration.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, passed: bool, message: str = ""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} | {test_name}")
    if message:
        print(f"         └─ {message}")


def test_imports():
    """Test that all imports work correctly."""
    print_header("Test 1: Import Verification")
    
    tests_passed = 0
    tests_total = 0
    
    # Test GA_LLaMEA import
    tests_total += 1
    try:
        from ga_llamea import GA_LLaMEA
        print_result("Import GA_LLaMEA", True)
        tests_passed += 1
    except ImportError as e:
        print_result("Import GA_LLaMEA", False, str(e))
    
    # Test DiscountedThompsonSampler import
    tests_total += 1
    try:
        from ga_llamea import DiscountedThompsonSampler
        print_result("Import DiscountedThompsonSampler", True)
        tests_passed += 1
    except ImportError as e:
        print_result("Import DiscountedThompsonSampler", False, str(e))
    
    # Test calculate_reward import
    tests_total += 1
    try:
        from ga_llamea import calculate_reward
        print_result("Import calculate_reward", True)
        tests_passed += 1
    except ImportError as e:
        print_result("Import calculate_reward", False, str(e))
    
    # Test Protocol imports
    tests_total += 1
    try:
        from ga_llamea import LLMProtocol, SolutionProtocol, ProblemProtocol
        print_result("Import Protocol interfaces", True)
        tests_passed += 1
    except ImportError as e:
        print_result("Import Protocol interfaces", False, str(e))
    
    return tests_passed, tests_total


def test_class_instantiation():
    """Test that GA_LLaMEA can be instantiated."""
    print_header("Test 2: Class Instantiation")
    
    tests_passed = 0
    tests_total = 0
    
    from unittest.mock import Mock
    from ga_llamea import GA_LLaMEA
    
    # Test basic instantiation
    tests_total += 1
    try:
        llm = Mock()
        method = GA_LLaMEA(llm=llm, budget=100)
        assert method.budget == 100
        assert method.name == "GA-LLAMEA"  # Default name should be GA-LLAMEA
        print_result("Basic instantiation", True, "Default name is 'GA-LLAMEA'")
        tests_passed += 1
    except Exception as e:
        print_result("Basic instantiation", False, str(e))
    
    # Test with custom parameters
    tests_total += 1
    try:
        llm = Mock()
        method = GA_LLaMEA(
            llm=llm,
            budget=50,
            n_parents=8,
            n_offspring=32,
            elitism=False,
            discount=0.8,
        )
        assert method.n_parents == 8
        assert method.n_offspring == 32
        assert method.elitism is False
        print_result("Custom parameters", True)
        tests_passed += 1
    except Exception as e:
        print_result("Custom parameters", False, str(e))
    
    return tests_passed, tests_total


def test_bandit_functionality():
    """Test the D-TS bandit functionality."""
    print_header("Test 3: Bandit Functionality")
    
    tests_passed = 0
    tests_total = 0
    
    from ga_llamea import DiscountedThompsonSampler
    
    # Test arm selection
    tests_total += 1
    try:
        bandit = DiscountedThompsonSampler(
            arm_names=["mutation", "crossover", "random_new"],
            discount=0.9,
        )
        arm, theta = bandit.select_arm()
        assert arm in ["mutation", "crossover", "random_new"]
        assert isinstance(theta, float)
        print_result("Arm selection", True, f"Selected: {arm} (θ={theta:.4f})")
        tests_passed += 1
    except Exception as e:
        print_result("Arm selection", False, str(e))
    
    # Test bandit update
    tests_total += 1
    try:
        bandit = DiscountedThompsonSampler(
            arm_names=["mutation", "crossover", "random_new"],
            discount=0.9,
        )
        bandit.update("mutation", reward=0.5)
        assert bandit.arms["mutation"].pulls == 1
        print_result("Bandit update", True)
        tests_passed += 1
    except Exception as e:
        print_result("Bandit update", False, str(e))
    
    # Test state snapshot
    tests_total += 1
    try:
        bandit = DiscountedThompsonSampler(
            arm_names=["mutation", "crossover", "random_new"],
            discount=0.9,
        )
        snapshot = bandit.get_state_snapshot()
        assert len(snapshot) == 3
        assert all(key in snapshot for key in ["mutation", "crossover", "random_new"])
        print_result("State snapshot", True)
        tests_passed += 1
    except Exception as e:
        print_result("State snapshot", False, str(e))
    
    return tests_passed, tests_total


def test_blade_solution_integration():
    """Test integration with BLADE's Solution class."""
    print_header("Test 4: BLADE Solution Integration")
    
    tests_passed = 0
    tests_total = 0
    
    # Test with BLADE's Solution class
    tests_total += 1
    try:
        from iohblade.solution import Solution
        from unittest.mock import Mock
        from ga_llamea import GA_LLaMEA
        
        llm = Mock()
        method = GA_LLaMEA(
            llm=llm,
            budget=100,
            solution_class=Solution,
        )
        
        # Create a solution using the method's factory
        sol = method._create_solution(code="def test(): pass", fitness=0.5)
        assert isinstance(sol, Solution)
        assert sol.code == "def test(): pass"
        assert sol.fitness == 0.5
        print_result("BLADE Solution integration", True)
        tests_passed += 1
    except ImportError:
        print_result("BLADE Solution integration", False, "Could not import iohblade.solution")
    except Exception as e:
        print_result("BLADE Solution integration", False, str(e))
    
    return tests_passed, tests_total


def test_reward_calculation():
    """Test the reward calculation function."""
    print_header("Test 5: Reward Calculation")
    
    tests_passed = 0
    tests_total = 0
    
    from ga_llamea import calculate_reward
    
    # Test improvement reward
    tests_total += 1
    try:
        reward = calculate_reward(parent_score=0.5, child_score=0.7, is_valid=True)
        assert abs(reward - 0.2) < 1e-6
        print_result("Improvement reward", True, f"Reward = {reward:.4f}")
        tests_passed += 1
    except Exception as e:
        print_result("Improvement reward", False, str(e))
    
    # Test no negative rewards
    tests_total += 1
    try:
        reward = calculate_reward(parent_score=0.7, child_score=0.5, is_valid=True)
        assert reward == 0.0
        print_result("No negative rewards", True)
        tests_passed += 1
    except Exception as e:
        print_result("No negative rewards", False, str(e))
    
    # Test invalid code reward
    tests_total += 1
    try:
        reward = calculate_reward(parent_score=0.5, child_score=0.7, is_valid=False)
        assert reward == 0.0
        print_result("Invalid code penalty", True)
        tests_passed += 1
    except Exception as e:
        print_result("Invalid code penalty", False, str(e))
    
    return tests_passed, tests_total


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("   GA-LLAMEA Integration Test Suite")
    print("=" * 60)
    
    total_passed = 0
    total_tests = 0
    
    # Run all test groups
    for test_func in [
        test_imports,
        test_class_instantiation,
        test_bandit_functionality,
        test_blade_solution_integration,
        test_reward_calculation,
    ]:
        passed, total = test_func()
        total_passed += passed
        total_tests += total
    
    # Print summary
    print_header("Summary")
    print(f"\n  Total Tests: {total_tests}")
    print(f"  Passed:      {total_passed}")
    print(f"  Failed:      {total_tests - total_passed}")
    
    if total_passed == total_tests:
        print("\n  🎉 All tests passed! GA-LLAMEA is ready to use with BLADE.")
    else:
        print(f"\n  ⚠️  {total_tests - total_passed} test(s) failed. Please check the errors above.")
    
    print("\n" + "=" * 60)
    
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
