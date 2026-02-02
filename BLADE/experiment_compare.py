"""
Unified Comparison Experiment: LLAMEA vs GA-LLAMEA

This script runs both the standard LLAMEA baseline and the GA-LLAMEA variant
on the BBOB benchmark to allow direct comparison.

It uses the enhanced logging and prompt guardrails implemented in iohblade.
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="llamea.llm")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.parallel")
warnings.filterwarnings("ignore", message=".*google.generativeai.*")


# Add root directory to path if needed (though running from root usually works)
sys.path.insert(0, str(Path(__file__).parent))

from iohblade.llm import Gemini_LLM
from iohblade.problems import MA_BBOB
from iohblade.experiment import Experiment
from iohblade.loggers import ExperimentLogger
from iohblade.plots import (
    plot_convergence,
    plot_boxplot_fitness_hue,
)

# Import methods from the updated iohblade.methods package
from iohblade.methods import LLaMEA, GA_LLaMEA

def create_comparison_methods(llm, budget):
    """Create both LLAMEA and GA-LLAMEA methods for comparison."""
    
    methods = [
        # Baseline LLAMEA
        LLaMEA(
            llm=llm,
            budget=budget,
            name="LLaMEA-Baseline",
            mutation_prompts=["Refine the strategy of the selected algorithm to improve it."],
            n_parents=4,
            n_offspring=8,
            elitism=True,
        ),
        
        # GA-LLAMEA (using the new implementation in iohblade.methods)
        GA_LLaMEA(
            llm=llm,
            budget=budget,
            name="GA-LLAMEA-0.9-0.2",
            n_parents=4,
            n_offspring=8,
            elitism=True,
            discount=0.9,     # Custom Param
            tau_max=0.2,      # Custom Param
            reward_variance=1.0,
        ),
    ]
    return methods

if __name__ == "__main__":
    # Get credentials - Google Gemini API
    api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyAQnPB1o_TKDNd9dsCftLW-SR7yZGGV8C8")

    print("=" * 80)
    print("Unified Comparison: LLAMEA vs GA-LLAMEA")
    print("=" * 80)

    # Experiment Parameters
    budget = 20
    RUNS = 1      
    dims = [5]    
    budget_factor = 1000
    
    # Use same instances as previous runs
    training_instances = list(range(10))
    test_instances = list(range(20, 50))

    print(f"\nConfiguration:")
    print(f"  Budget: {budget} algorithms per method")
    print(f"  Instances: {len(training_instances)} Train / {len(test_instances)} Test")

    # Initialize LLM
    print("\n🔧 Initializing LLM...")
    llm = Gemini_LLM(
        api_key=api_key,
        model="gemini-2.0-flash",
    )
    print("   ✓ LLM initialized")

    # Create Methods
    print("\n🔧 Creating methods...")
    methods = create_comparison_methods(llm, budget)
    print(f"   ✓ {len(methods)} methods created:")
    for m in methods:
        print(f"     - {m.name}")

    # Create Problem
    print("\n🔧 Creating problem...")
    problem = MA_BBOB(
        training_instances=training_instances,
        test_instances=test_instances,
        dims=dims,
        budget_factor=budget_factor,
        eval_timeout=600,
        name='MA-BBOB-Comparison',
    )
    problems = [problem]

    # Setup Logger with timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"Comparison_LLAMEA_vs_GA_LLAMEA_{timestamp}"
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    logger = ExperimentLogger(f"results/{experiment_name}")
    print(f"\n📁 Results directory: {logger.dirname}")

    # Run Experiment
    print("\nStarting Experiment...")
    try:
        experiment = Experiment(
            methods=methods,
            problems=problems,
            runs=RUNS,
            show_stdout=True,
            exp_logger=logger,
        )
        experiment()

        print("\n📊 Generating visualizations...")
        try:
            plot_convergence(logger, metric="AOCC", save=True, budget=budget)
            plot_boxplot_fitness_hue(logger, hue="method_name", save=True)
            print("   ✓ Visualizations generated")
        except Exception as plot_err:
            print(f"   ⚠️ Visualization failed (but data is saved): {plot_err}")
            print("   You can try running the visualization manually later.")
        
        print(f"\n✅ Done! Results saved in {logger.dirname}")

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
