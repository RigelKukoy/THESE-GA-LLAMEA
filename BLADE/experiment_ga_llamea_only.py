"""
GA-LLAMEA Custom Experiment

This script sets up an experiment for GA-LLAMEA with custom parameters:
- tau_max = 0.2
- discount = 0.9
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# Add root directory to path to import ga_llamea and iohblade
sys.path.insert(0, str(Path(__file__).parent))

from iohblade.llm import OpenAI_LLM
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB
from iohblade.experiment import Experiment
from iohblade.loggers import ExperimentLogger
from iohblade.plots import (
    plot_convergence,
    plot_boxplot_fitness_hue,
    plot_experiment_CEG,
)
import openai

# Import GA-LLAMEA from local package
from ga_llamea import GA_LLaMEA

class AIML_LLM(OpenAI_LLM):
    """AIML API LLM (OpenAI-compatible)"""

    def __init__(
        self,
        api_key,
        model="gemini-2.0-flash",
        base_url="https://api.aimlapi.com/v1",
        temperature=0.8,
        **kwargs
    ):
        super(OpenAI_LLM, self).__init__(api_key, model, base_url, **kwargs)
        self._client_kwargs = dict(api_key=api_key, base_url=base_url)
        self.client = openai.OpenAI(**self._client_kwargs)
        self.temperature = temperature

def create_methods(llm, budget):
    """Create GA-LLAMEA method."""
    
    methods = [
        GA_LLaMEA(
            llm=llm,
            budget=budget,
            name="GA-LLAMEA-Custom-0.2-0.9",
            solution_class=Solution,
            n_parents=4,
            n_offspring=8,
            elitism=True,
            discount=0.9,     # User requested: 0.9
            tau_max=0.2,      # User requested: 0.2
            reward_variance=1.0,
        ),
    ]
    return methods

if __name__ == "__main__":
    # Get credentials
    api_key = os.getenv("AIML_API_KEY", "7baca5864cfb4bc4a6553e68e69b8f6a")
    base_url = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")

    print("=" * 80)
    print("GA-LLAMEA Custom Experiment (tau_max=0.2, discount=0.9)")
    print("=" * 80)

    # Experiment Parameters
    budget = 100
    RUNS = 1      
    dims = [5]    
    budget_factor = 1000
    
    training_instances = list(range(10))
    test_instances = list(range(20, 50))

    print(f"\nConfiguration:")
    print(f"  Budget: {budget} algorithms")
    print(f"  tau_max: 0.2")
    print(f"  discount: 0.9")
    print(f"  Instances: {len(training_instances)} Train / {len(test_instances)} Test")

    # Initialize LLM
    print("\n🔧 Initializing LLM...")
    llm = AIML_LLM(
        api_key=api_key,
        model="gemini-2.0-flash",
        base_url=base_url,
    )
    print("   ✓ LLM initialized")

    # Create Methods
    print("\n🔧 Creating methods...")
    methods = create_methods(llm, budget)
    print(f"   ✓ {len(methods)} methods created")

    # Create Problem
    print("\n🔧 Creating problem...")
    problem = MA_BBOB(
        training_instances=training_instances,
        test_instances=test_instances,
        dims=dims,
        budget_factor=budget_factor,
        name='BBOB-GA-LLAMEA-Custom',
    )
    problems = [problem]

    # Setup Logger with timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"GA_LLAMEA_Custom_{timestamp}"
    
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
        plot_convergence(logger, metric="AOCC", save=True, budget=budget)
        plot_boxplot_fitness_hue(logger, hue="method_name", save=True)
        
        print(f"\n✅ Done! Results saved in {logger.dirname}")

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
