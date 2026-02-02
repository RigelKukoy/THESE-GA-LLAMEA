import os
import sys
from pathlib import Path

# Add root directory to path to import ga_llamea and iohblade
sys.path.insert(0, str(Path(__file__).parent.parent))

from iohblade.llm import OpenAI_LLM
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB
from iohblade.experiment import Experiment
from iohblade.loggers import ExperimentLogger
import openai

# Import GA-LLAMEA
from ga_llamea import GA_LLaMEA

class AIML_LLM(OpenAI_LLM):
    """AIML API LLM (OpenAI-compatible)"""

    def __init__(
        self,
        api_key,
        model="gpt-4o-mini",
        base_url="https://api.aimlapi.com/v1",
        temperature=0.8,
        **kwargs
    ):
        super(OpenAI_LLM, self).__init__(api_key, model, base_url, **kwargs)
        self._client_kwargs = dict(api_key=api_key, base_url=base_url)
        self.client = openai.OpenAI(**self._client_kwargs)
        self.temperature = temperature

if __name__ == "__main__":
    # Get credentials
    api_key = os.getenv("AIML_API_KEY", "7baca5864cfb4bc4a6553e68e69b8f6a")
    base_url = "https://api.aimlapi.com/v1"
    
    # Initialize LLM
    llm = AIML_LLM(api_key=api_key, model="gpt-4o-mini", base_url=base_url)
    budget = 10

    # Create GA-LLAMEA method
    method = GA_LLaMEA(
        llm=llm,
        budget=budget,
        name="GA-LLAMEA-1",
        solution_class=Solution,
        n_parents=2,
        n_offspring=4,
        elitism=True,
    )

    methods = [method]
    
    # Create problem
    problem = MA_BBOB(
        function_id=1,  # Sphere function
        dimension=5,
        instance=1,
    )
    problems = [problem]

    # Setup logger and experiment
    logger = ExperimentLogger("results/MA-BBOB-GA-LLAMEA")
    experiment = Experiment(
        methods=methods,
        problems=problems,
        runs=1,
        show_stdout=True,
        exp_logger=logger
    )
    
    # Run the experiment
    print("\n🚀 Starting GA-LLAMEA Experiment on MA-BBOB...")
    experiment()
    print("\n✅ Experiment complete!")


