import sys
import os

# Ensure BLADE/iohblade can be imported
# If running from BLADE-THESIS root:
current_dir = os.getcwd()
if os.path.exists(os.path.join(current_dir, "BLADE", "iohblade")):
    sys.path.insert(0, os.path.join(current_dir, "BLADE"))
else:
    # If running from a subdirectory, adjust path accordingly
    sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..", "..", "BLADE")))

from iohblade.methods.ga_llamea import GA_LLaMEA
from iohblade.solution import Solution

# Mock LLM and Problem for instantiation
class MockLLM:
    def __call__(self, prompt):
        return {"content": "Sample response"}
    def query(self, messages):
        return "Sample code response"

class MockProblem:
    def __init__(self, name, prompt_format):
        self.name = name
        self.example_prompt = "# Example code here" 
        self.format_prompt = "Provide the Python code..."

# --- Verification ---
print("Verifying Prompts...")

# 2. Verify GA_LLAMEA prompt
print("Checking GA_LLAMEA...")
try:
    ga_llamea = GA_LLaMEA(MockLLM(), budget=1000, problem_name="mock_problem", solution_class=Solution)
    
    # Mock problem instance
    mock_problem_instance = MockProblem("mock_problem", "Format prompt string")
    
    # 1. Verify Task Prompt
    task_prompt = ga_llamea._get_task_prompt(mock_problem_instance)
    
    persona_check = "computer scientist" in task_prompt
    excellent_check = "excellent and novel" in task_prompt
    
    print(f"✅ GA_LLAMEA Task Prompt: Persona found: {persona_check}")
    print(f"✅ GA_LLAMEA Task Prompt: 'excellent and novel' found: {excellent_check}")
    
    if not persona_check or not excellent_check:
        print("❌ GA_LLAMEA Task Prompt verification failed.")
        print(f"Prompt content mismatch.")

    # 2. Verify Mutation Prompt and History
    # Create some mock population
    sol1 = Solution(code="code1", name="Algo1", description="Desc1")
    sol1.fitness = 0.5
    sol2 = Solution(code="code2", name="Algo2", description="Desc2")
    sol2.fitness = 0.8
    ga_llamea.population = [sol1, sol2]
    
    mutation_prompt = ga_llamea._build_mutation_prompt(sol2, mock_problem_instance)
    
    # Check History Format
    history_header_check = "The current population of algorithms already evaluated (name, description, score) is:" in mutation_prompt
    history_entry_check = "Algo2: Desc2 (Score: 0.8)" in mutation_prompt
    
    print(f"✅ GA_LLAMEA History: Header found: {history_header_check}")
    print(f"✅ GA_LLAMEA History: Entry format correct: {history_entry_check}")
    
    # Check Mutation Specifics
    selected_update_check = "The selected solution to update is:" in mutation_prompt
    desc_check = "Desc2" in mutation_prompt
    instruction_check = "Refine the strategy of the selected algorithm to improve it." in mutation_prompt
    
    print(f"✅ GA_LLAMEA Mutation: 'Selected solution' text found: {selected_update_check}")
    print(f"✅ GA_LLAMEA Mutation: Description included: {desc_check}")
    print(f"✅ GA_LLAMEA Mutation: Instruction 'Refine the strategy' found: {instruction_check}")
    
    if not history_header_check or not instruction_check:
        print("❌ GA_LLAMEA Mutation Prompt verification failed.")

except Exception as e:
    print(f"❌ Error verifying GA_LLAMEA: {e}")
    import traceback
    traceback.print_exc()

print("Verification complete.")
sys.exit(0)
