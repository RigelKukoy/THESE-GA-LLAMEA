import json
import sys
import os

# Add path to BLADE (standard append)
sys.path.append(os.path.join(os.getcwd(), 'BLADE'))

from iohblade.problems.mabbob import MA_BBOB
from iohblade.methods.ga_llamea import GA_LLaMEA

# Mock LLM
class MockLLM:
    pass

def normalize_string(s):
    # Normalize by splitlines and strip to ignore whitespace diffs
    return "\n".join([line.strip() for line in s.strip().splitlines() if line.strip()])

def get_log_task_prompt():
    log_path = r"c:\Users\Kukoy\OneDrive\Documents\BLADE-THESIS\MA_BBOB\run-LLaMEA-3-MA_BBOB-8\conversationlog.jsonl"
    with open(log_path, 'r') as f:
        # First message usually has the prompt
        for line in f:
            data = json.loads(line)
            content = data.get('content', '')
            if "An example of such code" in content:
                # The task prompt is everything BEFORE the example
                return content.split("An example of such code")[0].strip()
    return None

def compare():
    print("--- Comparing Task Prompts ---\n")
    
    # 1. Get Target Prompt from Log
    log_prompt = get_log_task_prompt()
    if not log_prompt:
        print("❌ Could not find prompt in log.")
        return

    # 2. Get MA_BBOB Prompt
    problem = MA_BBOB()
    mabbob_prompt = problem.task_prompt.strip()

    # 3. Get GA_LLAMEA Prompt (part of it)
    ga = GA_LLaMEA(llm=MockLLM(), budget=100)
    # GA_LLAMEA uses _get_task_prompt which combines task_prompt + example + etc.
    # checking the first part manually
    
    # GA_LLAMEA constructs it as:
    # prompt = f"""{text}..."""
    # We can inspect the code string manually or inspect the output
    ga_full_prompt = ga._get_task_prompt(problem)
    ga_task_part = ga_full_prompt.split("An example of such code")[0].strip()

    # Normalize
    norm_log = normalize_string(log_prompt)
    norm_mabbob = normalize_string(mabbob_prompt)
    norm_ga = normalize_string(ga_task_part)

    print("Target VS MA_BBOB:")
    if norm_log == norm_mabbob:
        print("✅ Identical")
    else:
        print("❌ Different")
        import difflib
        print('\n'.join(difflib.ndiff(norm_log.splitlines(), norm_mabbob.splitlines())))

    print("\nTarget VS GA_LLAMEA:")
    if norm_log == norm_ga:
        print("✅ Identical")
    else:
        print("❌ Different")
        import difflib
        print('\n'.join(difflib.ndiff(norm_log.splitlines(), norm_ga.splitlines())))

if __name__ == "__main__":
    compare()
