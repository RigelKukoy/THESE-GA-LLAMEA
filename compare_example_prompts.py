import json
import sys
import os

# Add path to BLADE (standard append)
sys.path.append(os.path.join(os.getcwd(), 'BLADE'))

from iohblade.problems.mabbob import MA_BBOB
import inspect
print(f"DEBUG: MA_BBOB file: {inspect.getfile(MA_BBOB)}")


def normalize_string(s):
    return "\n".join([line.strip() for line in s.strip().splitlines() if line.strip()])

def compare_prompts():
    # 1. Get Log Prompt
    log_path = r"c:\Users\Kukoy\OneDrive\Documents\BLADE-THESIS\MA_BBOB\run-LLaMEA-3-MA_BBOB-8\conversationlog.jsonl"
    log_example_prompt = ""
    
    with open(log_path, 'r') as f:
        # Read first line/message which usually contains the system prompt
        for line in f:
            data = json.loads(line)
            content = data.get('content', '')
            if "An example of such code" in content:
                # Extract the example part
                start_marker = "An example of such code"
                end_marker = "Give an excellent and novel heuristic"
                try:
                    start_idx = content.index(start_marker)
                    end_idx = content.index(end_marker)
                    log_example_prompt = content[start_idx:end_idx].strip()
                    break
                except ValueError:
                    continue
    
    if not log_example_prompt:
        print("Could not find example prompt in log file.")
        return

    # 2. Get Current Code Prompt
    problem = MA_BBOB()
    code_example_prompt = problem.example_prompt.strip()

    # 3. Compare
    print("--- Log File Example Prompt ---")
    print(log_example_prompt)
    print("\n--- Current Code Example Prompt ---")
    print(code_example_prompt)
    print("\n--- Comparison ---")
    
    # Simple normalization for comparison (ignore leading/trailing whitespace per line)
    norm_log = normalize_string(log_example_prompt)
    norm_code = normalize_string(code_example_prompt)
    
    if norm_log == norm_code:
        print("✅ The example prompts are IDENTICAL (ignoring whitespace).")
    else:
        print("❌ The example prompts are DIFFERENT.")
        
        # Print Diff
        import difflib
        diff = difflib.ndiff(norm_log.splitlines(), norm_code.splitlines())
        print("\nDifferences:")
        for line in diff:
            if line.startswith('+') or line.startswith('-'):
                print(line)

if __name__ == "__main__":
    compare_prompts()
