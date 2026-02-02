
import sys
import pandas as pd
import json
import matplotlib.pyplot as plt

def load_log(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return pd.DataFrame(data)

def analyze(df, name):
    print(f"--- Analysis for {name} ---")
    print(f"Total entries: {len(df)}")
    if 'fitness' in df.columns:
        print(f"Average Fitness: {df['fitness'].mean()}")
        print(f"Max Fitness: {df['fitness'].max()}")
    
    if 'error' in df.columns:
        errors = df[df['error'] != ""]
        print(f"Total Errors: {len(errors)}")
        print("Error Counts:")
        print(errors['error'].value_counts())
    
    if 'description' in df.columns:
        missing_desc = len(df[df['description'] == ""])
        print(f"Missing Descriptions: {missing_desc}")

    print("\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_logs.py <path_to_ga_log> <path_to_baseline_log>")
        sys.exit(1)
        
    ga_path = sys.argv[1]
    baseline_path = sys.argv[2]
    
    print(f"Loading GA Log: {ga_path}")
    df_ga = load_log(ga_path)
    
    print(f"Loading Baseline Log: {baseline_path}")
    df_baseline = load_log(baseline_path)
    
    analyze(df_ga, "GA-LLAMEA")
    analyze(df_baseline, "LLAMEA Baseline")
    
