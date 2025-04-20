import os
import json
import pickle
from ai_scientist.treesearch.log_summarization import overall_summarize

def generate_report(experiment_dir):
    # Load the manager.pkl file
    manager_path = os.path.join(experiment_dir, "logs/0-run/manager.pkl")
    with open(manager_path, "rb") as f:
        manager = pickle.load(f)
    
    # Get the journals from the manager
    journals = [(stage_name, journal) for stage_name, journal in manager.journals.items()]
    
    # Generate summaries for all stages
    results = overall_summarize(journals)
    
    # Save the results
    results_path = os.path.join(experiment_dir, "logs/0-run/results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    import sys
    experiment_dir = sys.argv[1]
    generate_report(experiment_dir) 