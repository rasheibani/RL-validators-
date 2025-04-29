import spacy
import torch
import warnings
import argparse
import os
import numpy as np
from utils import load_envs
from multi_origin_training import run_multi_origin_experiment

def main():
    parser = argparse.ArgumentParser(description='Multi-origin route instruction learning')
    parser.add_argument('--max_iterations', type=int, default=50000,
                       help='Maximum number of training iterations')
    parser.add_argument('--num_origins', type=int, default=5,
                       help='Number of origins per destination in multi-origin mode')
    parser.add_argument('--num_envs', type=int, default=5,
                       help='Number of environments to use in experiment')
    args = parser.parse_args()
    
    # Set up environment
    print(f"CUDA Available: {torch.cuda.is_available()}")
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print("Loading environments...")
    all_env_pretraining = load_envs()
    
    print("Running multi-origin approach for novel origin generalization...")
    # Sort environments by complexity for even selection
    from utils import determine_complexity
    sorted_envs = sorted(all_env_pretraining, key=lambda x: determine_complexity(x['env']))
    
    # Select evenly distributed environments
    step = max(1, len(sorted_envs) // args.num_envs)
    selected_envs = sorted_envs[::step][:args.num_envs]
    
    print(f"Selected {len(selected_envs)} environments for experiment:")
    for env in selected_envs:
        print(f"- {env['env']}")
    
    # Create output directories if they don't exist
    os.makedirs('data/multi_origin', exist_ok=True)
    os.makedirs('charts/multi_origin', exist_ok=True)
    
    # Run the experiment
    results = run_multi_origin_experiment(
        selected_envs, 
        num_origins=args.num_origins,
        max_iterations=args.max_iterations
    )
    
    if results:
        print("\nExperiment completed successfully!")
        print(f"Number of environments tested: {len(results)}")
        
        # Calculate average improvement metrics
        success_improvements = [r['multi_origin']['success_rate'] - r['baseline']['success_rate'] for r in results]
        efficiency_improvements = [r['multi_origin']['avg_path_efficiency'] - r['baseline']['avg_path_efficiency'] 
                                  for r in results if r['multi_origin']['avg_path_efficiency'] > 0]
        
        print(f"Average success rate improvement: {np.mean(success_improvements):.4f}")
        if efficiency_improvements:
            print(f"Average path efficiency improvement: {np.mean(efficiency_improvements):.4f}")
        
        print("See data/multi_origin_results_all.csv for detailed results")
        print("See charts/multi_origin/ for visualizations")
    else:
        print("No results were generated. Check logs for errors.")

if __name__ == "__main__":
    main()