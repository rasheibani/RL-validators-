import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_envs
from multi_origin_training import run_multi_origin_experiment
from environment import normalize, admissible_actions_to_observation

def main():
    # Configure experiment parameters
    num_origins = 5
    max_iterations = 50000
    num_selected_envs = 5  # Number of environments to use in experiment
    
    print("Loading environment data...")
    # Load environments for the experiment
    all_envs = load_envs()
    
    # Select a subset of environments based on complexity
    from utils import determine_complexity
    
    # Sort environments by complexity to get a diverse sample
    sorted_envs = sorted(all_envs, key=lambda x: determine_complexity(x['env']))
    
    # Select evenly distributed environments from the sorted list
    step = max(1, len(sorted_envs) // num_selected_envs)
    selected_envs = sorted_envs[::step][:num_selected_envs]
    
    print(f"Selected {len(selected_envs)} environments for experiment:")
    for env in selected_envs:
        print(f"- {env['env']}")
    
    # Run the multi-origin experiment
    print("\nStarting multi-origin experiment...")
    results = run_multi_origin_experiment(selected_envs, num_origins=num_origins, max_iterations=max_iterations)
    
    # If results were generated, analyze and visualize them
    if os.path.exists('data/multi_origin_results_all.csv'):
        analyze_results('data/multi_origin_results_all.csv')

def analyze_results(results_path):
    """Analyze and visualize the results of the experiment"""
    print("\nAnalyzing results...")
    results_df = pd.read_csv(results_path)
    
    # Create output directory for charts
    os.makedirs('charts/multi_origin', exist_ok=True)
    
    # Calculate improvement metrics
    results_df['success_improvement'] = results_df['multi_origin_success_rate'] - results_df['baseline_success_rate']
    results_df['efficiency_improvement'] = results_df['multi_origin_path_efficiency'] - results_df['baseline_path_efficiency']
    
    # Print summary statistics
    print("\nMulti-Origin vs Baseline Performance:")
    print(f"Average success rate (Multi-Origin): {results_df['multi_origin_success_rate'].mean():.4f}")
    print(f"Average success rate (Baseline): {results_df['baseline_success_rate'].mean():.4f}")
    print(f"Average success rate improvement: {results_df['success_improvement'].mean():.4f}")
    print(f"Average path efficiency (Multi-Origin): {results_df['multi_origin_path_efficiency'].mean():.4f}")
    print(f"Average path efficiency (Baseline): {results_df['baseline_path_efficiency'].mean():.4f}")
    print(f"Average path efficiency improvement: {results_df['efficiency_improvement'].mean():.4f}")
    
    # Create visualization 1: Success Rate Comparison
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(results_df))
    
    plt.bar(x - width/2, results_df['multi_origin_success_rate'], width, label='Multi-Origin')
    plt.bar(x + width/2, results_df['baseline_success_rate'], width, label='Baseline')
    
    plt.xlabel('Environment')
    plt.ylabel('Success Rate')
    plt.title('Success Rate: Multi-Origin vs Baseline')
    plt.xticks(x, results_df['env_name'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('charts/multi_origin/success_rate_comparison.png')
    
    # Create visualization 2: Path Efficiency Comparison
    plt.figure(figsize=(10, 6))
    
    plt.bar(x - width/2, results_df['multi_origin_path_efficiency'], width, label='Multi-Origin')
    plt.bar(x + width/2, results_df['baseline_path_efficiency'], width, label='Baseline')
    
    plt.xlabel('Environment')
    plt.ylabel('Path Efficiency')
    plt.title('Path Efficiency: Multi-Origin vs Baseline')
    plt.xticks(x, results_df['env_name'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('charts/multi_origin/path_efficiency_comparison.png')
    
    # Create visualization 3: Success Rate Improvement
    plt.figure(figsize=(10, 6))
    
    plt.bar(x, results_df['success_improvement'], color='green')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.xlabel('Environment')
    plt.ylabel('Success Rate Improvement')
    plt.title('Success Rate Improvement with Multi-Origin Training')
    plt.xticks(x, results_df['env_name'], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('charts/multi_origin/success_improvement.png')
    
    # Create visualization 4: Path Efficiency Improvement
    plt.figure(figsize=(10, 6))
    
    plt.bar(x, results_df['efficiency_improvement'], color='purple')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.xlabel('Environment')
    plt.ylabel('Path Efficiency Improvement')
    plt.title('Path Efficiency Improvement with Multi-Origin Training')
    plt.xticks(x, results_df['env_name'], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('charts/multi_origin/efficiency_improvement.png')
    
    # Create visualization 5: Combined metrics heatmap
    plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
    heatmap_data = results_df.pivot(index='env_name', 
                                   columns=['multi_origin_success_rate', 'baseline_success_rate',
                                           'multi_origin_path_efficiency', 'baseline_path_efficiency'])
    
    # Create correlation heatmap
    correlation_data = results_df[['multi_origin_success_rate', 'baseline_success_rate',
                                   'multi_origin_path_efficiency', 'baseline_path_efficiency']]
    correlation_matrix = correlation_data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Performance Metrics')
    plt.tight_layout()
    plt.savefig('charts/multi_origin/metric_correlations.png')
    
    # Create pair plot for deeper analysis
    plt.figure(figsize=(12, 10))
    g = sns.pairplot(results_df[['multi_origin_success_rate', 'baseline_success_rate',
                                'multi_origin_path_efficiency', 'baseline_path_efficiency']])
    g.fig.suptitle('Relationships Between Performance Metrics', y=1.02)
    plt.tight_layout()
    g.savefig('charts/multi_origin/metrics_pairplot.png')
    
    print("\nVisualization complete! Check the 'charts/multi_origin/' directory for results.")

if __name__ == "__main__":
    main()