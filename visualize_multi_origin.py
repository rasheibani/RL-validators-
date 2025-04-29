import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_multi_origin_results():
    """
    Visualizes and analyzes the results of the multi-origin experiments.
    This function creates various charts comparing multi-origin training with baseline
    training approaches to demonstrate the benefits of learning from multiple origins.
    """
    results_path = 'data/multi_origin_results_all.csv'
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("Please run the multi-origin experiment first using main_multi_origin.py")
        return
    
    print("Loading and analyzing results...")
    results_df = pd.read_csv(results_path)
    
    # Create output directory for charts
    os.makedirs('charts/multi_origin', exist_ok=True)
    
    # Calculate improvement metrics
    results_df['success_improvement'] = results_df['multi_origin_success_rate'] - results_df['baseline_success_rate']
    results_df['efficiency_improvement'] = results_df['multi_origin_path_efficiency'] - results_df['baseline_path_efficiency']
    
    # Print summary statistics
    print("\n=== Multi-Origin vs Baseline Performance ===")
    print(f"Average success rate (Multi-Origin): {results_df['multi_origin_success_rate'].mean():.4f}")
    print(f"Average success rate (Baseline): {results_df['baseline_success_rate'].mean():.4f}")
    print(f"Average success rate improvement: {results_df['success_improvement'].mean():.4f}")
    print(f"Average path efficiency (Multi-Origin): {results_df['multi_origin_path_efficiency'].mean():.4f}")
    print(f"Average path efficiency (Baseline): {results_df['baseline_path_efficiency'].mean():.4f}")
    print(f"Average path efficiency improvement: {results_df['efficiency_improvement'].mean():.4f}")
    
    # Set the style for all plots
    sns.set(style="whitegrid")
    
    # 1. Create bar chart comparing success rates
    plt.figure(figsize=(12, 7))
    
    # Prepare data for grouped bar chart
    env_names = results_df['env_name']
    x = np.arange(len(env_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, results_df['multi_origin_success_rate'], width, label='Multi-Origin', color='#3498db')
    rects2 = ax.bar(x + width/2, results_df['baseline_success_rate'], width, label='Baseline (Single Origin)', color='#e74c3c')
    
    # Add labels and title
    ax.set_xlabel('Environment', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Novel Origin Generalization: Success Rates', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, rotation=45, ha='right')
    ax.legend(fontsize=12)
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('charts/multi_origin/success_rate_comparison.png')
    
    # 2. Create bar chart for path efficiency
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, results_df['multi_origin_path_efficiency'], width, label='Multi-Origin', color='#3498db')
    rects2 = ax.bar(x + width/2, results_df['baseline_path_efficiency'], width, label='Baseline (Single Origin)', color='#e74c3c')
    
    ax.set_xlabel('Environment', fontsize=12)
    ax.set_ylabel('Path Efficiency', fontsize=12)
    ax.set_title('Novel Origin Generalization: Path Efficiency', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, rotation=45, ha='right')
    ax.legend(fontsize=12)
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('charts/multi_origin/path_efficiency_comparison.png')
    
    # 3. Create bar chart for improvement metrics
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create a horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Create bars with different colors based on positive/negative values
    colors = ['green' if x > 0 else 'red' for x in results_df['success_improvement']]
    bars = ax.bar(x, results_df['success_improvement'], color=colors)
    
    ax.set_xlabel('Environment', fontsize=12)
    ax.set_ylabel('Success Rate Improvement', fontsize=12)
    ax.set_title('Novel Origin Generalization: Success Rate Improvement\nMulti-Origin vs Baseline', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        y_pos = height + 0.01 if height > 0 else height - 0.05
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha='center', va='center')
    
    fig.tight_layout()
    plt.savefig('charts/multi_origin/success_improvement.png')
    
    # 4. Create a boxplot to show the distribution of improvements
    plt.figure(figsize=(10, 6))
    
    improvement_data = pd.DataFrame({
        'Success Rate Improvement': results_df['success_improvement'],
        'Path Efficiency Improvement': results_df['efficiency_improvement']
    })
    
    # Melt the data for seaborn
    melted_data = pd.melt(improvement_data, var_name='Metric', value_name='Improvement')
    
    # Create boxplot
    sns.boxplot(x='Metric', y='Improvement', data=melted_data)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.title('Distribution of Improvement Metrics', fontsize=14)
    plt.tight_layout()
    plt.savefig('charts/multi_origin/improvement_distribution.png')
    
    # 5. Create a scatter plot to show relationship between metrics
    plt.figure(figsize=(9, 8))
    
    sns.scatterplot(
        data=results_df,
        x='multi_origin_success_rate',
        y='baseline_success_rate',
        s=100,
        alpha=0.7
    )
    
    # Add a diagonal line (y=x)
    max_val = max(results_df['multi_origin_success_rate'].max(), results_df['baseline_success_rate'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    # Mark points above the line
    for i, row in results_df.iterrows():
        if row['multi_origin_success_rate'] > row['baseline_success_rate']:
            plt.annotate(row['env_name'], 
                         (row['multi_origin_success_rate'], row['baseline_success_rate']),
                         xytext=(5, -5), 
                         textcoords='offset points',
                         fontsize=8)
    
    plt.title('Multi-Origin vs Baseline Success Rate', fontsize=14)
    plt.xlabel('Multi-Origin Success Rate', fontsize=12)
    plt.ylabel('Baseline Success Rate', fontsize=12)
    plt.tight_layout()
    plt.savefig('charts/multi_origin/success_rate_scatter.png')
    
    print("\nVisualization complete! Charts saved to charts/multi_origin/ directory.")

if __name__ == "__main__":
    visualize_multi_origin_results()