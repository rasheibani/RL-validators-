import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the CSV file
file_path = 'data/curriculum_evaluation_results.csv'
df = pd.read_csv(file_path)

# Set a style for more visually appealing charts
plt.style.use("ggplot")



plt.figure(figsize=(14, 8))

# Plotting grouped bar chart for Average Success Rate vs Complexity by Grammar Level
complexity_levels = df['complexity'].unique()
grammar_levels = df['grammar'].unique()
bar_width = 0.2  # Width of the bars

# Define positions for each bar group
indices = np.arange(len(complexity_levels))

# Plotting bars for each grammar level
for i, grammar in enumerate(grammar_levels):
    subset = df[df['grammar'] == grammar]
    avg_success_rate_per_complexity = subset.groupby('complexity')['average_success_rate'].mean().reindex(complexity_levels)
    plt.bar(indices + i * bar_width, avg_success_rate_per_complexity, width=bar_width, label=f'Grammar Level {grammar}', alpha=0.7)

# Setting labels and title
plt.xlabel('Complexity')
plt.ylabel('Average Success Rate')
plt.title('Average Success Rate vs Complexity and Grammar Level')
plt.xticks(indices + bar_width * (len(grammar_levels) / 2), complexity_levels, rotation=45)
plt.legend(title='Grammar Level')
plt.grid(axis='y', linestyle='--')

# Displaying the plot
plt.show()


# 3. Average Success Rate by Instruction Type and Reward Type (Stacked Bar Chart)
plt.figure(figsize=(12, 8))
instruction_types = df['instruction_type'].unique()
reward_types = df['reward_type'].unique()
bar_width = 0.35  # Width of each bar
x_positions = np.arange(len(instruction_types)) # Positions for each instruction type group

# Iterate through reward types and plot bars for each group
for i, reward in enumerate(reward_types):
    subset = df[df['reward_type'] == reward]
    avg_success_rate_per_instruction = subset.groupby('instruction_type')['average_success_rate'].mean()
    # Calculate x positions for each bar within the group
    bar_positions = x_positions + i * bar_width - (len(reward_types) -1) * bar_width / 2 # Adjust to center groups

    plt.bar(bar_positions, avg_success_rate_per_instruction[instruction_types], width=bar_width, label=f'Reward Type: {reward}', alpha=0.7) # use instruction_types to maintain order


plt.xlabel('Instruction Type')
plt.ylabel('Average Success Rate')
plt.title('Average Success Rate by Instruction Type and Reward Type (Grouped Bar Chart)')
plt.legend(title='Reward Type')

# Set x ticks in the center of each group and label them with instruction types
plt.xticks(x_positions, instruction_types, rotation=45, ha='right')
plt.tight_layout() # Adjust layout to prevent labels from being cut off
plt.show()


# Setting up the figure size
plt.figure(figsize=(14, 8))

# Creating subsets for complete and incomplete instructions for both trained and random agents
complete_trained_success = df[df['instruction_type'] == 'complete']['average_success_rate']
incomplete_trained_success = df[df['instruction_type'] == 'incomplete']['average_success_rate']
complete_random_success = df[df['instruction_type'] == 'complete']['random_agent_average_success_rate']
incomplete_random_success = df[df['instruction_type'] == 'incomplete']['random_agent_average_success_rate']

# Plotting the violin plot
plt.violinplot([complete_trained_success, incomplete_trained_success, complete_random_success, incomplete_random_success], showmeans=True)
plt.xticks([1, 2, 3, 4], ['Complete - Trained', 'Incomplete - Trained', 'Complete - Random', 'Incomplete - Random'])
plt.ylabel('Success Rate')
plt.title('Success Rate Distribution: Complete vs Incomplete Route Instructions')
plt.grid(axis='y', linestyle='--')

# Displaying the plot
plt.show()

# Setting up the figure size for the pair plot
plt.figure(figsize=(14, 8))

# Creating a scatter matrix to examine relationships between different metrics
from pandas.plotting import scatter_matrix

# Selecting key metrics to visualize
key_metrics = df[['average_success_rate', 'std_success_rate', 'random_agent_average_success_rate', 'random_agent_std_success_rate']]

# Creating the scatter matrix
scatter_matrix(key_metrics, alpha=0.7, figsize=(14, 14), diagonal='kde', marker='o', hist_kwds={'bins': 20}, color='purple')

# Adding a title for better understanding
plt.suptitle('Pair Plot Matrix: Key Metrics Relationships', size=16)

# Displaying the plot
plt.show()