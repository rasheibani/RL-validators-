import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the CSV file
file_path = 'data/evaluation_results.csv'
df = pd.read_csv(file_path)

# Set a style for more visually appealing charts
plt.style.use("ggplot")



# 2. Success Rate Comparison: Trained vs Random Agents by Complexity (Box Plot)
plt.figure(figsize=(14, 8))
complexities = df['complexity'].unique()
for complexity in complexities:
    subset_seen = df[(df['complexity'] == complexity) & (df['evaluated_env'] == 'seen')]
    subset_unseen = df[(df['complexity'] == complexity) & (df['evaluated_env'] == 'unseen')]
    plt.boxplot([subset_seen['average_success_rate'], subset_unseen['average_success_rate']], positions=[complexities.tolist().index(complexity) - 0.2, complexities.tolist().index(complexity) + 0.2], widths=0.3, patch_artist=True, labels=['Seen', 'Unseen'], showmeans=True)
plt.xlabel('Complexity')
plt.ylabel('Success Rate')
plt.title('Success Rate Comparison: Trained vs Random Agents by Complexity')
plt.xticks(range(len(complexities)), complexities, rotation=45)
plt.show()

# 3. Average Success Rate by Instruction Type and Reward Type (Scatter Plot)
plt.figure(figsize=(12, 8))
instruction_types = df['instruction_type'].unique()
reward_types = df['reward_type'].unique()
colors = ['b', 'r', 'g', 'orange']
for i, reward in enumerate(reward_types):
    subset = df[df['reward_type'] == reward]
    plt.scatter(subset['instruction_type'], subset['average_success_rate'], color=colors[i % len(colors)], label=reward, alpha=0.6)
plt.xlabel('Instruction Type')
plt.ylabel('Average Success Rate')
plt.title('Average Success Rate by Instruction Type and Reward Type')
plt.legend(title='Reward Type')
plt.show()




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

bottom_values = np.zeros(len(instruction_types))
for reward in reward_types:
    subset = df[df['reward_type'] == reward]
    avg_success_rate_per_instruction = subset.groupby('instruction_type')['average_success_rate'].mean()
    plt.bar(avg_success_rate_per_instruction.index, avg_success_rate_per_instruction, bottom=bottom_values[:len(avg_success_rate_per_instruction)], label=f'Reward Type: {reward}', alpha=0.7)
    bottom_values[:len(avg_success_rate_per_instruction)] += avg_success_rate_per_instruction.values

plt.xlabel('Instruction Type')
plt.ylabel('Average Success Rate')
plt.title('Average Success Rate by Instruction Type and Reward Type')
plt.legend(title='Reward Type')
plt.xticks(rotation=45)
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
