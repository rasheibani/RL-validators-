
# Import pandas as it was not defined in this session
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from sklearn.metrics import auc
# Load the new CSV file
file_path = 'data/evaluation_result_DQNs.csv'
df = pd.read_csv(file_path)
# Load the corrected CSV file
df_corrected = df

# Convert 'Mean Reward' column from string representation of list to actual list
df_corrected['Mean Reward'] = df_corrected['Mean Reward'].apply(ast.literal_eval)

# Assign new categories based on the updated complexity ranges
def assign_complexity_category_updated(complexity):
    if complexity < 0.26:
        return '0-0.26'
    elif 0.26 <= complexity < 0.51:
        return '0.26-0.51'
    elif 0.51 <= complexity < 0.76:
        return '0.51-0.76'
    elif 0.76 <= complexity <= 1:
        return '0.76-1'

# Apply the updated function to categorize complexity levels
df_corrected['Complexity Category'] = df_corrected['Complexity_of_Environment'].apply(assign_complexity_category_updated)

# Calculate success and failure counts for each model
df_corrected['Success Count'] = df_corrected['Mean Reward'].apply(lambda rewards: sum(1 for reward in rewards if reward >= 20))
df_corrected['Failure Count'] = df_corrected['Mean Reward'].apply(lambda rewards: sum(1 for reward in rewards if reward < 20))

# Calculate success rate for each model
df_corrected['Success Rate'] = df_corrected['Success Count'] / (df_corrected['Success Count'] + df_corrected['Failure Count'])

# 1. Success Rate Trend by Complexity Category
avg_success_rate_by_category_corrected = df_corrected.groupby('Complexity Category')['Success Rate'].mean()

# Plot success rate trend by complexity category
plt.figure(figsize=(10, 6))
avg_success_rate_by_category_corrected.plot(marker='o', linestyle='-', color='dodgerblue')
plt.xlabel('Complexity Category')
plt.ylabel('Average Success Rate')
plt.title('Success Rate Trend by Complexity Category (Corrected)')
plt.grid(axis='y', linestyle='--')
plt.show()

# 2. Area Under Curve (AUC) for Success Rate by Complexity Category
category_labels_corrected = avg_success_rate_by_category_corrected.index.values
success_rates_category_corrected = avg_success_rate_by_category_corrected.values

# Numeric representation for updated categories
category_numeric_corrected = np.array([0.13, 0.385, 0.635, 0.88])

# Calculate the AUC for success rate as a function of complexity category (Corrected)
auc_value_category_corrected = auc(category_numeric_corrected, success_rates_category_corrected)

# Plot the success rate curve with AUC annotation for complexity categories (Corrected)
plt.figure(figsize=(10, 6))
plt.plot(category_numeric_corrected, success_rates_category_corrected, marker='o', linestyle='-', color='mediumseagreen')
plt.fill_between(category_numeric_corrected, success_rates_category_corrected, alpha=0.2, color='mediumseagreen')
plt.xlabel('Complexity Category (Numeric Representation)')
plt.xticks(category_numeric_corrected, category_labels_corrected)
plt.ylabel('Average Success Rate')
plt.title(f'Success Rate Curve with AUC for Complexity Categories (Corrected AUC = {auc_value_category_corrected:.2f})')
plt.grid(axis='y', linestyle='--')
plt.show()

# 3. Box Plot of Rewards by Complexity Category (Corrected)
reward_data_by_category_corrected = df_corrected.explode('Mean Reward').groupby('Complexity Category')['Mean Reward'].apply(list)

# Prepare data for box plot (Corrected)
box_plot_data_corrected = [reward_data_by_category_corrected[category] for category in reward_data_by_category_corrected.index]

# Plot box plot of rewards by complexity category (Corrected)
plt.figure(figsize=(10, 6))
plt.boxplot(box_plot_data_corrected, labels=reward_data_by_category_corrected.index, patch_artist=True, showfliers=False)
plt.xlabel('Complexity Category')
plt.ylabel('Rewards')
plt.title('Box Plot of Rewards by Complexity Category (Corrected)')
plt.grid(axis='y', linestyle='--')
plt.show()

# 4. Cumulative Reward Distribution (CDF) by Complexity Category (Corrected)
plt.figure(figsize=(10, 6))

# Plot the CDF for each complexity category (Corrected)
for category, group in df_corrected.groupby('Complexity Category'):
    all_rewards = [reward for rewards_list in group['Mean Reward'] for reward in rewards_list]
    all_rewards_sorted = np.sort(all_rewards)
    cdf = np.arange(1, len(all_rewards_sorted) + 1) / len(all_rewards_sorted)
    plt.plot(all_rewards_sorted, cdf, marker='.', linestyle='-', label=f'Category: {category}')

plt.xlabel('Reward')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution Function (CDF) of Rewards by Complexity Category (Corrected)')
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.show()


