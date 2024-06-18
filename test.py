import pandas as pd
import matplotlib.pyplot as plt
import os

from Pretraining import Pretraining25, Pretraining50, Pretraining75, Pretraining100
# Dictionary to hold the success rates for each category
pretraining_data = {
    "0-0.25": [],
    "0.25-0.5": [],
    "0.5-0.75": [],
    "0.75-1": []
}

# Iterate through the folders in data/trained
for folder in os.listdir('data/trained'):
    file_path = os.path.join('data/trained', folder, 'Logs', 'monitor.log.monitor.csv')

    if os.path.isfile(file_path):
        data = pd.read_csv(file_path, skiprows=1)

        # Initialize a list to hold the success rates
        success_rates = []

        # Create chunks based on cumulative time steps
        cumulative_time = 0
        chunk_counts = 0
        chunk_size = 0
        for index, row in data.iterrows():
            cumulative_time += row['l']
            chunk_size += 1
            if row['r'] + row['l'] > 20:
                chunk_counts += 1
            if cumulative_time >= 1000:
                success_rate = chunk_counts / chunk_size
                success_rates.append(success_rate)
                cumulative_time = 0
                chunk_counts = 0
                chunk_size = 0

        # Add the last chunk if it exists
        if chunk_size > 0:
            success_rate = chunk_counts / chunk_size
            success_rates.append(success_rate)

        # Determine the pretraining category
        if any(name in folder for name in Pretraining25):
            pretraining_data["0-0.25"].append(success_rates)
        elif any(name in folder for name in Pretraining50):
            pretraining_data["0.25-0.5"].append(success_rates)
        elif any(name in folder for name in Pretraining75):
            pretraining_data["0.5-0.75"].append(success_rates)
        elif any(name in folder for name in Pretraining100):
            pretraining_data["0.75-1"].append(success_rates)


# Function to calculate average success rates for each chunk
def average_success_rates(success_rates_list):
    max_length = max(len(success_rates) for success_rates in success_rates_list)
    averages = []
    for i in range(max_length):
        chunk_rates = [success_rates[i] for success_rates in success_rates_list if len(success_rates) > i]
        averages.append(sum(chunk_rates) / len(chunk_rates))
    return averages


# Plot the average success rates for each category
plt.figure(figsize=(15, 10))

for idx, category in enumerate(pretraining_data.keys()):
    plt.subplot(2, 2, idx + 1)
    if pretraining_data[category]:
        avg_success_rates = average_success_rates(pretraining_data[category])
        plt.plot(range(1, len(avg_success_rates) + 1), avg_success_rates, marker='o')
        plt.xlabel('Chunk Number (each chunk = 1000 time steps)')
        plt.ylabel('Average Success Rate')
        plt.title(f'Success Rates in {category} Complexity Category')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No Data Available', horizontalalignment='center', verticalalignment='center')
        plt.title(f'Success Rates in {category} Complexity Category')
        plt.xlabel('Chunk Number (each chunk = 1000 time steps)')
        plt.ylabel('Average Success Rate')

plt.tight_layout()
plt.show()
