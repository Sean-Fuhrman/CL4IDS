#%%
import pandas as pd
import os
import glob
import math
import matplotlib.pyplot as plt

strategy_years = {
    "EWC": 2016,
    "GEM": 2017,
    "ICaRL": 2016,
    "LwF": 2016,
    "MIR": 2019,
    "SI": 2017,
    "PackNet": 2017,
    "MAS": 2017,
}

def read_csv_from_logs(folder_name):
    """Reads CSV files and returns separate dictionaries for train and eval results.

    Args:
        folder_name: The name of the folder under the "logs/" directory.

    Returns:
        tuple: A tuple of two dictionaries: (train_results, eval_results).
    """

    train_results = {}
    eval_results = {}

    logs_path = "logs/" + folder_name  
    for subfolder in os.listdir(logs_path):
        subfolder_path = os.path.join(logs_path, subfolder)

        if os.path.isdir(subfolder_path):
            for file in glob.glob(os.path.join(subfolder_path, "*results.csv")):
                df = pd.read_csv(file)
                df['folder'] = subfolder

                if "train" in file:
                    train_results[subfolder] = df
                elif "eval" in file:
                    eval_results[subfolder] = df

    return train_results, eval_results 
def graph_average_accuracy_on_past_experiences(train_results, eval_results, name):
    """Graphs the average accuracy on past experiences for each strategy. """
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'H', 'd', '*', 'X', 'P', '8', '1', '2', '3', '4', 'x', '|', '_', '']

    # Average accuracy on past experiences
    average_accuracies = {}
    for strategy in eval_results:
        average_accuracies[strategy] = []
        max_experience = eval_results[strategy]['training_exp'].max() + 1
        if math.isnan(max_experience):
            print("No experiences found for", strategy)
            continue
        for exp in range(max_experience):
            #get the average accuracy on past experiences
            filtered_df = eval_results[strategy][
                (eval_results[strategy]['training_exp'] == exp) &
                (eval_results[strategy]['eval_exp'] <= exp)
            ]
            average_accuracies[strategy].append(filtered_df['eval_accuracy'].mean())

    # Plot the graph
    experiences = range(max_experience)
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed

    for strategy, accuracies in average_accuracies.items():
        if(len(accuracies) != max_experience):
            print("Skipping", strategy, "due to missing experiences")
            continue
        year = strategy_years[strategy] if strategy in strategy_years else ""
        plt.plot(experiences, accuracies, label=strategy + "-" + str(year), marker=markers.pop(0))

    plt.xlabel("Experiences")
    plt.ylabel("Accuracy")
    plt.title("Average Accuracy on seen Experiences for " + name)
    plt.legend()  # Display the legend
    plt.grid(True) 
    plt.show()

def graph_average_f1_on_past_experiences(train_results, eval_results, name):
    """Graphs the average accuracy on past experiences for each strategy. """
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'H', 'd', '*', 'X', 'P', '8', '1', '2', '3', '4', 'x', '|', '_', '']

    # Average accuracy on past experiences
    average_accuracies = {}
    for strategy in eval_results:
        average_accuracies[strategy] = []
        max_experience = eval_results[strategy]['training_exp'].max() + 1
        if math.isnan(max_experience):
            print("No experiences found for", strategy)
            continue
        for exp in range(max_experience):
            #get the average accuracy on past experiences
            filtered_df = eval_results[strategy][
                (eval_results[strategy]['training_exp'] == exp) &
                (eval_results[strategy]['eval_exp'] <= exp)
            ]
            average_accuracies[strategy].append(filtered_df['eval_f1'].mean())

    # Plot the graph
    experiences = range(max_experience)
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed

    for strategy, accuracies in average_accuracies.items():
        if(len(accuracies) != max_experience):
            print("Skipping", strategy, "due to missing experiences")
            continue
        year = strategy_years[strategy] if strategy in strategy_years else ""
        plt.plot(experiences, accuracies, label=strategy + "-" + str(year), marker=markers.pop(0))

    plt.xlabel("Experiences")
    plt.ylabel("F1")
    plt.title("Average F1 on seen Experiences for " + name)
    plt.legend()  # Display the legend
    plt.grid(True) 
    plt.show()

def graph_dataset(name):
    train_results, eval_results = read_csv_from_logs(name)

    graph_average_accuracy_on_past_experiences(train_results, eval_results, name)
    graph_average_f1_on_past_experiences(train_results, eval_results, name)


graph_dataset("X-IIoT")
graph_dataset("Edge-IIoT")

# graph_dataset("SMOTE-X-IIoT")
# graph_dataset("SMOTE-Edge-IIoT")
# %%

