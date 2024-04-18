#%%
import pandas as pd
import os
import glob
import math
import matplotlib.pyplot as plt

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

def graph_average_accuracy_on_past_experiences():
    """Graphs the average accuracy on past experiences for each strategy. """
    
    # Average accuracy on past experiences
    average_accuracies = {}
    for strategy in eval_results:
        if strategy not in ["Cumulative", "Naive", "ICaRL", "PackNet", "Naive"]:
            continue
        average_accuracies[strategy] = []
        max_experience = eval_results[strategy]['training_exp'].max()
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
    experiences = range(1, max_experience + 1)
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed

    for strategy, accuracies in average_accuracies.items():
        if(len(accuracies) != max_experience):
            print("Skipping", strategy, "due to missing experiences")
            continue
        plt.plot(experiences, accuracies, label=strategy)

    plt.xlabel("Experiences")
    plt.ylabel("Accuracy")
    plt.title("Average Accuracy on seen Experiences")
    plt.legend()  # Display the legend
    plt.grid(True) 
    plt.show()

folder_to_process = "SplitMNIST"
train_results, eval_results = read_csv_from_logs(folder_to_process)

graph_average_accuracy_on_past_experiences()

# folder_to_process = "SplitCIFAR10"
# train_results, eval_results = read_csv_from_logs(folder_to_process)


# graph_average_accuracy_on_past_experiences()

# %%
