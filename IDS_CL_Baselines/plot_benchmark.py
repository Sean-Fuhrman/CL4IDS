#%%
import utils
import torch
import seaborn as sns
import pandas as pd
# Plot Classes Per Experience
import matplotlib.pyplot as plt
import numpy as np
#%%

benchmark = utils.get_Edge_IIoT_benchmark(n_experiences=5)
labels = utils.EDGE_ATTACK_TO_LABEL

df = pd.DataFrame(columns=['Experience', 'Attack Num', 'Attack Type', 'Count', 'Percentage of Experience'])


for i, experience in enumerate(benchmark.train_stream):
    Y = experience.dataset.targets
    unique, counts = np.unique(Y, return_counts=True)
    total = sum(counts)
    for j, count in zip(unique, counts):
        df.loc[len(df.index)]= [i, j, labels[j], count, count/total]


sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Experience", y="Count", hue="Attack Type", data=df)

plt.title("Edge-IIoT: Attacks per Experience")
#Save df
df.to_csv("./logs/dataset-experiences/Edge-IIoT_Attacks_Per_Experience.csv")

#move legend outside of plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()



# %%
benchmark = utils.get_X_IIoT_benchmark(n_experiences=5)
labels = utils.X_ATTACK_TO_LABEL

df = pd.DataFrame(columns=['Experience', 'Attack Num', 'Attack Type', 'Count', 'Percentage of Experience'])


for i, experience in enumerate(benchmark.train_stream):
    Y = experience.dataset.targets
    unique, counts = np.unique(Y, return_counts=True)
    total = sum(counts)
    for j, count in zip(unique, counts):
        df.loc[len(df.index)]= [i, j, labels[j], count, count/total]


sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Experience", y="Count", hue="Attack Type", data=df)

plt.title("X-IIoT: Attacks per Experience")

#move legend outside of plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

#Save df
df.to_csv("./logs/dataset-experiences/X-IIoT_Attacks_Per_Experience.csv")
# %%

benchmark = utils.get_SMOTE_Edge_IIoT_benchmark(n_experiences=5)
labels = utils.EDGE_ATTACK_TO_LABEL

df = pd.DataFrame(columns=['Experience', 'Attack Num', 'Attack Type', 'Count', 'Percentage of Experience'])


for i, experience in enumerate(benchmark.train_stream):
    Y = experience.dataset.targets
    unique, counts = np.unique(Y, return_counts=True)
    total = sum(counts)
    for j, count in zip(unique, counts):
        df.loc[len(df.index)]= [i, j, labels[j], count, count/total]


sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Experience", y="Count", hue="Attack Type", data=df)

plt.title("SMOTE-Edge-IIoT: Attacks per Experience")
#Save df
df.to_csv("./logs/dataset-experiences/Edge-IIoT_Attacks_Per_Experience.csv")

#move legend outside of plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

#%%
benchmark = utils.get_SMOTE_X_IIoT_benchmark(n_experiences=5)
labels = utils.X_ATTACK_TO_LABEL

df = pd.DataFrame(columns=['Experience', 'Attack Num', 'Attack Type', 'Count', 'Percentage of Experience'])


for i, experience in enumerate(benchmark.train_stream):
    Y = experience.dataset.targets
    unique, counts = np.unique(Y, return_counts=True)
    total = sum(counts)
    for j, count in zip(unique, counts):
        df.loc[len(df.index)]= [i, j, labels[j], count, count/total]


sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Experience", y="Count", hue="Attack Type", data=df)

plt.title("SMOTE-Edge-IIoT: Attacks per Experience")
#Save df
df.to_csv("./logs/dataset-experiences/Edge-IIoT_Attacks_Per_Experience.csv")

#move legend outside of plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
#%%