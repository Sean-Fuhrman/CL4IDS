#%%
import pandas as pd

#load in classification report

classification_report = pd.read_csv('./results/'+'DNN_classification_report.csv')
n_experiences = 5
n_classes = 19
EDGE_NORMAL_ATTACK = 7
print(classification_report)
#%%

#get class order of experiences
class_order = [ [] for _ in range(n_experiences) ] 
for i in range(19):
    if i == EDGE_NORMAL_ATTACK:
        continue
    class_order[i % n_experiences].append(i)
#add normal attack to every experience
class_order = [[EDGE_NORMAL_ATTACK] + c for c in class_order]
# get the average macro f1 score for each experience
average_f1 = {}

for n,i in enumerate(class_order):
    f1 = 0
    for c in i:
        f1 += classification_report.loc[c]['f1-score']
    average_f1[n] = (f1/len(i))

print(average_f1)

# %%
