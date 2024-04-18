#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Load the data
X_train = np.load('../Datasets/Edge-IIoT-pre-processed/Default Preprocessing/Multi/X_train.npy')
Y_train = np.load('../Datasets/Edge-IIoT-pre-processed/Default Preprocessing/Multi/Y_train.npy')
X_val = np.load('../Datasets/Edge-IIoT-pre-processed/Default Preprocessing/Multi/X_val.npy')
Y_val = np.load('../Datasets/Edge-IIoT-pre-processed/Default Preprocessing/Multi/Y_val.npy')
X_test = np.load('../Datasets/Edge-IIoT-pre-processed/Default Preprocessing/Multi/X_test.npy')
Y_test = np.load('../Datasets/Edge-IIoT-pre-processed/Default Preprocessing/Multi/Y_test.npy')

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train).float()
Y_train = torch.tensor(Y_train).float()
X_val = torch.tensor(X_val).float()
Y_val = torch.tensor(Y_val).float()
X_test = torch.tensor(X_test).float()
Y_test = torch.tensor(Y_test).float()

#Print the shape of the data
print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)
print(X_test.shape)
print(Y_test.shape)
#%%
# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(95, 90) #input size is 91
        self.fc2 = nn.Linear(90, 90)
        self.fc3 = nn.Linear(90, 90)
        self.fc4 = nn.Linear(90, 15) #output size is 2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)

model = Net()
#%%
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

# Train the model
EPOCHS = 25
batch_size = 800
for epoch in range(EPOCHS):
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        Y_batch = Y_train[i:i+batch_size]
        # print(X_batch.shape)
        # print(Y_batch[0])
        optimizer.zero_grad()
        output = model(X_batch)
        # print(output[0])
        loss = F.cross_entropy(output, Y_batch)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    # Validate the model
    with torch.no_grad():
        output = model(X_val)
        loss = F.cross_entropy(output, Y_val)
        print(f'Validation Loss: {loss.item()}')
        pred = torch.argmax(output, 1)
        print(pred[:10])
        Y_val_pred = torch.argmax(Y_val, 1)
        print(Y_val_pred[:10])
        accuracy = (pred == Y_val_pred).sum() / len(Y_val)
        print(f'Validation Accuracy: {accuracy}')
        
#%%
torch.save(model, 'DNN_multi_edgeIIoT.pth')
#%%
# load the model
model = torch.load('DNN_multi_edgeIIoT.pth')

# Test the model
predictions = []
with torch.no_grad():
    for i, data in enumerate(X_test):
        data = data[np.newaxis, :]
        output = model(data)
        predictions.append(torch.argmax(output).item())
#get the accuracy
Y_test_pred = torch.argmax(Y_test, 1)
accuracy = np.sum(Y_test_pred.numpy() == predictions) / len(Y_test)
print('Accuracy:', accuracy)

# Print the classification report
print(classification_report(Y_test_pred, predictions))

# Print the confusion matrix
print(confusion_matrix(Y_test_pred, predictions))

# Save the model
#%%

# Accuracy: 0.9313338657101339
#       precision    recall  f1-score   support

#    0       0.72      0.93      0.81      4830
#    1       0.94      0.60      0.74      9618
#    2       1.00      0.96      0.98     13519
#    3       0.70      1.00      0.83      9949
#    4       1.00      1.00      1.00     24506
#    5       0.14      0.82      0.24       199
#    6       0.97      1.00      0.99        69
#    7       1.00      1.00      1.00    272695
#    8       0.44      0.89      0.59      9988
#    9       0.00      0.00      0.00      4049
#   10       0.00      0.00      0.00      1972
#   11       0.89      0.17      0.28     10048
#   12       0.64      0.48      0.55      7251
#   13       0.94      0.85      0.89     10158
#   14       0.32      0.79      0.46      3084

#graph confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)  # Reset defaults

cm = np.array(confusion_matrix(Y_test_pred, predictions))
print(cm.shape)

num_to_label = {0: 'Backdoor', 1: 'DDoS_HTTP', 2: 'DDoS_ICMP', 3: 'DDoS_TCP', 4: 'DDoS_UDP', 5: 'Fingerprinting', 6: 'MITM', 7: 'Normal', 8: 'Password', 9: 'Port_Scanning', 10: 'Ransomware', 11: 'SQL_injection', 12: 'Uploading', 13: 'Vulnerability_scanner', 14: 'XSS'}
labels = [num_to_label[i] for i in range(15)]
#normalize the confusion matrix
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
#
#%%