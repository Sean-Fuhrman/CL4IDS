#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load the data
X = np.load('../../Datasets/Edge-IIoT-pre-processed/X.npy')
Y = np.load('../../Datasets/Edge-IIoT-pre-processed/Y.npy')

# Split the data    
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

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
class SimpleMLP(nn.Module):
    """
    Multi-Layer Perceptron with custom parameters.
    It can be configured to have multiple layers and dropout.

    **Example**::

        >>> from avalanche.models import SimpleMLP
        >>> n_classes = 10 # e.g. MNIST
        >>> model = SimpleMLP(num_classes=n_classes)
        >>> print(model) # View model details
    """

    def __init__(
        self,
        num_classes=10,
        input_size=28 * 28,
        hidden_size=512,
        hidden_layers=1,
        drop_rate=0.5,
    ):
        """
        :param num_classes: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        """
        super().__init__()

        layers = nn.Sequential(
            *(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_rate),
            )
        )
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}",
                nn.Sequential(
                    *(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=drop_rate),
                    )
                ),
            )

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x

model = SimpleMLP(input_size=95, num_classes=15)
#%%
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
EPOCHS = 25
batch_size = 512
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
        Y_val_pred = torch.argmax(Y_val, 1)
        accuracy = (pred == Y_val_pred).sum() / len(Y_val)
        print(f'Validation Accuracy: {accuracy}')
        
#%%
torch.save(model, './models/DNN_Edge_IIoT.pth')
#%%
# load the model
model = torch.load('./models/DNN_Edge_IIoT.pth')

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
import pandas as pd
# save classification report
report = classification_report(Y_test_pred, predictions, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv('./results/'+'DNN_classification_report.csv')

# Save the model
#%%


#graph confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)  # Reset defaults

cm = np.array(confusion_matrix(Y_test_pred, predictions))
print(cm.shape)

num_to_label = {0: 'BruteForce', 1: 'C&C', 2: 'Dictionary', 3: 'Discovering_resources', 4: 'Exfiltration', 5: 'Fake_notification', 6: 'False_data_injection', 7: 'Generic_scanning', 8: 'MQTT_cloud_broker_subscription', 9: 'MitM', 10: 'Modbus_register_reading', 11: 'Normal', 12: 'RDOS', 13: 'Reverse_shell', 14: 'Scanning_vulnerability', 15: 'TCP Relay', 16: 'crypto-ransomware', 17: 'fuzzing', 18: 'insider_malcious'}
labels = [num_to_label[i] for i in range(15)]
#normalize the confusion matrix
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
#