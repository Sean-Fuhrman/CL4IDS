#%%

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from avalanche.benchmarks.utils import _make_taskaware_tensor_classification_dataset
from sklearn.model_selection import train_test_split
from avalanche.benchmarks import nc_benchmark, NCScenario, tensors_benchmark
from avalanche.benchmarks import dataset_benchmark
from imblearn.over_sampling import SMOTE


EDGE_NORMAL_ATTACK = 7
EDGE_ATTACK_TO_LABEL = {0: 'Backdoor', 1: 'DDoS_HTTP', 2: 'DDoS_ICMP', 3: 'DDoS_TCP', 4: 'DDoS_UDP', 5: 'Fingerprinting', 6: 'MITM', 7: 'Normal', 8: 'Password', 9: 'Port_Scanning', 10: 'Ransomware', 11: 'SQL_injection', 12: 'Uploading', 13: 'Vulnerability_scanner', 14: 'XSS'}

X_NORMAL_ATTACK = 11
X_ATTACK_TO_LABEL = {0: 'BruteForce', 1: 'C&C', 2: 'Dictionary', 3: 'Discovering_resources', 4: 'Exfiltration', 5: 'Fake_notification', 6: 'False_data_injection', 7: 'Generic_scanning', 8: 'MQTT_cloud_broker_subscription', 9: 'MitM', 10: 'Modbus_register_reading', 11: 'Normal', 12: 'RDOS', 13: 'Reverse_shell', 14: 'Scanning_vulnerability', 15: 'TCP Relay', 16: 'crypto-ransomware', 17: 'fuzzing', 18: 'insider_malcious'}

def create_split_experiences(X, Y, class_order, n_experiences, name="Edge-IIoT"):
    if name == "Edge-IIoT":
        normal_mask = torch.isin(Y, EDGE_NORMAL_ATTACK)
    elif name == "X-IIoT":
        normal_mask = torch.isin(Y, X_NORMAL_ATTACK)
    X_normal = X[normal_mask]
    Y_normal = Y[normal_mask]
    shuffle_idx = torch.randperm(X_normal.size(0))
    X_normal = X_normal[shuffle_idx]
    Y_normal = Y_normal[shuffle_idx]
    X_normal = torch.chunk(X_normal, n_experiences)
    Y_normal = torch.chunk(Y_normal, n_experiences)

    experiences = []
    for classes, X_n, Y_n in zip(class_order, X_normal, Y_normal):
        mask = np.isin(Y, classes)
        X_exp = X[mask]
        Y_exp = Y[mask]
        X_exp = torch.cat((X_exp, X_n))
        Y_exp = torch.cat((Y_exp, Y_n))
        shuffle_idx = torch.randperm(X_exp.size(0))
        X_exp = X_exp[shuffle_idx]
        Y_exp = Y_exp[shuffle_idx]
        experiences.append((X_exp, Y_exp))
    return experiences
def add_SMOTE(experiences):
    smote = SMOTE()
    for i, (X, Y) in enumerate(experiences):
        X = X.numpy()
        Y = Y.numpy()
        X, Y = smote.fit_resample(X, Y)
        experiences[i] = (torch.tensor(X), torch.tensor(Y))
    return experiences

def get_Edge_IIoT_benchmark(n_experiences=5, class_order=None, device='cpu'):
    # Load the dataset
    X = np.load('../Datasets/Edge-IIoT-pre-processed/X.npy')
    Y = np.load('../Datasets/Edge-IIoT-pre-processed/Y.npy')
    
    # Convert to tensors
    X = torch.tensor(X)
    Y = torch.argmax(torch.tensor(Y), dim=1)

    if class_order is None:
        class_order = [ [] for _ in range(n_experiences) ] 
        for i in range(15):
            if i == EDGE_NORMAL_ATTACK:
                continue
            class_order[i % n_experiences].append(i)
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_experiences = create_split_experiences(X_train, Y_train, class_order, n_experiences, name="Edge-IIoT")
    test_experiences = create_split_experiences(X_test, Y_test, class_order, n_experiences, name="Edge-IIoT")
  
    benchmark = tensors_benchmark(
        train_tensors=train_experiences,
        test_tensors=test_experiences,
        task_labels=list(range(len(train_experiences)))
    )
    benchmark.n_classes = 15
    class_order[0] = [EDGE_NORMAL_ATTACK] + class_order[0]
    benchmark.n_classes_per_exp = torch.tensor([len(classes) for classes in class_order])
    benchmark.classes_order =torch.tensor([c for classes in class_order for c in classes])
    return benchmark


def get_X_IIoT_benchmark(n_experiences=5, class_order=None, device="cpu"):
    # Load the dataset
    X = np.load('../Datasets/X-IIoT-pre-processed/X.npy')
    Y = np.load('../Datasets/X-IIoT-pre-processed/Y_1.npy')
    
    # Convert to tensors
    X = torch.tensor(X).float()
    Y = torch.tensor(Y).int()

    if class_order is None:
        class_order = [ [] for _ in range(n_experiences) ] 
        for i in range(19):
            if i == X_NORMAL_ATTACK:
                continue
            class_order[i % n_experiences].append(i)
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_experiences = create_split_experiences(X_train, Y_train, class_order, n_experiences, name="X-IIoT")
    test_experiences = create_split_experiences(X_test, Y_test, class_order, n_experiences, name="X-IIoT")
        
    benchmark = tensors_benchmark(
        train_tensors=train_experiences,
        test_tensors=test_experiences,
        task_labels=list(range(len(train_experiences)))
    )
    benchmark.n_classes = 19
    class_order[0] = [X_NORMAL_ATTACK] + class_order[0]
    benchmark.n_classes_per_exp = torch.tensor([len(classes) for classes in class_order])
    benchmark.classes_order =torch.tensor([c for classes in class_order for c in classes])
    return benchmark

def get_SMOTE_Edge_IIoT_benchmark(n_experiences=5, class_order=None, device='cpu'):
     # Load the dataset
    X = np.load('../Datasets/Edge-IIoT-pre-processed/X.npy')
    Y = np.load('../Datasets/Edge-IIoT-pre-processed/Y.npy')
    
    # Convert to tensors
    X = torch.tensor(X)
    Y = torch.argmax(torch.tensor(Y), dim=1)

    if class_order is None:
        class_order = [ [] for _ in range(n_experiences) ] 
        for i in range(15):
            if i == EDGE_NORMAL_ATTACK:
                continue
            class_order[i % n_experiences].append(i)
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_experiences = create_split_experiences(X_train, Y_train, class_order, n_experiences, name="Edge-IIoT")
    test_experiences = create_split_experiences(X_test, Y_test, class_order, n_experiences, name="Edge-IIoT")
  
    train_experiences = add_SMOTE(train_experiences)

    benchmark = tensors_benchmark(
        train_tensors=train_experiences,
        test_tensors=test_experiences,
        task_labels=list(range(len(train_experiences)))
    )
    benchmark.n_classes = 15
    class_order[0] = [EDGE_NORMAL_ATTACK] + class_order[0]
    benchmark.n_classes_per_exp = torch.tensor([len(classes) for classes in class_order])
    benchmark.classes_order =torch.tensor([c for classes in class_order for c in classes])
    return benchmark

def get_SMOTE_X_IIoT_benchmark(n_experiences=5, class_order=None, device='cpu'):
        # Load the dataset
        X = np.load('../Datasets/X-IIoT-pre-processed/X.npy')
        Y = np.load('../Datasets/X-IIoT-pre-processed/Y_1.npy')
        
        # Convert to tensors
        X = torch.tensor(X).float()
        Y = torch.tensor(Y).int()
    
        if class_order is None:
            class_order = [ [] for _ in range(n_experiences) ] 
            for i in range(19):
                if i == X_NORMAL_ATTACK:
                    continue
                class_order[i % n_experiences].append(i)
        # Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        train_experiences = create_split_experiences(X_train, Y_train, class_order, n_experiences, name="X-IIoT")
        test_experiences = create_split_experiences(X_test, Y_test, class_order, n_experiences, name="X-IIoT")

        train_experiences = add_SMOTE(train_experiences)

        benchmark = tensors_benchmark(
            train_tensors=train_experiences,
            test_tensors=test_experiences,
            task_labels=list(range(len(train_experiences)))
        )
        benchmark.n_classes = 19
        class_order[0] = [X_NORMAL_ATTACK] + class_order[0]
        benchmark.n_classes_per_exp = torch.tensor([len(classes) for classes in class_order])
        benchmark.classes_order =torch.tensor([c for classes in class_order for c in classes])
        return benchmark

def restrict_dataset_size(scenario, size: int):
    """
    Util used to restrict the size of the datasets coming from a scenario
    param: size: size of the reduced training dataset
    """
    modified_train_ds = []
    modified_test_ds = []
    modified_valid_ds = []
    
    n_classes = scenario.n_classes
    n_classes_per_exp = scenario.n_classes_per_exp
    classes_order = scenario.classes_order

    if hasattr(scenario, "valid_stream"):
        valid_list = list(scenario.valid_stream)

    for i, train_ds in enumerate(scenario.train_stream):
        train_ds_idx, _ = torch.utils.data.random_split(
            torch.arange(len(train_ds.dataset)),
            (size, len(train_ds.dataset) - size),
        )
        dataset = train_ds.dataset.subset(train_ds_idx)

        modified_train_ds.append(dataset)
        modified_test_ds.append(scenario.test_stream[i].dataset)
        if hasattr(scenario, "valid_stream"):
            modified_valid_ds.append(valid_list[i].dataset)

    scenario = dataset_benchmark(
        modified_train_ds,
        modified_test_ds,
        other_streams_datasets={"valid": modified_valid_ds}
        if len(modified_valid_ds) > 0
        else None,
    )
    scenario.n_classes = n_classes
    scenario.n_classes_per_exp = n_classes_per_exp
    scenario.classes_order = classes_order
    
    return scenario
# %%
