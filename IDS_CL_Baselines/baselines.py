#%%
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from torch import nn
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, forward_transfer_metrics
from avalanche.models import SimpleMLP, MultiHeadClassifier, MTSimpleMLP
from avalanche.logging import CSVLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import ICaRL, GEM, LwF, EWC, SynapticIntelligence, MAS, Cumulative, Naive, MIR
from avalanche.training.supervised.strategy_wrappers import PackNet
from avalanche.models.packnet import PackNetModel
from avalanche.benchmarks.scenarios import split_online_stream

import os
import glob
from torchvision import transforms
from avalanche.models import TrainEvalModel

from custom_csv_logger import CustomCSVLogger
from custom_metrics import average_f1_metrics
import utils
import torch
import unit_tests

def init_benchmark(dataset):
    if dataset == "SplitMNIST":
        benchmark = SplitMNIST(n_experiences=5, return_task_id=True)
        input_size = 28 * 28
    elif dataset == "SplitCIFAR10":
        benchmark = SplitCIFAR10(
            5, 
            return_task_id=True,
            train_transform=transforms.ToTensor(),
            eval_transform=transforms.ToTensor()
        )
        input_size = 3 * 32 * 32
    elif dataset == "Edge-IIoT":
        benchmark = utils.get_Edge_IIoT_benchmark(n_experiences=5)
        input_size = 95
    elif dataset == "X-IIoT":
        benchmark = utils.get_X_IIoT_benchmark(n_experiences=5)
        input_size = 56
        # benchmark = utils.restrict_dataset_size(benchmark, 1000)
    else:
        raise Exception("Invalid dataset name")
    return benchmark, input_size

#Shared init eval function
def init_eval(name, dataset):
    loggers = []
    #remove any files in the current logs folder
    for file in glob.glob("./logs/"+dataset+"/" + name + "/*"):
        os.remove(file)

    loggers.append(CustomCSVLogger("./logs/"+dataset+"/" + name))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        average_f1_metrics(epoch=True, experience=True, stream=True),
        # loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        # forgetting_metrics(experience=True, stream=True),
        loggers=loggers
    )

    return eval_plugin
def test_strategy(name, epochs=1, dataset="SplitMNIST", device="cuda"):

    benchmark, input_size = init_benchmark(dataset)

    eval_plugin = init_eval(name, dataset)
   
    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = None 

    if name == "ICaRL": #Replay methods
        feature_size = 64 
        feature_extractor = SimpleMLP(input_size=input_size, num_classes=feature_size).to(device)
        classifier = nn.Linear(feature_size, benchmark.n_classes).to(device)

        buffer_transform = None
        cl_strategy = ICaRL(
            feature_extractor=feature_extractor, classifier=classifier,
            optimizer=Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=0.001),
            buffer_transform=buffer_transform,
            device=device,
            fixed_memory=True, memory_size=1000,
            evaluator=eval_plugin,eval_every=1,
            train_mb_size=500, train_epochs=epochs, eval_mb_size=100)
    elif name == "GEM":
        model = SimpleMLP(input_size=input_size, num_classes=benchmark.n_classes).to(device)
        cl_strategy = GEM(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), patterns_per_exp=500, device=device, memory_strength=0.5, evaluator=eval_plugin,
                          eval_every=1, train_mb_size=500, train_epochs=epochs, eval_mb_size=100)
    elif name == "LwF": #Regularization methods
        model = SimpleMLP(input_size=input_size, num_classes=benchmark.n_classes).to(device)
        cl_strategy = LwF(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), device=device, alpha=1, temperature=2,
                           evaluator=eval_plugin, eval_every=1, train_mb_size=500, train_epochs=epochs, eval_mb_size=100)
    elif name == "EBLL":
        pass
    elif name == "EWC":  #NEEDS MULTIHEAD FOR GOOD RESULTS
        model = MTSimpleMLP(input_size=input_size).to(device)
        cl_strategy = EWC(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), device=device, ewc_lambda=0.1, mode='separate',
                            evaluator=eval_plugin, eval_every=1, train_mb_size=256, train_epochs=epochs, eval_mb_size=128)
    elif name == "SI": # NEEDS MULTIHEAD FOR GOOD RESULTS
        model = MTSimpleMLP(input_size=input_size).to(device)
        cl_strategy = SynapticIntelligence(model, Adam(model.parameters(), lr=0.001), CrossEntropyLoss(), si_lambda=1, eps=0.1,device=device,
                                             evaluator=eval_plugin, eval_every=1, train_mb_size=500, train_epochs=epochs, eval_mb_size=100)
    elif name == "MAS":  # NEEDS MULTIHEAD FOR GOOD RESULTS
        model = MTSimpleMLP(input_size=input_size).to(device)
        cl_strategy = MAS(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), device=device,
                          evaluator=eval_plugin, eval_every=1, train_mb_size=500, train_epochs=epochs, eval_mb_size=100)    
    elif name == "mean-IMM":
        pass
    elif name == "mode-IMM":
        pass
    elif name == "MIR": #DOESNT WORK, WITH ANY DATASET (mbatch is None at end of experience)
        model = SimpleMLP(input_size=input_size, num_classes=benchmark.n_classes).to(device)
        # benchmark = utils.restrict_dataset_size(benchmark, 1000)
        # benchmark = split_online_stream(
        #     original_stream=benchmark.train_stream,
        #     experience_size=10,
        #     access_task_boundaries=False,
        # )
        cl_strategy = MIR(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), device=device,batch_size_mem=10, mem_size=500, subsample=50,evaluator=eval_plugin, eval_every=1, train_mb_size=10, train_epochs=epochs, eval_mb_size=64)
    elif name == "PackNet": #Parameter isolation methods
        model = SimpleMLP(input_size=input_size, num_classes=benchmark.n_classes).to(device) #needs very high prune proportion to work
        model = PackNetModel(model)
        cl_strategy = PackNet(model, SGD(model.parameters(), lr=0.001, momentum=0.9), 
                     post_prune_epochs=1, prune_proportion=0.95, device=device,
                     evaluator=eval_plugin, eval_every=1, train_mb_size=500, train_epochs=2, eval_mb_size=100)
    elif name == "HAT":
        pass
    elif name == "Cumulative": #Baseline methods
        model = SimpleMLP(input_size=input_size, num_classes=benchmark.n_classes).to(device)
        cl_strategy = Cumulative(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), device=device, evaluator=eval_plugin, eval_every=1, train_mb_size=500, train_epochs=epochs, eval_mb_size=100)
    elif name == "Naive":
        model = SimpleMLP(input_size=input_size, num_classes=benchmark.n_classes).to(device)
        cl_strategy = Naive(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), device=device, evaluator=eval_plugin, eval_every=-1, train_mb_size=500, train_epochs=epochs, eval_mb_size=100)
    elif name == "Naive-Multihead":
        model = MTSimpleMLP(input_size=input_size).to(device)
        cl_strategy = Naive(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), device=device, evaluator=eval_plugin, eval_every=1, train_mb_size=500, train_epochs=epochs, eval_mb_size=100)
    else:
        raise Exception("Invalid strategy name")
        

    # TRAINING LOOP
    if cl_strategy is not None:
        for i, experience in enumerate(benchmark.train_stream):
            # train returns a dictionary which contains all the metric values
            cl_strategy.train(experience, eval_streams=[benchmark.test_stream[i]])
            # test also returns a dictionary whi ch contains all the metric values
            cl_strategy.eval(benchmark.test_stream)
            
            # unit_tests.test_f1_scrore(cl_strategy, benchmark.test_stream[i])
            


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    strategies = ["ICaRL", "GEM", "LwF", "EWC", "SI", "MAS", "PackNet", "Cumulative", "Naive", "Naive-Multihead"]
    for strategy in strategies:
        test_strategy(strategy, 1, "X-IIoT",device)
    # test_strategy("Naive", 1, "Edge-IIoT",device)

#%%