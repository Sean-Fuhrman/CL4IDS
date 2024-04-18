#%%
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from torch import nn
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, forward_transfer_metrics
from avalanche.models import SimpleMLP, MultiHeadClassifier, MTSimpleMLP
from avalanche.logging import CSVLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import ICaRL, GEM, LwF, EWC, SynapticIntelligence, MAS, Cumulative, Naive
from avalanche.training.supervised.strategy_wrappers import PackNet
from avalanche.models.packnet import PackNetModel
import os
import glob
from torchvision import transforms
from avalanche.models import TrainEvalModel


benchmark = SplitMNIST(n_experiences=5, return_task_id=True)

#Shared init eval function
def init_eval(name):
    loggers = []
    #remove any files in the current logs folder
    for file in glob.glob("./logs/SplitMNIST/" + name + "/*"):
        os.remove(file)

    loggers.append(CSVLogger("./logs/SplitMNIST/" + name))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        # forward_transfer_metrics(experience=True, stream=True),
        loggers=loggers
    )

    return eval_plugin


def test_strategy(name, epochs=1):

    eval_plugin = init_eval(name)
   
    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = None 

    if name == "ICaRL": #Replay methods
        feature_size = 64 
        feature_extractor = SimpleMLP(num_classes=feature_size)
        classifier = nn.Linear(feature_size, benchmark.n_classes)

        buffer_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor()
        ])
        cl_strategy = ICaRL(
            feature_extractor=feature_extractor, classifier=classifier,
            optimizer=Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=0.001),
            buffer_transform=buffer_transform,
            fixed_memory=True, memory_size=1000,
            evaluator=eval_plugin,eval_every=1,
            train_mb_size=500, train_epochs=epochs, eval_mb_size=100)
    elif name == "GEM":
        model = SimpleMLP(num_classes=benchmark.n_classes)
        cl_strategy = GEM(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), patterns_per_exp=500, memory_strength=0.5, evaluator=eval_plugin,
                          eval_every=1, train_mb_size=500, train_epochs=epochs, eval_mb_size=100)
    elif name == "LwF": #Regularization methods
        model = SimpleMLP(num_classes=benchmark.n_classes)
        cl_strategy = LwF(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), alpha=1, temperature=2,
                           evaluator=eval_plugin, eval_every=1, train_mb_size=500, train_epochs=epochs, eval_mb_size=100)
    elif name == "EBLL":
        pass
    elif name == "EWC":  #NEEDS MULTIHEAD FOR GOOD RESULTS
        model = SimpleMLP(num_classes=benchmark.n_classes)
        cl_strategy = EWC(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), ewc_lambda=0.1, mode='separate',
                            evaluator=eval_plugin, eval_every=1, train_mb_size=256, train_epochs=epochs, eval_mb_size=128)
    elif name == "SI": # NEEDS MULTIHEAD FOR GOOD RESULTS
        model = SimpleMLP(num_classes=benchmark.n_classes)
        cl_strategy = SynapticIntelligence(model, Adam(model.parameters(), lr=0.001), CrossEntropyLoss(), si_lambda=1, eps=0.1,
                                             evaluator=eval_plugin, eval_every=1, train_mb_size=500, train_epochs=epochs, eval_mb_size=100)
    elif name == "MAS":  # NEEDS MULTIHEAD FOR GOOD RESULTS
        model = SimpleMLP(num_classes=benchmark.n_classes)
        cl_strategy = MAS(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), 
                          evaluator=eval_plugin, eval_every=1, train_mb_size=500, train_epochs=epochs, eval_mb_size=100)    
    elif name == "mean-IMM":
        pass
    elif name == "mode-IMM":
        pass
    elif name == "PackNet": #Parameter isolation methods
        model = SimpleMLP(num_classes=benchmark.n_classes) #needs very high prune proportion to work
        model = PackNetModel(model)
        cl_strategy = PackNet(model, SGD(model.parameters(), lr=0.001, momentum=0.9), 
                     post_prune_epochs=1, prune_proportion=0.95,
                     evaluator=eval_plugin, eval_every=1, train_mb_size=500, train_epochs=2, eval_mb_size=100)
    elif name == "HAT":
        pass
    elif name == "Cumulative": #Baseline methods
        model = SimpleMLP(input_size=3072, num_classes=benchmark.n_classes)
        cl_strategy = Cumulative(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), evaluator=eval_plugin, eval_every=1, train_mb_size=500, train_epochs=epochs, eval_mb_size=100)
    elif name == "Naive":
        model = SimpleMLP(num_classes=benchmark.n_classes)
        cl_strategy = Naive(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), evaluator=eval_plugin, eval_every=1, train_mb_size=500, train_epochs=epochs, eval_mb_size=100)
    elif name == "Naive-Multihead":
        model = MTSimpleMLP()
        cl_strategy = Naive(model, SGD(model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), evaluator=eval_plugin, eval_every=1, train_mb_size=500, train_epochs=epochs, eval_mb_size=100)
    else:
        raise Exception("Invalid strategy name")
        

    # TRAINING LOOP
    for experience in benchmark.train_stream:
        # train returns a dictionary which contains all the metric values
        cl_strategy.train(experience)
        # test also returns a dictionary which contains all the metric values
        cl_strategy.eval(benchmark.test_stream)


if __name__ == "__main__":
    # strategies = ["ICaRL", "GEM", "LwF", "EBLL", "EWC", "SI", "MAS", "mean-IMM", "mode-IMM", "PackNet", "HAT", "Cumulative", "Naive"]
    # for strategy in strategies:
        # test_strategy(strategy)
    test_strategy("PackNet", 1)

# %%
