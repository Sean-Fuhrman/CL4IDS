import torch

def test_f1_scrore(cl_strategy, test_exp):
    model = cl_strategy.model
    X = torch.stack([x for x, _, _ in test_exp.dataset], dim=0)
    Y = torch.tensor([test_exp.dataset.targets])

    Y_pred = torch.argmax(model(X), dim=1)
    f1s = []
    for i in Y.unique():
        true_i = Y == i
        pred_i = Y_pred == i
        true_positives = torch.sum(true_i & pred_i)
        false_positives = torch.sum(~true_i & pred_i)
        false_negatives = torch.sum(true_i & ~pred_i)
        f1_score = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
        print("F1 Score for class ", i, ": ", f1_score)
        f1s.append(f1_score)
    print("Mean F1 Score: ", sum(f1s)/len(f1s))