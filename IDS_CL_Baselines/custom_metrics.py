from typing import List, Optional, Union, Dict

import torch
from torch import Tensor
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from collections import defaultdict

class AverageF1Score(Metric[float]):
    """Accuracy metric. This is a standalone metric.

    The update method computes the accuracy incrementally
    by keeping a running average of the <prediction, target> pairs
    of Tensors provided over time.

    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average accuracy
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an accuracy value of 0.
    """

    def __init__(self):
        """Creates an instance of the standalone Accuracy metric.

        By default this metric in its initial state will return an accuracy
        value of 0. The metric can be updated by using the `update` method
        while the running accuracy can be retrieved using the `result` method.
        """
        self._class_f1_scores = defaultdict(Mean)
        self._mean_f1_score = Mean()
        """The mean utility that will be used to store the running accuracy."""

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
    ) -> None:
        """Update the running accuracy given the true and predicted labels.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.

        :return: None.
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]
        
        for i in true_y.unique():
            true_i = true_y == i
            pred_i = predicted_y == i
            true_positives = torch.sum(true_i & pred_i)
            false_positives = torch.sum(~true_i & pred_i)
            false_negatives = torch.sum(true_i & ~pred_i)
            f1_score = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
            self._class_f1_scores[i.item()].update(f1_score)

        self._mean_f1_score.update(sum([f1.result() for f1 in self._class_f1_scores.values()]) / len(self._class_f1_scores))
    def result(self) -> float:
        """Retrieves the running accuracy.

        Calling this method will not change the internal state of the metric.

        :return: The current running accuracy, which is a float value
            between 0 and 1.
        """
     
        return self._mean_f1_score.result()

    def reset(self) -> None:
        """Resets the metric.

        :return: None.
        """
        self._mean_f1_score.reset()
        self._class_f1_scores = defaultdict(Mean)

class AverageF1ScorePluginMetric(GenericPluginMetric[float, AverageF1Score]):
    """
    Base class for all accuracies plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """Creates the Accuracy plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        super().__init__(AverageF1Score(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)

class MinibatchAverageF1Score(AverageF1ScorePluginMetric):
    """
    The minibatch plugin accuracy metric.
    This metric only works at training time.

    This metric computes the average accuracy over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochAccuracy` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchAccuracy metric.
        """
        super(MinibatchAverageF1Score, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "F1_MB"


class EpochAverageF1Score(AverageF1ScorePluginMetric):
    """
    The average accuracy over a single training epoch.
    This plugin metric only works at training time.

    The accuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochAccuracy metric.
        """

        super(EpochAverageF1Score, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "F1_Epoch"


class RunningEpochAverageF1Score(AverageF1ScorePluginMetric):
    """
    The average accuracy across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the accuracy averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochAccuracy metric.
        """

        super(RunningEpochAverageF1Score, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "F1_Epoch"


class ExperienceAverageF1Score(AverageF1ScorePluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceAverageF1Score, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "F1_Exp"


class StreamAverageF1Scroe(AverageF1ScorePluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average accuracy over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamAccuracy metric
        """
        super(StreamAverageF1Scroe, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "F1_Stream"
    
def average_f1_metrics(
    *,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
) -> List[AverageF1ScorePluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch accuracy at training time.
    :param epoch: If True, will return a metric able to log
        the epoch accuracy at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch accuracy at training time.
    :param experience: If True, will return a metric able to log
        the accuracy on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the accuracy averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation accuracy only for experiences that the
        model has been trained on

    :return: A list of plugin metrics.
    """

    metrics: List[AverageF1ScorePluginMetric] = []
    if minibatch:
        metrics.append(MinibatchAverageF1Score())

    if epoch:
        metrics.append(EpochAverageF1Score())

    if epoch_running:
        metrics.append(RunningEpochAverageF1Score())

    if experience:
        metrics.append(ExperienceAverageF1Score())

    if stream:
        metrics.append(StreamAverageF1Scroe())
    return metrics
