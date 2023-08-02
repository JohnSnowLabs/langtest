import numpy as np
import evaluate
from typing import List, Callable


def compute_ner_metrics(label_list: List[str]) -> Callable:
    """Compute various metrics for token classification tasks

    Args:
        label_list (List[str]): list of available classes

    Returns:
        Callable: function doing the actual computation
    """

    def compute(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, token_l) in zip(prediction, label) if token_l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [
                label_list[token_l]
                for (p, token_l) in zip(prediction, label)
                if token_l != -100
            ]
            for prediction, label in zip(predictions, labels)
        ]
        seqeval = evaluate.load("seqeval")
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute
