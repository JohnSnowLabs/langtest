from collections import Counter
from typing import List, Union, Dict
import numpy as np


def classification_report(
    y_true: List[Union[str, int]], y_pred: List[Union[str, int]], zero_division: int = 0
) -> Dict[str, Dict[str, Union[float, int]]]:
    """Generate a classification report including precision, recall, f1-score, and support.

    Args:
        y_true (List[Union[str, int]]): List of true labels.
        y_pred (List[Union[str, int]]): List of predicted labels.
        zero_division (int, optional): Specifies the value to return when there is a zero division case, i.e., when all predictions and true values are negative. Defaults to 0.

    Returns:
        Dict[str, Dict[str, Union[float, int]]]: Classification report, each class has a dictionary containing precision, recall, f1-score, and support.
    """
    # Count total true labels for each class (support)
    support = Counter(y_true)

    # Count correct predictions for precision and recall
    correct_predictions = Counter(
        [pred for true, pred in zip(y_true, y_pred) if true == pred]
    )
    predicted_labels = Counter(y_pred)

    # Initialize data structure for the report
    report = {}

    # Compute stats for each class
    for class_label in set(y_true).union(set(y_pred)):
        # Precision is the ratio of correct predictions to total predictions for each class
        if predicted_labels[class_label] > 0:
            precision = correct_predictions[class_label] / predicted_labels[class_label]
        else:
            precision = zero_division

        # Recall is the ratio of correct predictions to total true instances of the class
        if support[class_label] > 0:
            recall = correct_predictions[class_label] / support[class_label]
        else:
            recall = zero_division

        # F1 score is the harmonic mean of precision and recall
        if (precision + recall) > 0:
            f1_score = (2 * precision * recall) / (precision + recall)
        else:
            f1_score = zero_division

        # Add stats to the report
        report[class_label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,
            "support": support[class_label],
        }

    # Compute macro averages
    avg_precision = sum([metrics["precision"] for metrics in report.values()]) / len(
        report
    )
    avg_recall = sum([metrics["recall"] for metrics in report.values()]) / len(report)
    avg_f1_score = sum([metrics["f1-score"] for metrics in report.values()]) / len(report)

    report["macro avg"] = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1-score": avg_f1_score,
        "support": sum(support.values()),
    }

    return report


def calculate_f1_score(
    y_true: List[Union[str, int]],
    y_pred: List[Union[str, int]],
    average: str = "macro",
    zero_division: int = 0,
) -> float:
    """Calculate the F1 score for the provided true and predicted labels.

    Args:
        y_true (List[Union[str, int]]): List of true labels.
        y_pred (List[Union[str, int]]): List of predicted labels.
        average (str, optional): Method to calculate F1 score, can be 'micro', 'macro' or 'weighted'. Defaults to 'macro'.
        zero_division (int, optional): Value to return when there is a zero division case, i.e., when all predictions and true values are negative. Defaults to 0.

    Returns:
        float: Calculated F1 score.

    Raises:
        AssertionError: If lengths of y_true and y_pred are not equal.
        ValueError: If invalid averaging method is provided.
    """
    assert len(y_true) == len(y_pred), "Lengths of y_true and y_pred must be equal."

    unique_labels = set(y_true + y_pred)
    num_classes = len(unique_labels)

    if average == "micro":
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for true_label, predicted_label in zip(y_true, y_pred):
            if true_label == predicted_label:
                true_positives += 1
            else:
                false_negatives += 1
                false_positives += 1

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else zero_division
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else zero_division
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else zero_division
        )

    elif average == "macro" or average == "weighted":
        f1_score = 0.0

        for label in unique_labels:
            true_positives = 0
            false_positives = 0
            false_negatives = 0

            for true_label, predicted_label in zip(y_true, y_pred):
                if true_label == label and predicted_label == label:
                    true_positives += 1
                elif true_label == label and predicted_label != label:
                    false_negatives += 1
                elif true_label != label and predicted_label == label:
                    false_positives += 1

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else zero_division
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else zero_division
            )

            if precision + recall == 0:
                class_f1_score = 0.0
            else:
                class_f1_score = 2 * (precision * recall) / (precision + recall)

            if average == "macro":
                f1_score += class_f1_score / num_classes
            else:  # average == 'weighted'
                class_weight = sum(
                    1 for true_label in y_true if true_label == label
                ) / len(y_true)
                f1_score += class_weight * class_f1_score

    else:
        raise ValueError(
            "Invalid averaging method. Must be one of 'macro', 'micro', or 'weighted'."
        )
    return f1_score


def cosine_similarity(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """Compute the cosine similarity between two arrays.

    Args:
        array1 (numpy.ndarray): The first input array. This is a two-dimensional array, where each row is a vector.
        array2 (numpy.ndarray): The second input array. This should have the same shape as array1.

    Returns:
        numpy.ndarray: An array of cosine similarity values. Each value corresponds to the cosine similarity between a pair of vectors from array1 and array2.
    """
    dot_products = np.einsum("ij,ij->i", array1, array2)
    magnitudes1 = np.linalg.norm(array1, axis=1)
    magnitudes2 = np.linalg.norm(array2, axis=1)
    cosine_similarities = dot_products / (magnitudes1 * magnitudes2)

    return cosine_similarities
