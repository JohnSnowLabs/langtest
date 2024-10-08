from collections import Counter
from typing import List, Set, Union, Dict

from langtest.logger import logger
from ..errors import Errors
import pandas as pd


def classification_report(
    y_true: List[Union[str, int]],
    y_pred: List[Union[str, int]],
    zero_division: int = 0,
    multi_label: bool = False,
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

    if isinstance(y_true, list) and isinstance(y_pred, list):
        unique_labels = set(y_true + y_pred)
    elif isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        unique_labels = set(y_true.tolist() + y_pred.tolist())
    else:
        raise ValueError(
            "y_true and y_pred must be of the same type. Supported types are list and pandas Series."
        )

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
        raise ValueError(Errors.E074)
    return f1_score


def simple_multilabel_binarizer(y_true, y_pred):
    """
    A simple implementation of a multilabel binarizer for y_true and y_pred.

    Args:
        y_true (list of lists or sets): Actual labels for the data.
        y_pred (list of lists or sets): Predicted labels for the data.

    Returns:
        binarized_y_true (list of lists): Binary matrix of true labels.
        binarized_y_pred (list of lists): Binary matrix of predicted labels.
        classes (list): List of all unique classes (labels).
    """
    # Ensure we collect unique classes from both y_true and y_pred
    classes = sorted(set(label for labels in y_true + y_pred for label in labels))

    # Create a binary matrix for y_true and y_pred
    y_true_bin = [[1 if label in labels else 0 for label in classes] for labels in y_true]
    y_pred_bin = [[1 if label in labels else 0 for label in classes] for labels in y_pred]

    # Return the binarized labels and the consistent set of classes
    return y_true_bin, y_pred_bin, classes


def classification_report_multi_label(
    y_true: List[Set[Union[str, int]]],
    y_pred: List[Set[Union[str, int]]],
    zero_division: int = 0,
) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Generate a classification report for multi-label classification.

    Args:
        y_true (List[Set[Union[str, int]]]): List of sets of true labels.
        y_pred (List[Set[Union[str, int]]]): List of sets of predicted labels.
        zero_division (int, optional): Specifies the value to return when there is a zero division case. Defaults to 0.

    Returns:
        Dict[str, Dict[str, Union[float, int]]]: Classification report.
    """
    # Binarize the multi-label data
    y_true_bin, y_pred_bin, classes = simple_multilabel_binarizer(y_true, y_pred)

    # Initialize data structure for the report
    report = {}
    for i, class_label in enumerate(classes):
        support = sum(row[i] for row in y_true_bin)
        predicted_labels = sum(row[i] for row in y_pred_bin)
        correct_predictions = sum(
            1
            for true_row, pred_row in zip(y_true_bin, y_pred_bin)
            if true_row[i] == pred_row[i] == 1
        )

        # Precision, recall, and F1 score calculations
        if predicted_labels > 0:
            precision = correct_predictions / predicted_labels
        else:
            precision = zero_division

        if support > 0:
            recall = correct_predictions / support
        else:
            recall = zero_division

        if (precision + recall) > 0:
            f1_score = (2 * precision * recall) / (precision + recall)
        else:
            f1_score = zero_division

        # Add stats to the report
        report[class_label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,
            "support": support,
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
        "support": len(y_true),
    }

    return report


def calculate_f1_score_multi_label(
    y_true: List[Set[Union[str, int]]],
    y_pred: List[Set[Union[str, int]]],
    average: str = "macro",
    zero_division: int = 0,
) -> float:
    """
    Calculate the F1 score for multi-label classification using binarized labels.

    Args:
        y_true (List[Set[Union[str, int]]]): List of sets of true labels.
        y_pred (List[Set[Union[str, int]]]): List of sets of predicted labels.
        average (str, optional): Method to calculate F1 score, can be 'micro', 'macro', or 'weighted'. Defaults to 'macro'.
        zero_division (int, optional): Value to return when there is a zero division case. Defaults to 0.

    Returns:
        float: Calculated F1 score for multi-label classification.

    Raises:
        AssertionError: If lengths of y_true and y_pred are not equal.
        ValueError: If invalid averaging method is provided.
    """
    assert len(y_true) == len(y_pred), "Lengths of y_true and y_pred must be equal."

    # Binarize the labels and get the unique class set
    y_true_bin, y_pred_bin, classes = simple_multilabel_binarizer(y_true, y_pred)

    # Number of classes should remain consistent
    num_classes = len(classes)

    if average == "micro":
        true_positives = sum(
            1
            for i in range(len(y_true_bin))
            for j in range(num_classes)
            if y_true_bin[i][j] == y_pred_bin[i][j] == 1
        )
        false_positives = sum(
            1
            for i in range(len(y_true_bin))
            for j in range(num_classes)
            if y_pred_bin[i][j] == 1 and y_true_bin[i][j] == 0
        )
        false_negatives = sum(
            1
            for i in range(len(y_true_bin))
            for j in range(num_classes)
            if y_pred_bin[i][j] == 0 and y_true_bin[i][j] == 1
        )

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

    elif average in ["macro", "weighted"]:
        f1_score = 0.0
        total_support = sum(
            sum(y_true_bin[i][j] for i in range(len(y_true_bin)))
            for j in range(num_classes)
        )

        for j in range(num_classes):
            true_positives = sum(
                1
                for i in range(len(y_true_bin))
                if y_true_bin[i][j] == y_pred_bin[i][j] == 1
            )
            false_positives = sum(
                1
                for i in range(len(y_true_bin))
                if y_pred_bin[i][j] == 1 and y_true_bin[i][j] == 0
            )
            false_negatives = sum(
                1
                for i in range(len(y_true_bin))
                if y_pred_bin[i][j] == 0 and y_true_bin[i][j] == 1
            )

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

            if precision + recall > 0:
                class_f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                class_f1_score = 0.0

            # Support for the current class (how many times it appears in y_true)
            support = sum(y_true_bin[i][j] for i in range(len(y_true_bin)))

            if average == "macro":
                f1_score += class_f1_score / num_classes
            elif average == "weighted":
                # Normalize weights by dividing the support by the total number of labels
                weight = support / total_support if total_support > 0 else 0
                f1_score += weight * class_f1_score

    else:
        raise ValueError(
            "Invalid averaging method. Must be 'micro', 'macro', or 'weighted'."
        )

    return min(f1_score, 1.0)  # Ensure the F1 score is capped at 1.0


def combine_labels(labels: List[str]) -> List[str]:
    """
    Combines labels for degradation analysis.
    input labels: ["B-ORG", "I-ORG", "B-PER", "I-PER"]
    output labels: ["ORG", "PER"]
    """
    try:
        output_list = []
        if isinstance(labels, str):
            if labels.startswith("B-") or labels.startswith("I-"):
                output_list.append(labels[2:])
                return output_list
            else:
                return [labels]
        elif isinstance(labels, list):
            for label in labels:
                if label.startswith("I-") and label[2:] == output_list[-1]:
                    continue
                if label.startswith("I-") and label[2:] != output_list[-1]:
                    continue
                output_list.append(label.split("-")[-1])

            return output_list
        else:
            raise ValueError("Input should be a list or a string.")
    except ValueError as e:
        logger.error(f"{e}")
