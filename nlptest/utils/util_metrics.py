from collections import Counter

# suppose y_true and y_pred are your lists of true and predicted labels respectively

def classification_report(y_true, y_pred):
    # Count total true labels for each class (support)
    support = Counter(y_true)
    
    # Count correct predictions for precision and recall
    correct_predictions = Counter([pred for true, pred in zip(y_true, y_pred) if true == pred])
    predicted_labels = Counter(y_pred)
    
    # Initialize data structure for report
    report = {}

    # Compute stats for each class
    for class_label in set(y_true).union(set(y_pred)):
        # Precision is the ratio of correct predictions to total predictions for each class
        precision = correct_predictions[class_label] / predicted_labels[class_label] if predicted_labels[class_label] > 0 else 0
        
        # Recall is the ratio of correct predictions to total true instances of the class
        recall = correct_predictions[class_label] / support[class_label] if support[class_label] > 0 else 0
        
        # F1 score is the harmonic mean of precision and recall
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        # Add stats to report
        report[class_label] = {'precision': precision,
                               'recall': recall,
                               'f1-score': f1_score,
                               'support': support[class_label]}
    
    # Compute macro averages
    avg_precision = sum([metrics['precision'] for metrics in report.values()]) / len(report)
    avg_recall = sum([metrics['recall'] for metrics in report.values()]) / len(report)
    avg_f1_score = sum([metrics['f1-score'] for metrics in report.values()]) / len(report)
    
    report['macro avg'] = {'precision': avg_precision,
                           'recall': avg_recall,
                           'f1-score': avg_f1_score,
                           'support': sum(support.values())}
    
    return report
