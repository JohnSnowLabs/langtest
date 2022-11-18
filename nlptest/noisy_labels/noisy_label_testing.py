import json
import numpy as np
import pandas as pd
from typing import List

from sparknlp.base import Pipeline
from pyspark.sql import SparkSession, Window
from sparknlp.training import CoNLL
from pyspark.sql.functions import row_number, monotonically_increasing_id


def calculate_label_error_score(
        labels: np.array,
        pred_probs: np.array
) -> np.array:
    return np.array([np.mean(pred_probs[i, l]) for i, l in enumerate(labels)])


def get_label_quality_scores(
        labels: List[List[int]],
        pred_probs: List[np.array],
) -> List[pd.Series]:
    """
    A method to calculate error score for each label in the sentence, using model confidence scores.
    :param labels: List of labels for each sentence. Labels should be in integer format.
    :param pred_probs: List of np.array where each array contains model confidence score for each class.
    :return: Error score between 0 and 1 for each label, high scores mean model confident with this class.
    """
    labels_flatten = np.array([l for label in labels for l in label])
    pred_probs_flatten = np.array([p for pred_prob in pred_probs for p in pred_prob])

    sentence_length = [len(label) for label in labels]

    def nested_list(x, sentence_length):
        i = iter(x)
        return [[next(i) for _ in range(length)] for length in sentence_length]

    token_scores = calculate_label_error_score(
        labels=labels_flatten, pred_probs=pred_probs_flatten
    )
    scores_nl = nested_list(token_scores, sentence_length)

    token_info = [pd.Series(scores) for scores in scores_nl]

    return token_info


def get_unique_entities(conll_path: str) -> List[str]:
    entities = []
    with open(conll_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if len(line) == 4 and line[-1] not in entities:
                entities.append(line[-1])
    return entities


def test_label_errors(spark: SparkSession, training_pipeline: Pipeline,
                      conll_path: str, k: int = 4, threshold: float = 0.3,
                      log_path: str = 'noisy_label_test_results.json') -> pd.DataFrame:
    """
    This function splits the data in k parts and gets predictions for each part using a model trained in the rest of the
    parts (k - 1). The process is similar to cross-validation. This function creates an error report that can be used to
    identify the most probable labeling errors in the dataset.

    :param spark: An active spark session.
    :param training_pipeline: Training pipeline with NerDLApproach to train with each k-folds. Output column of
    NerDLApproach should be `ner` and 'setIncludeConfidenceScores' and 'setIncludeConfidence' is set as True.
    :param conll_path: Path to CoNLL data.
    :param k:  Amount of parts in which the training data is going to be split for cross-validation. This number 
    should be between [3, 10]. The higher the number, the longer it will take to get predictions.
    :param threshold: Threshold to filter noisy label issues. Should be between 0 and 1.
    :param log_path: Path to log file, False to avoid saving test results. Default 'noisy_label_test_results.json'
    :return: Result dataframe consist of sentence, sentence indexes, token, token indexes, ground truth label,
    predicted label and score. Each key has same length of list with corresponding values.
    """

    result = {
        'sentence': [],
        'sent_indx': [],
        'token': [],
        'token_indx': [],
        'ground_truth': [],
        'prediction': [],
        'score': [],
        'prediction_confidence': [],
        'prediction_indx': [],
        'chunk_indx': []
    }

    data = CoNLL().readDataset(spark, conll_path)

    entities = get_unique_entities(conll_path)
    label2id = {entity: indx for indx, entity in enumerate(entities)}
    id2label = {indx: entity for indx, entity in enumerate(entities)}

    data = data.withColumn("sent_id", row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)

    portion_split = [1 / k] * k
    data_splits = data.randomSplit(portion_split)
    assert 2 < k < 11, "Argument 'k' out of range [3, 10]"

    # global counter to track chunks
    chunk_counter = 0
    prediction_counter = 0
    for k_trial in range(k):

        test_data = data_splits[k_trial]
        test_ids = test_data.select('sent_id').collect()
        test_ids = [int(row.sent_id) for row in test_ids]

        train_data = data.filter(~data.sent_id.isin(test_ids))

        print(f"Training {k_trial + 1}/{k} is started.")
        trained_model = training_pipeline.fit(train_data)

        is_model_allowed = False
        for annotator in trained_model.stages:
            if str(annotator).startswith('NerDLModel') or str(annotator).startswith('MedicalNerModel'):
                assert annotator.getIncludeAllConfidenceScores() and annotator.getIncludeConfidence(), \
                    "Include confidence scores should be set as True.\n" \
                    "Make sure 'setIncludeConfidenceScores' and 'setIncludeConfidence' is set as True in NerDLApproach!"
                is_model_allowed = True
        assert is_model_allowed, 'Pipeline is invalid! Make sure it contains NerDLApproach or MedicalNerApproach.'

        print(f"Training {k_trial + 1}/{k} is completed.")
        ner_results = trained_model.transform(test_data).collect()

        labels_all = []
        token_all = []
        sentences = []
        sentence_ids = []
        confidence_scores_all = []
        for row in ner_results:

            sentence_scores = []
            sentence_labels = []
            sentence_token = []
            sentence = row.sentence[0].result
            sent_id = row.sent_id
            for label, ner in zip(row.label, row.ner):

                confidence_scores = [0] * len(label2id)

                for entity_type, score in ner.metadata.items():
                    if label2id.get(entity_type) is not None:
                        confidence_scores[label2id[entity_type]] = float(score)

                sentence_scores.append(confidence_scores)
                sentence_labels.append(label2id[label.result])
                sentence_token.append(label.metadata['word'])

            confidence_scores_all.append(np.array(sentence_scores))
            labels_all.append(sentence_labels)
            token_all.append(sentence_token)
            sentences.append(sentence)
            sentence_ids.append(sent_id)

        token_scores = get_label_quality_scores(labels_all, confidence_scores_all)
        print(f"Noisy labels are extracted from fold {k_trial + 1}.")
        assert 0 < threshold < 1, "Threshold must be between 0 and 1."

        for sent_indx, sent_scores in enumerate(token_scores):

            prediction_counter += 1
            chunk_counter += 1

            chunk_ent_type = None
            prediction_ent_type = None

            for token_indx, label_score in enumerate(sent_scores):

                result['sent_indx'].append(sentence_ids[sent_indx])
                result['token_indx'].append(token_indx)

                sentence = sentences[sent_indx]
                result['sentence'].append(sentence)

                token = token_all[sent_indx][token_indx]
                result['token'].append(token)

                score = 1 - token_scores[sent_indx][token_indx]
                result['score'].append(score)

                predicted_token = confidence_scores_all[sent_indx][token_indx].argmax()
                prediction = id2label[predicted_token]
                result['prediction'].append(prediction)

                label_id = labels_all[sent_indx][token_indx]
                ground_truth = id2label[label_id]
                result['ground_truth'].append(ground_truth)

                model_confidence = max(confidence_scores_all[sent_indx][token_indx])
                result['prediction_confidence'].append(round(model_confidence, 2))

                #   get ground_truth chunk boundaries
                if ground_truth.startswith('B'):
                    chunk_counter += 1
                    chunk_ent_type = ground_truth[2:]
                    result['chunk_indx'].append(chunk_counter)

                elif ground_truth.startswith('I'):

                    if ground_truth[2:] != chunk_ent_type:
                        chunk_counter += 1

                    # sometimes model made an error to label I without B
                    if result['chunk_indx'] and result['chunk_indx'][-1] == chunk_counter:
                        result['chunk_indx'].append(chunk_counter)
                    else:
                        chunk_counter += 1
                        result['chunk_indx'].append(chunk_counter)

                else:
                    result['chunk_indx'].append(0)
                    chunk_ent_type = None

                #   get prediction chunk boundaries
                if prediction.startswith('B'):
                    prediction_counter += 1
                    prediction_ent_type = prediction[2:]
                    result['prediction_indx'].append(prediction_counter)

                elif prediction.startswith('I'):

                    if prediction[2:] != prediction_ent_type:
                        prediction_counter += 1

                    # sometimes model made an error to label I without B
                    if result['prediction_indx'] and result['prediction_indx'][-1] == prediction_counter:
                        result['prediction_indx'].append(prediction_counter)
                    else:
                        prediction_counter += 1
                        result['prediction_indx'].append(prediction_counter)

                else:
                    result['prediction_indx'].append(0)
                    prediction_ent_type = None

    df = pd.DataFrame.from_dict(result)

    #   calculate chunk score for each chunk
    df['chunk_score'] = df['score']
    df['chunk'] = df['token']

    for indx, group in df.groupby('chunk_indx'):
        if indx == 0:
            continue
        df.loc[group.index, 'chunk'] = " ".join(group['token'])
        df.loc[group.index, 'chunk_score'] = group['score'].max()

    for indx, group in df.groupby('prediction_indx'):
        if indx == 0:
            continue
        max_score = group['chunk_score'].max()
        chunk = group['chunk'].iloc[0]
        if len(group['token']) > len(chunk.split(' ')):
            chunk = " ".join(group['token'])

        df.loc[group.index, 'chunk'] = chunk
        df.loc[group.index, 'chunk_score'] = max_score

    filtered_by_threshold = df[df.chunk_score > threshold]
    filtered_by_threshold = filtered_by_threshold.drop(columns=['prediction_indx', 'chunk_indx'])
    sorted_df = filtered_by_threshold.sort_values(
        by=['chunk_score', 'sent_indx', 'token_indx'],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    if log_path:
        try:
            with open(log_path, 'w') as log_file:
                try:
                    json_content = json.dumps(sorted_df.to_dict('list'))
                    log_file.write(json_content)

                except (IOError, OSError) as e:
                    print(f"Error while writing to the {log_path}. Log file will not be written.")
                    print(e)

        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"Error while opening the {log_path}. Log file will be ignored.")
            print(e)

    return sorted_df
