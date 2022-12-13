"""Functions to test NER model robustness against different kinds of perturbations"""
import json
import random
from copy import deepcopy
import pandas as pd
from pandas import DataFrame

from .utils import A2B_DICT
from .perturbations import PERTURB_FUNC_MAP, PERTURB_DESCRIPTIONS, create_terminology

from pyspark.sql import SparkSession
from sparknlp.base import PipelineModel
from typing import Optional, List, Dict, Tuple, Any
from sklearn.metrics import classification_report


def remove_context_tokens(column: List[str], starting_context_tokens: List[str],
                          ending_context_tokens: List[str]) -> List[str]:
    """
    Removes user-defined context tokens from strings

    :param column: list of sentences to process
    :param starting_context_tokens: list of starting context tokens to remove
    :param ending_context_tokens: list of ending context tokens to remove
    """

    def match_starting_context(token_list):

        for context_token in starting_context_tokens:
            length_context = len(context_token)
            token_string = " ".join([token.metadata['word'] for token in token_list[:length_context]])
            if token_string == context_token:
                return length_context

        return 0

    def match_ending_context(token_list):

        for context_token in ending_context_tokens:
            length_context = len(context_token)
            token_string = " ".join([token.metadata['word'] for token in token_list[-length_context:]])
            if token_string == context_token:
                return len(token_list) - length_context

        return len(token_list)

    outcome_list = []
    for token_list in column:
        starting_indx = match_starting_context(token_list)
        ending_indx = match_ending_context(token_list)
        outcome_list.append(token_list[starting_indx:ending_indx])

    return outcome_list


def remove_contraction_tokens(list_with_contractions: List[str], list_without_contractions: List[str]) -> None:
    """
    Removes contraction tokens

    :param list_with_contractions: list of sentences with contractions
    :param list_without_contractions: list of sentences without contractions
    """
    idx = []

    for i, element in enumerate(list_with_contractions):

        if list_with_contractions[i].metadata['word'] != list_without_contractions[i].metadata['word']:
            idx.append(i)

    if len(idx) > 0:
        contraction_idx = idx[0]

        del list_with_contractions[contraction_idx]

        del list_without_contractions[contraction_idx:contraction_idx + 2]


def remove_punctuation_tokens(column: List[str]) -> List[str]:
    """
    Removes all punctuation tokens from input sentences

    :param column: list of sentences to process
    """
    outcome_list = []

    punc_list = ['!', '?', ',', '.', '-', ':', ';']

    for token_list in column:

        lst = []

        for token in token_list:
            if token.metadata['word'] not in punc_list:
                lst.append(token)

        outcome_list.append(lst)

    return outcome_list


def calculate_metrics(filtered_df: DataFrame, method: str = 'strict') -> Dict[str, Any]:
    """
    Calculates comparison metrics for robustness

    :param filtered_df: dataframe created during robustness tests
    :param method: 'strict' calculates metrics for IOB2 format, 'flex' calculates for IO format
    """
    comparison_df = pd.DataFrame()

    comparison_df['original_token'] = filtered_df['ner'].apply(lambda x: x.metadata['word'] if pd.notnull(x) else x)

    comparison_df['original_label'] = filtered_df['ner'].apply(lambda x: x.result if pd.notnull(x) else x)

    comparison_df['original_flex_label'] = filtered_df['ner'].apply(
        lambda x: x.result.split("-")[-1] if pd.notnull(x) else x)

    comparison_df['noisy_token'] = filtered_df['noisy_ner'].apply(lambda x: x.metadata['word'] if pd.notnull(x) else x)

    comparison_df['noisy_label'] = filtered_df['noisy_ner'].apply(lambda x: x.result if pd.notnull(x) else x)

    comparison_df['noisy_flex_label'] = filtered_df['noisy_ner'].apply(
        lambda x: x.result.split("-")[-1] if pd.notnull(x) else x)

    def round_metrics(m):
        for ent, metric_dict in m.items():
            if ent == 'accuracy':
                m['accuracy'] = round(m['accuracy'], 2)
                continue
            for k, v in metric_dict.items():
                if k != 'support':
                    m[ent][k] = round(v, 2)
        return m

    if method == 'strict':
        metrics = classification_report(comparison_df['original_label'].astype(str),
                                        comparison_df['noisy_label'].astype(str),
                                        output_dict=True)
        metrics = round_metrics(metrics)

    if method == 'flex':
        metrics = classification_report(comparison_df['original_flex_label'].astype(str),
                                        comparison_df['noisy_flex_label'].astype(str), output_dict=True)
        metrics = round_metrics(metrics)

    outcome = {'metrics': metrics, 'comparison_df': comparison_df}

    return outcome


def run_test(spark: SparkSession, noise_type: str, noise_description: str, pipeline_model: PipelineModel,
             test_set: List[str], total_amount: int, original_annotations_df: DataFrame, noisy_test_set: List[str],
             metric_type: str, starting_context_token_list: Optional[List[str]] = None,
             ending_context_token_list: Optional[List[str]] = None,
             ) -> Tuple[DataFrame, str, DataFrame]:
    """
    Runs comparisons between original list of sentences and noisy list of sentences, returning metrics and dataframe
    for comparison

    :param spark: An active Spark Session to create spark DataFrame
    :param noise_type: type of noise to introduce in sentences for running tests on 'modify_capitalization_upper',
    'modify_capitalization_lower', 'modify_capitalization_title', 'add_punctuation', 'strip_punctuation',
    'introduce_typos', 'add_contractions', 'add_context', 'american_to_british', 'british_to_american',
    'swap_entities', 'swap_cohyponyms'
    :param noise_description: description of the type of noise for user awareness
    :param pipeline_model: PipelineModel with document assembler, sentence detector, tokenizer, word embeddings if
    applicable, NER model with 'ner' output name, NER converter with 'ner_chunk' output name
    :param test_set: list of original sentences to process
    :param total_amount: length of the list of original sentences to process
    :param original_annotations_df: DataFrame containing LightPipeline model fullAnnotate results on original list of
    sentences
    :param noisy_test_set: list of sentences with perturbations
    :param metric_type: 'strict' calculates metrics for IOB2 format, 'flex' calculates for IO format
    which disrupt token match-up between original test set and noisy test set, options are None,
    'remove_context_tokens', 'remove_contraction_tokens', 'remove_punctuation_tokens'
    :param starting_context_token_list: list of starting context tokens to add when applying the 'add_context' noise type
    :param ending_context_token_list: list of ending context tokens to add when applying the 'add_context' noise type
    """
    report_text = '\n\n' + noise_type + '\nGenerated noise: ' + noise_description

    noisy_test_data = [[i] for i in noisy_test_set]
    noisy_test_data = spark.createDataFrame(noisy_test_data).toDF("text")
    noisy_annotations = pipeline_model.transform(noisy_test_data)
    noisy_annotations_df = noisy_annotations.select('ner').toPandas()

    noisy_lst = []
    for i in range(len(test_set)):
        noisy_lst.append(test_set[i] != noisy_test_set[i])

    report_text = report_text + '\nGenerated noise affects ' + str(sum(noisy_lst)) + ' sentences (' + str(
        (round(100 * (sum(noisy_lst) / total_amount), 2))) + '% of the test set).\n'

    if noise_type == 'add_punctuation' or noise_type == 'strip_punctuation':

        noisy_annotations_df['ner'] = remove_punctuation_tokens(column=noisy_annotations_df['ner'])

        noisy_annotations_df['token_count'] = noisy_annotations_df['ner'].apply(lambda x: len(x))

        reduced_original_annotations_df = original_annotations_df.copy()

        reduced_original_annotations_df['ner'] = remove_punctuation_tokens(
            column=reduced_original_annotations_df['ner'])

        reduced_original_annotations_df['token_count'] = reduced_original_annotations_df['ner'].apply(
            lambda x: len(x))

        noisy_annotations_df = noisy_annotations_df.rename(
            columns={'ner': 'noisy_ner', 'sentence': 'noisy_sentence', 'token_count': 'noisy_token_count'})

        joined_df = reduced_original_annotations_df[['token_count', 'ner']].join(
            noisy_annotations_df[['noisy_token_count', 'noisy_ner']])

    elif noise_type == 'add_contractions':

        noisy_annotations_df['token_count'] = noisy_annotations_df['ner'].apply(lambda x: len(x))

        noisy_annotations_df = noisy_annotations_df.rename(
            columns={'ner': 'noisy_ner', 'sentence': 'noisy_sentence', 'token_count': 'noisy_token_count'})

        joined_df = original_annotations_df[['token_count', 'ner']].join(
            noisy_annotations_df[['noisy_token_count', 'noisy_ner']])

        for index in range(len(joined_df)):
            remove_contraction_tokens(list_with_contractions=joined_df['noisy_ner'][index],
                                      list_without_contractions=joined_df['ner'][index])

        joined_df['token_count'] = joined_df['ner'].apply(lambda x: len(x))

        joined_df['noisy_token_count'] = joined_df['noisy_ner'].apply(lambda x: len(x))

    elif noise_type == 'add_context':

        noisy_annotations_df['ner'] = remove_context_tokens(column=noisy_annotations_df['ner'],
                                                            starting_context_tokens=starting_context_token_list,
                                                            ending_context_tokens=ending_context_token_list)

        noisy_annotations_df['token_count'] = noisy_annotations_df['ner'].apply(lambda x: len(x))

        noisy_annotations_df = noisy_annotations_df.rename(
            columns={'ner': 'noisy_ner', 'sentence': 'noisy_sentence', 'token_count': 'noisy_token_count'})

        reduced_original_annotations_df = original_annotations_df.copy()

        for indx, (noisy, original) in enumerate(zip(
                noisy_annotations_df['noisy_ner'], reduced_original_annotations_df['ner'])):
            if len(noisy) == len(original) - 1 and not original[-1].metadata['word'][-1].isalnum():
                reduced_original_annotations_df['ner'].iloc[indx] = original[:-1]

        reduced_original_annotations_df['token_count'] = reduced_original_annotations_df['ner'].apply(
            lambda x: len(x))

        joined_df = reduced_original_annotations_df[['token_count', 'ner']].join(
            noisy_annotations_df[['noisy_token_count', 'noisy_ner']])

    else:

        noisy_annotations_df['token_count'] = noisy_annotations_df['ner'].apply(lambda x: len(x))

        noisy_annotations_df = noisy_annotations_df.rename(
            columns={'ner': 'noisy_ner', 'sentence': 'noisy_sentence', 'token_count': 'noisy_token_count'})

        joined_df = original_annotations_df[['token_count', 'ner']].join(
            noisy_annotations_df[['noisy_token_count', 'noisy_ner']])

    filtered_df = joined_df[joined_df['token_count'] == joined_df['noisy_token_count']][['ner', 'noisy_ner']]

    filtered_df = filtered_df[
        filtered_df['ner'].apply(lambda x: len(x)) == filtered_df['noisy_ner'].apply(lambda x: len(x))]

    filtered_sentences = total_amount - len(filtered_df)

    report_text = report_text + '\nA total amount of ' + str(filtered_sentences) + \
                  " were filtered out due to mismatching tokenization (" + \
                  str((round(100 * (filtered_sentences / total_amount), 2))) + "% of the test set)."

    filtered_df = filtered_df.apply(pd.Series.explode).reset_index()

    test_outcomes = calculate_metrics(filtered_df=filtered_df, method=metric_type)

    test_metrics = test_outcomes['metrics']

    test_metrics['test'] = noise_type

    comparison_df = test_outcomes['comparison_df'][
        ['original_token', 'noisy_token', 'original_label', 'noisy_label']].copy()

    comparison_df['test'] = noise_type

    report_text = report_text + "\nf1 - macro average: " + str(
        round(test_metrics['macro avg']['f1-score'], 2)
    )

    report_text = report_text + "\nf1 - weighted average: " + str(
        round(test_metrics['weighted avg']['f1-score'], 2)
    )

    return test_metrics, report_text, comparison_df


def conll_sentence_reader(conll_path: str) -> List[str]:
    """
    Read CoNLL file and convert it to the list of sentences.

    :param conll_path: CoNLL file path.
    :return: list of sentences in the conll data.
    """
    with open(conll_path) as f:
        data = []
        content = f.read()
        docs = [i.strip() for i in content.strip().split('-DOCSTART- -X- -X- O') if i != '']
        for doc in docs:

            #  file content to sentence split
            sentences = doc.strip().split('\n\n')

            if sentences == ['']:
                data.append(([''], [''], ['']))
                continue

            for sent in sentences:
                sentence_data = []

                #  sentence string to token level split
                tokens = sent.strip().split('\n')

                # get annotations from token level split
                token_list = [t.split() for t in tokens]

                #  get token and labels from the split
                for split in token_list:
                    sentence_data.append(split[0])

                data.append(" ".join(sentence_data))

    return data


def test_robustness(spark: SparkSession, pipeline_model: PipelineModel, test_file_path: str,
                    metric_type: Optional[str] = 'flex',
                    metrics_output_format: str = 'dictionary',
                    log_path: str = 'robustness_test_results.json',
                    noise_prob: float = 0.5,
                    sample_sentence_count: int = None,
                    test: Optional[List[str]] = None,
                    starting_context: Optional[List[str]] = None,
                    ending_context: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Tests robustness of a NER model by applying different types of noise generating functions to a list of sentences.
    Metrics are calculated by comparing model's extractions in the original list of sentences set with the extractions
    done in the noisy list of sentences.

    :param spark: An active SparkSession to create Spark DataFrame
    :param pipeline_model: PipelineModel with document assembler, sentence detector, tokenizer, word embeddings if
    applicable, NER model with 'ner' output name, NER converter with 'ner_chunk' output name
    :param test_file_path: Path to test file to test robustness. Can be .txt or .conll file in CoNLL format or
    .csv file with just one column (text) with series of test samples.
    :param test: type of robustness test implemented by the function, options include
    'capitalization_upper': capitalization of the test set is turned into uppercase
    'capitalization_lower': capitalization of the test set is turned into lowercase
    'capitalization_title': capitalization of the test set is turned into title case
    'add_punctuation': special characters at end of each sentence are replaced by other special characters, if no
    special character at the end, one is added
    'strip_punctuation': special characters are removed from the sentences (except if found in numbers, such as '2.5')
    'introduce_typos': typos are introduced in sentences
    'add_contractions': contractions are added where possible (e.g. 'do not' contracted into 'don't')
    'add_context': tokens are added at the beginning and at the end of the sentences.
    'swap_entities': named entities replaced with same entity type with same token count from terminology.
    'swap_cohyponyms': Named entities replaced with co-hyponym from the WordNet database.
    'american_to_british': American English will be changed to British English.
    'british_to_american': British English will be changed to American English.
    :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
    'american_to_british' or 'british_to_american' will be run depending on test_set_language value
    (by default, 'american_to_british')
    :param sample_sentence_count: Number of sentence that will be sampled from the test_data.
    :param metric_type: 'strict' calculates metrics for IOB2 format, 'flex' (default) calculates for IO format
    :param metrics_output_format: 'dictionary' to get a dictionary report, 'dataframe' to get a dataframe report
    :param log_path: Path to log file, False to avoid saving test results. Default 'robustness_test_results.json'
    :param starting_context: list of context tokens to add to beginning when running the 'add_context' test
    :param ending_context: list of context tokens to add to end when running the 'add_context' test
    """

    outcome = {}
    perturb_metrics = dict()
    complete_comparison_df = pd.DataFrame(
        columns=['original_token', 'noisy_token', 'original_label', 'noisy_label', 'test'])

    if test_file_path.endswith('.txt') or test_file_path.endswith('.conll'):
        test_set = conll_sentence_reader(test_file_path)
    elif test_file_path.endswith('.csv'):
        test_df = pd.read_csv(test_file_path)
        test_set = test_df['text'].tolist()
    else:
        raise ValueError("'test_file_path' could not be read! It should be txt, conll or csv file.")

    test_set = [sent for sent in test_set if not len(sent.split()) < 2]
    if sample_sentence_count:
        if sample_sentence_count < len(test_set):
            test_set = random.sample(test_set, sample_sentence_count)
        else:
            raise ValueError(
                f"sample_sentence_count ({sample_sentence_count}) "
                f"must be greater than test_set length ({len(test_set)}).")
    total_amount = len(test_set)

    if test is None:
        test = ['capitalization_upper', 'capitalization_lower', 'capitalization_title',
                'add_punctuation', 'strip_punctuation', 'introduce_typos', 'add_contractions', 'add_context',
                'american_to_british', 'swap_entities', 'swap_cohyponyms']

    report_text = 'Test type: ' + ', '.join(test) + '\nTest set size: ' + str(total_amount) + ' sentences\n'

    test_data = [[i] for i in test_set]
    test_data = spark.createDataFrame(test_data).toDF('text')

    original_annotations = pipeline_model.transform(test_data)
    original_annotations_df = original_annotations.select('ner').toPandas()
    original_annotations_df['token_count'] = original_annotations_df['ner'].apply(lambda x: len(x))

    if 'add_context' not in test:
        starting_context_tokens = None
        ending_context_tokens = None

    else:
        if starting_context is None:
            starting_context = ["Description:", "MEDICAL HISTORY:", "FINDINGS:", "RESULTS: ",
                                "Report: ", "Conclusion is that "]

        if ending_context is None:
            ending_context = ["according to the patient's family", "as stated by the patient",
                              "due to various circumstances", "confirmed by px"]

        starting_context_token_list = [[i] for i in starting_context]
        starting_context_token_list = spark.createDataFrame(starting_context_token_list).toDF('text')
        starting_context_token_list = pipeline_model.transform(starting_context_token_list)

        starting_context_tokens = []
        for context_token in starting_context_token_list.select('token').collect():
            starting_context_tokens.append([token.result for token in context_token[0]])

        ending_context_token_list = [[i] for i in ending_context]
        ending_context_token_list = spark.createDataFrame(ending_context_token_list).toDF('text')
        ending_context_token_list = pipeline_model.transform(ending_context_token_list)

        ending_context_tokens = []
        for context_token in ending_context_token_list.select('token').collect():
            ending_context_tokens.append([token.result for token in context_token[0]])

    terminology = None
    labels = None
    if 'swap_entities' in test:

        labels = []
        sentences = []
        for row in original_annotations_df['ner']:
            sent_tokens = [token.metadata['word'] for token in row]
            sentences.append(" ".join(sent_tokens))

            sent_labels = [token.result for token in row]
            labels.append(" ".join(sent_labels))

        terminology = create_terminology(sentences, labels)

    a2b_dict = A2B_DICT
    b2a_dict = {v: k for k, v in a2b_dict.items()}

    perturb_args = {
        "capitalization_upper": {},
        "capitalization_lower": {},
        "capitalization_title": {},
        "add_punctuation": {},
        "strip_punctuation": {},
        "introduce_typos": {},
        "american_to_british": {'accent_map': a2b_dict},
        "british_to_american": {'accent_map': b2a_dict},
        "add_context": {
            'ending_context': ending_context_tokens,
            'starting_context': starting_context_tokens,
        },
        "add_contractions": {},
        "swap_entities": {'labels': labels, 'terminology': terminology},
        "swap_cohyponyms": {'labels': labels}
    }

    for test_type in test:

        noise_description = PERTURB_DESCRIPTIONS[test_type]

        aug_indx, aug_sent, _, _ = PERTURB_FUNC_MAP[test_type](test_set, noise_prob=noise_prob,
                                                                **perturb_args[test_type])
        noisy_test_sent = deepcopy(test_set)
        for sentence, indx in zip(aug_sent, aug_indx):
            noisy_test_sent[indx] = sentence

        test_metrics, text, comparison_df = run_test(
            spark=spark,
            noise_type=test_type,
            noise_description=noise_description,
            pipeline_model=pipeline_model,
            test_set=test_set,
            total_amount=total_amount,
            original_annotations_df=original_annotations_df,
            noisy_test_set=noisy_test_sent,
            metric_type=metric_type,
            starting_context_token_list=starting_context_tokens,
            ending_context_token_list=ending_context_tokens
        )

        perturb_metrics[test_type] = test_metrics
        report_text = report_text + text
        complete_comparison_df = pd.concat([complete_comparison_df, comparison_df]).reset_index(drop=True)

    outcome['metrics'] = perturb_metrics
    outcome['comparison_df'] = complete_comparison_df
    outcome['test_details'] = report_text

    if log_path:
        test_results = outcome.copy()
        complete_comparison = dict()
        groups = outcome['comparison_df'].groupby('test')
        for name, group in groups:
            group = group.drop('test', axis=1)
            complete_comparison[name] = group.to_dict('list')
        test_results['comparison_df'] = complete_comparison

        try:
            with open(log_path, 'w') as f:
                try:
                    f.write(json.dumps(test_results))

                except (IOError, OSError) as e:
                    print(f"Error while writing to the {log_path}. Log file will not be written.")
                    print(e)

        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"Error while opening the {log_path}. Log file will be ignored.")
            print(e)

    if metrics_output_format == 'dataframe':

        metrics_df = pd.DataFrame(columns=['entity', 'precision', 'recall', 'f1-score', 'support', 'test'])

        for perturb_type, metrics in outcome['metrics'].items():
            metrics = pd.DataFrame.from_dict(metrics) \
                .transpose() \
                .reset_index() \
                .rename({'index': 'entity'},
                        axis='columns')
            metrics['test'] = perturb_type
            metrics_df = pd.concat([metrics_df, metrics]).reset_index(drop=True)

        outcome['metrics'] = metrics_df

    return outcome
