"""Functions to fix robustness of NER model with different kinds of perturbations in CoNLL data"""
import re
import random
import logging
import itertools
import numpy as np
from typing import Optional, List, Dict, Any

from sparknlp.base import PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import Row

from .robustness_testing import test_robustness
from .perturbations import PERTURB_FUNC_MAP, LIST_OF_PERTURBATIONS, create_terminology
from .utils import DF_SCHEMA, A2B_DICT, suggest_perturbations


def create_dataframe(
        spark: SparkSession,
        data: List[str],
        pos_sync_tags: List[str],
        labels: List[str]
) -> DataFrame:
    """
    Create Spark Dataframe from given data and labels extracted from the CoNLL file.

    :param spark: An active spark session to create the dataframe.
    :param data: List of sentences.
    :param pos_sync_tags: List of pos and synthetic tags separated by '-' and whitespace.
    :param labels: Corresponding NER labels of the data.
    :return: Spark DataFrame consisting of 'text', 'document', 'sentence', 'token', 'label', 'pos' columns.
    """
    spark_data = []
    for sentence, sentence_tags, sentence_label in zip(data, pos_sync_tags, labels):

        token_rows = []
        label_rows = []
        tags_rows = []

        begin = end = 0
        for token, tag, label in zip(sentence.split(), sentence_tags.split(), sentence_label.split()):
            end += len(token) - 1

            token_rows.append(Row(
                annotatorType='token',
                begin=begin,
                end=end,
                result=token,
                metadata={'sentence': '0'},
                embeddings=[]
            )),

            label_rows.append(Row(
                annotatorType='named_entity',
                begin=begin,
                end=end,
                result=label,
                metadata={'sentence': '0', 'word': token},
                embeddings=[]
            ))

            tags_rows.append(Row(
                annotatorType='pos',
                begin=begin,
                end=end,
                result=tag.split('*')[0],
                metadata={'sentence': '0', 'word': token},
                embeddings=[]
            ))

            end = begin = end + 2

        spark_data.append(
            {
                'text': sentence,
                'document': [Row(
                    annotatorType='document',
                    begin=0,
                    end=len(sentence) - 1,
                    result=sentence,
                    metadata={'training': 'true'},
                    embeddings=[]
                )],
                'sentence': [Row(
                    annotatorType='document',
                    begin=0,
                    end=len(sentence) - 1,
                    result=sentence,
                    metadata={'sentence': '0'},
                    embeddings=[]
                )],
                'pos': tags_rows,
                'token': token_rows,
                'label': label_rows
            }
        )

    return spark.createDataFrame(spark_data, DF_SCHEMA)


def conll_reader(conll_path: str) -> List[tuple]:
    """
    Read CoNLL file and convert it to the list of labels and sentences.

    :param conll_path: CoNLL file path.
    :return: data and labels which have sentences and labels joined with the single space.
    """
    with open(conll_path) as f:
        data = []
        content = f.read()
        docs = [i.strip() for i in content.strip().split('-DOCSTART- -X- -X- O') if i != '']
        for doc in docs:
            doc_sent = []
            pos_sync_tag = []
            labels = []

            #  file content to sentence split
            sentences = doc.strip().split('\n\n')

            if sentences == ['']:
                data.append(([''], [''], ['']))
                continue

            for sent in sentences:
                sentence_data = []
                sentence_labels = []
                sentence_tags = []

                #  sentence string to token level split
                tokens = sent.strip().split('\n')

                # get annotations from token level split
                token_list = [t.split() for t in tokens]

                #  get token and labels from the split
                for split in token_list:
                    sentence_data.append(split[0])
                    sentence_labels.append(split[-1])
                    sentence_tags.append("*".join([split[1], split[2]]))

                doc_sent.append(" ".join(sentence_data))
                labels.append(" ".join(sentence_labels))
                pos_sync_tag.append(" ".join(sentence_tags))

            data.append((doc_sent, pos_sync_tag, labels))

    return data


def filter_by_entity_type(data: List[str], pos_sync_tag: List[str], labels: List[str], ent_type: str) \
        -> (List[str], List[str], List[str]):
    """
    A function to filter data by the entity type.

    :param data: List of sentences to be filtered by the entity type.
    :param pos_sync_tag: List of tags extracted from CoNLL file.
    :param labels: List of labels to be filtered by the entity type.
    :param ent_type: Entity type without IOB2 suffix to filter the data and the labels.
    :return: Filtered data and labels by the entity type.
    """
    filtered_labels = []
    filtered_data = []
    filtered_tags = []
    for indx, label in enumerate(labels):
        if 'B-' + ent_type in label:
            filtered_labels.append(label)
            filtered_data.append(data[indx])
            filtered_tags.append(pos_sync_tag[indx])

    assert len(filtered_data) == len(filtered_labels) == len(filtered_tags), \
        "Length of the labels and the data should be the same."
    return filtered_data, filtered_tags, filtered_labels


def get_sample(k: int, *args: List[Any]) -> List[List[str], ]:
    """
    A function to get sample from passed lists.

    :param k: Number of sample for each list.
    :param args: List to get sample. Every list passed should be exactly same length.
    :return: Sampled lists.
    """
    random_indxs = None
    filtered_arrays = []
    for arr in args:

        if random_indxs is None:

            #  Get first sample from the input and create indx for the given sample
            arr_indx = random.sample(list(enumerate(arr)), k=k)

            arr = []
            random_indxs = []
            for indx, elem in arr_indx:
                arr.append(elem)
                random_indxs.append(indx)
            filtered_arrays.append(arr)

        else:
            filtered_arr = [arr[indx] for indx in random_indxs]
            filtered_arrays.append(filtered_arr)

    return filtered_arrays


def augment_robustness(
        conll_path: str,
        perturbation_map: Optional[Dict[str, float]] = None,
        entity_perturbation_map: Optional[Dict[str, Dict[str, float]]] = None,
        optimized_inplace: bool = False,
        random_state: int = None,
        return_spark: bool = False,
        ending_context: List[str] = None,
        starting_context: List[str] = None,
        regex_pattern: str = "\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])",
        spark: SparkSession = None,
        conll_save_path: str = None,
        print_info: bool = False,
        ignore_warnings: bool = False
):
    """
    This function augments the training set by generating noisy text.
    The resulting dataset includes both the original samples and the noisy ones.
    Noise is applied at the beginning of the training, and it can also be applied after each epoch.

    :param conll_path: CoNLL file path to augment the dataset with given perturbations
    :param perturbation_map: A dictionary of perturbation names and desired perturbation proportions. Options include
    'modify_capitalization_upper': capitalization of the sentences is turned into uppercase
    'modify_capitalization_lower': capitalization of the sentences is turned into lowercase
    'modify_capitalization_title': capitalization of the sentences is turned into title case
    'add_punctuation': special characters at end of each sentence are replaced by other special characters, if no
    special character at the end, one is added
    'strip_punctuation': special characters are removed from the sentences (except if found in numbers, such as '2.5')
    'introduce_typos': typos are introduced in sentences
    'add_contractions': contractions are added where possible (e.g. 'do not' contracted into 'don't')
    'add_context': tokens are added at the beginning and at the end of the sentences.
    'swap_entities': named entities replaced with same entity type with same token count from terminology.
    'swap_cohyponyms': Named entities replaced with co-hyponym from the WordNet database.
    'american_to_british': American English will be changed to British English
    'british_to_american': British English will be changed to American English
    :param entity_perturbation_map: A dictionary of perturbation names and desired perturbation proportions defined
    for each entity class. Options are identical to those passed in `perturbation_map`.
    :param optimized_inplace: Optimization algorithm to run augmentations inplace. This means the modified CoNLL will
    contain the same number of sentences as the original CoNLL.
    :param random_state: A random state to create perturbation in the same samples of data.
    :param return_spark: Return Spark DataFrame instead of CoNLL file.
    :param regex_pattern: Regex pattern to tokenize context and contractions by splitting.
    :param spark: An active spark session to create DataFrame.
    :param conll_save_path: A path to save augmented CoNLL file.
    :param ignore_warnings: Ignore warnings about augmentation, default is False.
    :param print_info: Log information about augmentation, default is False.
    :param starting_context: Pass custom starting context for add_context.
    :param ending_context: Pass custom ending context for add_context.
    :return: Augmented CoNLL file or Spark DataFrame.
    """
    logging.basicConfig()
    logger = logging.getLogger("Robustness Fixing")
    logger.setLevel(level=logging.WARNING)

    if print_info:
        logger.setLevel(level=logging.INFO)

    augmentation_report = dict()
    if ignore_warnings and not print_info:
        logger.setLevel(level=logging.ERROR)

    if conll_save_path is None and not return_spark:
        raise ValueError("conll_save_path or return_spark must be set to return augmented data.")

    if perturbation_map is not None and entity_perturbation_map is not None:
        raise ValueError('You used `entity_perturbation_map` and `perturbation_map`. Please only use one of these '
                         'parameters since they accomplish the same task with a different level of detail.')
    elif perturbation_map is None and entity_perturbation_map is None:
        raise ValueError('You need to use one of `entity_perturbation_map` or `perturbation_map`.')

    docs = conll_reader(conll_path)

    counter = 0
    docs_indx = []

    data = []
    pos_sync_tag = []
    labels = []
    for doc in docs:
        #   keep track of the doc ending indxs
        counter += len(doc[0])
        docs_indx.append(counter)

        #   collect all doc sentences in the same list to process at the same time
        data.extend(doc[0])
        pos_sync_tag.extend(doc[1])
        labels.extend(doc[2])

    logger.info(f' {len(docs_indx)} number of documents from the CoNLL file is read.')
    logger.info(f' {len(data)} number of samples collected from the {len(docs_indx)} docs.')

    augmented_data = []
    augmented_labels = []
    augmented_tags = []

    entities = []
    for sent_labels in labels:
        for label in sent_labels.split():

            if label == 'O':
                continue

            label = label[2:]
            if label not in entities:
                entities.append(label)

    #   Save entity coverage to the augmentation_report
    augmentation_report['entity_coverage'] = dict()
    augmentation_report['augmentation_coverage'] = dict()

    num_instances = dict()
    for ent_type in entities:
        filtered_data, _, _ = filter_by_entity_type(data, pos_sync_tag, labels, ent_type=ent_type)
        num_instances[ent_type] = len(filtered_data)

        entity_coverage_info = round(len(filtered_data) / len(data), 2)
        augmentation_report['entity_coverage'][ent_type] = entity_coverage_info
        logger.info(f' Entity coverage of the data is {len(filtered_data)} '
                    f'out of {len(data)} ({entity_coverage_info}).')

    if random_state:
        random.seed(random_state)
        logger.info(f' Random state is set to {random_state}.')

    def _check_undefined_perturbations(passed_perturbation_map, perturbation_list):
        """
        Checks if the user has passed some invalid keys to the perturbation dictionary.

        :param passed_perturbation_map: The perturbation dictionary passed by the user.
        :param perturbation_list: The list of valid perturbations.
        :return: Raises an error if user passes invalid perturbations
        """
        undefined_perturbations = [k for k in passed_perturbation_map.keys() if k not in perturbation_list]
        if len(undefined_perturbations) > 0:
            for single_undefined_perturbation in undefined_perturbations:
                raise ValueError(
                    f"'{single_undefined_perturbation}' is not a valid perturbation. \nPlease pick a perturbation from "
                    f"the following list:\n{perturbation_list}")

    if perturbation_map is not None:
        utility_perturbation_map = perturbation_map
        _check_undefined_perturbations(passed_perturbation_map=utility_perturbation_map,
                                       perturbation_list=LIST_OF_PERTURBATIONS)
        #   construct perturbation dictionary for use within rest of function
        perturbation_dict = {key: value for (key, value) in
                             tuple((x, dict(zip(entities, [perturbation_map[x]] * len(entities)))) for x in
                                   [i for i in perturbation_map.keys() if i in LIST_OF_PERTURBATIONS])}

    elif entity_perturbation_map is not None:
        utility_perturbation_map = entity_perturbation_map
        _check_undefined_perturbations(passed_perturbation_map=utility_perturbation_map,
                                       perturbation_list=LIST_OF_PERTURBATIONS)
        #   construct perturbation dictionary for use within rest of function
        perturbation_dict = entity_perturbation_map

    if 'swap_entities' in utility_perturbation_map.keys():
        terminology = create_terminology(data, labels)
    else:
        terminology = None

    a2b_dict = A2B_DICT
    b2a_dict = {v: k for k, v in a2b_dict.items()}

    if starting_context is None:
        starting_context = ["Description:", "MEDICAL HISTORY:", "FINDINGS:", "RESULTS: ",
                            "Report: ", "Conclusion is that "]
    starting_context = [re.split(regex_pattern, i) for i in starting_context if i != '']

    if ending_context is None:
        ending_context = ["according to the patient's family", "as stated by the patient",
                          "due to various circumstances", "confirmed by px"]
    ending_context = [re.split(regex_pattern, i) for i in ending_context if i != '']

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
            'ending_context': ending_context,
            'starting_context': starting_context,
        },
        "add_contractions": {},
        "swap_entities": {'terminology': terminology},
        "swap_cohyponyms": {}
    }

    if optimized_inplace:

        optimization_matrix = np.zeros((len(data), len(entities)))
        for x_indx, sent_label in enumerate(labels):
            for y_indx, ent_type in enumerate(entities):
                if ent_type in sent_label:
                    optimization_matrix[x_indx][y_indx] = 1

        for ent_type in entities:

            total = 0
            for proportions in perturbation_dict.values():

                if proportions is None:
                    continue

                if proportions.get(ent_type):
                    total += proportions[ent_type]

            if total > 1:
                raise ValueError(f" Perturbation values for {ent_type} is invalid. "
                                 f"Sum of proportion values should be smaller than 1!")

    for perturb_type, proportions in perturbation_dict.items():

        if proportions is None:
            continue

        if optimized_inplace:

            logger.info(f" Data is being augmented by {perturb_type}, optimized inplace is running.")

            max_possible_perturbation = {k: int(num_instances[k] * v) for k, v in proportions.items()}
            possible_entities = list(max_possible_perturbation.keys())
            num_possible_entity = len(possible_entities)
            sample_total = 0

            inplace_data = []
            inplace_tags = []
            inplace_labels = []

            while num_possible_entity:

                for combination in itertools.combinations(possible_entities, num_possible_entity):

                    entity_condition = np.zeros(len(entities), dtype=int)
                    for ent_type in combination:
                        entity_condition[entities.index(ent_type)] = 1

                    filter_indx, = np.where(np.all(optimization_matrix == entity_condition, axis=1))

                    filtered_data = [data[indx] for indx in filter_indx]
                    filtered_tags = [pos_sync_tag[indx] for indx in filter_indx]
                    filtered_labels = [labels[indx] for indx in filter_indx]

                    optimized_max_sample = min(max_possible_perturbation.values())
                    if optimized_max_sample > len(filter_indx):
                        optimized_max_sample = len(filter_indx)

                    if optimized_max_sample == 0:
                        continue

                    #  get sample from the filtered data
                    sample_indx, sample_data, sample_tags, sample_labels = get_sample(
                        optimized_max_sample, filter_indx, filtered_data, filtered_tags, filtered_labels)
                    sample_total += len(sample_indx)

                    #  apply transformation to the proportion
                    aug_indx, aug_data, aug_tags, aug_labels = PERTURB_FUNC_MAP[perturb_type](
                        sample_data, sample_tags, sample_labels, **perturb_args[perturb_type])

                    inplace_data.extend(aug_data)
                    inplace_tags.extend(aug_tags)
                    inplace_labels.extend(aug_labels)

                    #   drop samples from data and optimization matrix
                    drop_indx = [sample_indx[i] for i in aug_indx]
                    remaining_indx = np.delete(np.arange(len(data)), drop_indx)
                    optimization_matrix = optimization_matrix[remaining_indx]

                    data = [data[indx] for indx in remaining_indx]
                    pos_sync_tag = [pos_sync_tag[indx] for indx in remaining_indx]
                    labels = [labels[indx] for indx in remaining_indx]

                    #   update max_possible_perturbation
                    tmp = dict()
                    for k, v in max_possible_perturbation.items():
                        if v - optimized_max_sample > 0:
                            tmp[k] = v - optimized_max_sample

                    max_possible_perturbation = tmp.copy()
                    possible_entities = list(max_possible_perturbation.keys())

                num_possible_entity -= 1

            logger.info(f" {perturb_type} augmentation is finished!")
            if sample_total == 0:
                logger.warning(
                    f" {perturb_type} could not be applied! No samples remained.")
                logger.warning(
                    f" You may also consider 'optimized_inplace=False' setting to allow multiple copies.")
                continue

            logger.info(f" Totally, {len(inplace_data)} number of samples augmented inplace.")
            if not all(max_possible_perturbation.values()):
                for k, v in max_possible_perturbation.items():
                    if v == 0:
                        continue
                    logger.warning(f" {v} number of '{k}' samples could not augmented with {perturb_type}. "
                                   f"Not enough sample!")
                    logger.warning(
                        f" You may also consider 'optimized_inplace=False' setting to allow multiple copies.")

            augmented_data.append(inplace_data)
            augmented_tags.append(inplace_tags)
            augmented_labels.append(inplace_labels)

            augmentation_coverage_info = round(len(inplace_data) / sample_total, 2)
            logger.info(
                f' Augmentation coverage of {perturb_type} is {len(inplace_data)} '
                f'out of {sample_total} ({augmentation_coverage_info}).')
            augmentation_report['augmentation_coverage'][perturb_type] = augmentation_coverage_info
            if len(inplace_data) < 50:
                logger.warning(f' There is not much samples that conversion needed! Augmentation Coverage is '
                               f'({augmentation_coverage_info})')
                logger.warning(f' You may need to increase proportion value to cover more samples.')

        else:

            for ent_type, proportion in proportions.items():

                logger.info(f" Augmenting {ent_type} with {perturb_type}.")

                #  filter the data by entity types
                filtered_data, filtered_tags, filtered_labels = filter_by_entity_type(
                    data, pos_sync_tag, labels, ent_type)

                if len(filtered_data) == 0:
                    #   0 entity sample in the data, no need to continue
                    logger.error(f' There is no such entity "{ent_type}" in the data. '
                                 f'Augmentation will continue without {ent_type}.')
                    continue

                #  get proportion from the filtered data
                sample_data, sample_tags, sample_labels = get_sample(int(proportion * len(filtered_data)),
                                                                     filtered_data, filtered_tags, filtered_labels)

                #  apply transformation to the proportion
                _, aug_data, aug_tags, aug_labels = PERTURB_FUNC_MAP[perturb_type](
                    sample_data, sample_tags, sample_labels, **perturb_args[perturb_type])

                augmentation_coverage_info = round(len(aug_data) / len(sample_data), 2)
                logger.info(f' {perturb_type} is applied with {len(aug_data)} number of samples ')
                logger.info(
                    f' Augmentation coverage of the data is {len(aug_data)} '
                    f'out of {len(sample_data)} ({augmentation_coverage_info}).\n')
                augmentation_report['augmentation_coverage'][perturb_type] = augmentation_coverage_info
                entity_coverage_info = augmentation_report['entity_coverage'][ent_type]

                if augmentation_coverage_info < 0.1 and proportion < 0.7:
                    logger.warning(f' There is not much samples that conversion needed! Augmentation Coverage is '
                                   f'({augmentation_coverage_info})')
                    logger.warning(f' You may need to increase proportion value to cover more samples.')

                elif entity_coverage_info * proportion < 0.05:
                    logger.warning(f' Proportion desired for {perturb_type}: ({proportion}). '
                                   f'Entity Coverage of {ent_type}: ({entity_coverage_info})')
                    logger.warning(
                        f' With desired proportion value {proportion} and '
                        f'entity coverage: ({entity_coverage_info}), '
                        f' only {len(sample_data)} number of augmented samples added to data. '
                        f'You may need to increase proportion value to add more samples!')

                augmented_data.append(aug_data)
                augmented_labels.append(aug_labels)
                augmented_tags.append(aug_tags)

    number_of_sentences = len(data)
    for aug_data, aug_tags, aug_labels in zip(augmented_data, augmented_tags, augmented_labels):
        data.extend(aug_data)
        labels.extend(aug_labels)
        pos_sync_tag.extend(aug_tags)
    num_of_augmentation_samples = len(data) - number_of_sentences

    if return_spark:
        assert spark is not None, 'It is required to pass an active SparkSession in order to return Spark DataFrame'
        return create_dataframe(spark, data, pos_sync_tag, labels)

    try:
        with open(conll_save_path, 'w') as f:
            try:
                counter = 0
                f.write("-DOCSTART- -X- -X- O\n")
                for indx, (sentence, sent_tags, sent_labels) in enumerate(zip(data, pos_sync_tag, labels)):

                    if counter < len(docs_indx) and indx == docs_indx[counter]:
                        f.write("\n-DOCSTART- -X- -X- O\n")
                        counter += 1

                    f.write("\n")
                    for token, tags, label in zip(sentence.split(), sent_tags.split(), sent_labels.split()):
                        f.write(f"{token} {' '.join(tags.split('*'))} {label}\n")

            except (IOError, OSError) as e:
                print(f"Error while writing to the {conll_save_path}.")
                print(e)

    except (FileNotFoundError, PermissionError, OSError) as e:
        print(f"Error while opening the {conll_save_path}.")
        print(e)

    augmentation_report['number_of_sentences'] = number_of_sentences
    augmentation_report['number_of_augmented_samples'] = num_of_augmentation_samples

    entity_coverage_after_aug = dict()
    for ent_type in entities:
        filtered_data, _, _ = filter_by_entity_type(data, pos_sync_tag, labels, ent_type=ent_type)
        entity_coverage_after_aug[ent_type] = round(len(filtered_data) / len(data), 2)

    augmentation_report['entity_coverage_after_augmentation'] = entity_coverage_after_aug

    return augmentation_report


def test_and_augment_robustness(spark: SparkSession, pipeline_model: PipelineModel, test_file_path: str,
                                conll_path_to_augment: str, conll_save_path: str = None,
                                metric_type: Optional[str] = 'flex',
                                metrics_output_format: str = 'dictionary',
                                log_path: str = 'robustness_test_results.json',
                                noise_prob: float = 0.5,
                                sample_sentence_count: int = None,
                                test: Optional[List[str]] = None,
                                starting_context: Optional[List[str]] = None,
                                ending_context: Optional[List[str]] = None,
                                optimized_inplace: bool = False,
                                random_state=None,
                                return_spark: bool = False,
                                regex_pattern: str = "\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])",
                                print_info: bool = False,
                                ignore_warnings: bool = False):
    """
    One-liner to test and augment CoNLL for robustness.

    :param spark: An active SparkSession to create Spark DataFrame
    :param pipeline_model: PipelineModel with document assembler, sentence detector, tokenizer, word embeddings if
    applicable, NER model with 'ner' output name, NER converter with 'ner_chunk' output name
    :param test_file_path: Path to file to test robustness. Can be .txt or .conll file in CoNLL format or
    .csv file with just one column (text) with series of test samples.
    :param conll_path_to_augment: Path to the CoNLL file that will be used for augmentation purposes.
    :param conll_save_path: Path to save augmented CoNLL file.
    :param metric_type: 'strict' calculates metrics for IOB2 format, 'flex' (default) calculates for IO format
    :param metrics_output_format: 'dictionary' to get a dictionary report, 'dataframe' to get a dataframe report
    :param log_path: Path to log file, False to avoid saving test results. Default 'robustness_test_results.json'
    :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
    :param sample_sentence_count: Number of sentence that will be sampled from the test_data.
    :param test: type of robustness test implemented by the function, options include
    'modify_capitalization_upper': capitalization of the test set is turned into uppercase
    'modify_capitalization_lower': capitalization of the test set is turned into lowercase
    'modify_capitalization_title': capitalization of the test set is turned into title case
    'add_punctuation': special characters at end of each sentence are replaced by other special characters, if no
    special character at the end, one is added
    'strip_punctuation': special characters are removed from the sentences (except if found in numbers, such as '2.5')
    'introduce_typos': typos are introduced in sentences
    'add_contractions': contractions are added where possible (e.g. 'do not' contracted into 'don't')
    'add_context': tokens are added at the beginning and at the end of the sentences.
    'swap_entities': named entities replaced with same entity type with same token count from terminology.
    'swap_cohyponyms': Named entities replaced with co-hyponym from the WordNet database.
    'american_to_british': American English will be changed to British English
    'british_to_american': British English will be changed to American English
    :param starting_context: list of context tokens to add to beginning when running the 'add_context' test
    :param ending_context: list of context tokens to add to end when running the 'add_context' test
    :param optimized_inplace: Optimization algorithm to run augmentations inplace. This means the modified CoNLL will
    contain the same number of sentences as the original CoNLL.
    :param random_state: A random state to create perturbation in the same samples of data.
    :param return_spark: Return Spark DataFrame instead of CoNLL file.
    :param regex_pattern: Regex pattern to tokenize context and contractions by splitting.
    :param print_info: Log information about augmentation, default is False.
    :param ignore_warnings: Ignore warnings about augmentation, default is False.
    """
    test_results = test_robustness(spark=spark, pipeline_model=pipeline_model, test_file_path=test_file_path,
                                   metric_type=metric_type, metrics_output_format=metrics_output_format,
                                   log_path=log_path, noise_prob=noise_prob,
                                   sample_sentence_count=sample_sentence_count, test=test,
                                   starting_context=starting_context, ending_context=ending_context)

    suggestions = suggest_perturbations(test_results)

    if suggestions == {}:
        print("Test metrics all have over 0.9 f1-score for all perturbations. Perturbations will not be applied.")

    else:
        augment_robustness(conll_path=conll_path_to_augment, entity_perturbation_map=suggestions,
                           optimized_inplace=optimized_inplace, random_state=random_state, return_spark=return_spark,
                           ending_context=ending_context, starting_context=starting_context,
                           regex_pattern=regex_pattern, spark=spark, conll_save_path=conll_save_path,
                           print_info=print_info, ignore_warnings=ignore_warnings)
