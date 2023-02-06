import json
import pandas as pd

from typing import Dict, Any

from sparknlp.base import *
from sparknlp_jsl.annotator import *
from sparknlp.training import CoNLL
from sparknlp_jsl.eval import NerDLMetrics

from pyspark.sql.types import *
from pyspark.sql.functions import col, udf
from pyspark.sql import SparkSession, DataFrame, Row


def init_medical_gender_classifier() -> Pipeline:
    """
    This function initialize BioBERT based gender classification model in Spark NLP
    :return: Gender classification pipeline
    """
    document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

    tokenizer = Tokenizer().setInputCols(['document']).setOutputCol('token')

    biobert_embeddings = BertEmbeddings().pretrained('biobert_pubmed_base_cased') \
        .setInputCols(["document", 'token']) \
        .setOutputCol("bert_embeddings")

    sentence_embeddings = SentenceEmbeddings() \
        .setInputCols(["document", "bert_embeddings"]) \
        .setOutputCol("sentence_bert_embeddings") \
        .setPoolingStrategy("AVERAGE")

    gender_classifier = ClassifierDLModel.pretrained('classifierdl_gender_biobert', 'en', 'clinical/models') \
        .setInputCols(["document", "sentence_bert_embeddings"]) \
        .setOutputCol("gender")

    nlp_pipeline = Pipeline(
        stages=[document_assembler, tokenizer, biobert_embeddings, sentence_embeddings, gender_classifier])

    return nlp_pipeline


class RuleBasedClassifier:

    def transform(self, dataframe: DataFrame):
        """
        Rule based gender classifier that counts number of gender-related words in the text.
        :param dataframe: Spark DataFrame contains CoNLL data.
        :return: Transformed dataframe with new gender column.
        """
        @udf(returnType=ArrayType(StructType([
              StructField("result", StringType())
            ])))
        def apply_regex_match(column):
            female_entities = ['she', 'her', 'hers', 'herself', 'girl', 'girls', 'woman', 'women',
                               'madam', 'madame', 'lady', 'miss', 'mrs', 'female', 'breast', 
                               'ovary', 'ovarian', 'vagina']
            male_entities = ['he', 'his', 'him', 'himself', 'boy', 'man', 'men', 'sir', 'gentleman', 'mr',
                             'male', 'prostate', 'testicle', 'testicular', 'penis']
            female_count = 0
            male_count = 0
            for row in column:
                if row.result.lower() in female_entities:
                    female_count += 1
                elif row.result.lower() in male_entities:
                    male_count += 1

            #   result type should be compatible with regular classifier results.
            #   thats why we are not returning directly string results
            if female_count > male_count:
                return [Row(**{'result': 'Female'})]
            elif male_count > female_count:
                return [Row(**{'result': 'Male'})]
            else:
                return [Row(**{'result': 'Unknown'})]

        dataframe = dataframe.withColumn(
            'gender',
            apply_regex_match(col("token"))
        )
        return dataframe


def test_gender_bias(spark: SparkSession, ner_pipeline: PipelineModel,
                     test_conll: str, classifier_pipeline: PipelineModel = None,
                     training_conll: str = None,
                     log_path: str = 'gender_bias_results.json',
                     is_medical=True, explode_sentences=False) -> Dict[str, Any]:
    """
    This function evaluate NerDL model against gender bias. It compares the performance of the model depending
    on the gender of the test set and describes the gender distribution of the entities and extractions from a
    manually annotated dataset. Calculations are document based on CoNLL file, thus each sample should be separated
    by regular ConLL doc string. Pipeline that contains NerDL model should be passed and it should be compatible with
    the CoNLL data, e.g. both have same entities in IOB format.
    :param spark: An active spark session.
    :param ner_pipeline: PipelineModel that predicts NER in CoNLL data.
    :param test_conll: Path to NER test dataset in CoNLL format.
    :param classifier_pipeline: SparkNLP PipelineModel that classify gender.
    :param training_conll: Path to NER training dataset in CoNLL format.
    :param log_path: Path to log file, False to avoid saving test results. Default 'gender_bias_results.json'
    :param is_medical: Whether gender classifier and NER model is medical or not.
    :param explode_sentences: Explode sentences in CoNLL data while loading.
    If explode_sentences passed as True, NER model should process sentences, otherwise documents.
    Exploding will resulted in single document instance for each doc string in the data.
    :return: Test results dictionary with following keys;
        'doc_amounts': Number of documents for each gender in the training and test set.
        'training_set_gender_distribution': Gender distribution in the training dataset,
        calculated using BioBERT based gender classification model
        'test_set_gender_distribution': Gender distribution in the test dataset,
        calculated using BioBERT based gender classification model
        'test_set_metrics': Gender specific test metrics for test set.
        Calculations are based on NER model trained on the given train set
    """
    outcome = {}
    doc_amounts = []

    if not classifier_pipeline:

        if is_medical:
            classifier_pipeline = init_medical_gender_classifier() \
                .fit(spark.createDataFrame([['']]).toDF("text"))
        else:
            classifier_pipeline = RuleBasedClassifier()

    def get_gender_distribution(classifier_results_):

        label_distribution = {
            'Female': dict(),
            'Male': dict(),
            'Unknown': dict()
        }

        for _, row in classifier_results_.iterrows():

            gender = row['gender']

            for label in row['label']:

                label = label.result
                if label == 'O':
                    continue

                label = label.split('-')[-1]
                if label_distribution[gender].get(label, None):
                    label_distribution[gender][label] += 1
                else:
                    label_distribution[gender][label] = 1

        df = pd.DataFrame.from_dict(label_distribution).reset_index().rename({'index': 'Label'}, axis=1)
        return df

    if training_conll:

        training_set = CoNLL(explodeSentences=explode_sentences).readDataset(spark, training_conll)
        classifier_results = classifier_pipeline.transform(training_set)

        classified_training_set = classifier_results.select('text', 'label', 'gender').toPandas()
        classified_training_set['gender'] = classified_training_set['gender'].apply(lambda x: x[0]['result'])

        num_female_samples = len(classified_training_set[classified_training_set['gender'] == 'Female'])
        num_male_samples = len(classified_training_set[classified_training_set['gender'] == 'Male'])
        num_unknown_samples = len(classified_training_set[classified_training_set['gender'] == 'Unknown'])

        female_training_docs = {'gender': 'female', 'data': 'training', 'doc_amount': num_female_samples}
        male_training_docs = {'gender': 'male', 'data': 'training', 'doc_amount': num_male_samples}
        unknown_training_docs = {'gender': 'unknown', 'data': 'training', 'doc_amount': num_unknown_samples}

        doc_amounts.append(female_training_docs)
        doc_amounts.append(male_training_docs)
        doc_amounts.append(unknown_training_docs)

        outcome['training_set_gender_distribution'] = get_gender_distribution(classified_training_set)

    test_set = CoNLL(explodeSentences=explode_sentences).readDataset(spark, test_conll)

    ner_results = ner_pipeline.transform(test_set)

    classifier_results = classifier_pipeline.transform(ner_results)

    classified_test_set = classifier_results.select('text', 'label', 'gender').toPandas()
    classified_test_set['gender'] = classified_test_set['gender'].apply(lambda x: x[0]['result'])
    outcome['test_set_gender_distribution'] = get_gender_distribution(classified_test_set)

    num_female_samples = len(classified_test_set[classified_test_set['gender'] == 'Female'])
    num_male_samples = len(classified_test_set[classified_test_set['gender'] == 'Male'])
    num_unk_samples = len(classified_test_set[classified_test_set['gender'] == 'Unknown'])

    female_test_docs = {'gender': 'female', 'data': 'test', 'doc_amount': num_female_samples}
    male_test_docs = {'gender': 'male', 'data': 'test', 'doc_amount': num_male_samples}
    unknown_test_docs = {'gender': 'unknown', 'data': 'test', 'doc_amount': num_unk_samples}

    doc_amounts.append(female_test_docs)
    doc_amounts.append(male_test_docs)
    doc_amounts.append(unknown_test_docs)

    female_test_set = classifier_results.filter(F.array_contains(F.col('gender.result'), 'Female'))
    male_test_set = classifier_results.filter(F.array_contains(F.col('gender.result'), 'Male'))
    unknown_test_set = classifier_results.filter(F.array_contains(F.col('gender.result'), 'Unknown'))

    evaluate = NerDLMetrics(mode="partial_chunk_per_token")

    female_result = evaluate.computeMetricsFromDF(female_test_set.select("label", "ner"),
                                                  prediction_col="ner",
                                                  label_col="label", drop_o=True).toPandas()
    male_result = evaluate.computeMetricsFromDF(male_test_set.select("label", "ner"),
                                                prediction_col="ner",
                                                label_col="label", drop_o=True).toPandas()
    unknown_result = evaluate.computeMetricsFromDF(unknown_test_set.select("label", "ner"),
                                                   prediction_col="ner",
                                                   label_col="label", drop_o=True).toPandas()

    female_result['gender'] = 'female'
    male_result['gender'] = 'male'
    unknown_result['gender'] = 'unknown'
    outcome['test_set_metrics'] = pd.concat([female_result, male_result, unknown_result], ignore_index=True)
    outcome['doc_amounts'] = doc_amounts

    if log_path:

        test_results = outcome.copy()
        try:
            train_distribution = test_results['training_set_gender_distribution'].set_index('Label').to_dict('index')
            test_results['training_set_gender_distribution'] = train_distribution
        except:
            pass

        test_distribution = test_results['test_set_gender_distribution'].set_index('Label').to_dict('index')
        test_results['test_set_gender_distribution'] = test_distribution

        test_metrics = dict()
        for gender, group in test_results['test_set_metrics'].groupby('gender'):
            group = group.drop('gender', axis=1)
            group_dict = group.to_dict('list')
            test_metrics[gender] = group_dict
        test_results['test_set_metrics'] = test_metrics

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

    return outcome
