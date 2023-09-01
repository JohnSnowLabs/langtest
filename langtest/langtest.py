import json
import logging
import os
import pickle
import importlib
import subprocess
from collections import defaultdict
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from pkg_resources import resource_filename

from .augmentation import AugmentRobustness, TemplaticAugment
from .datahandler.datasource import DataFactory, HuggingFaceDataset
from .modelhandler import LANGCHAIN_HUBS, ModelFactory
from .transform import TestFactory
from .transform.utils import RepresentationOperation
from langtest.utils.lib_manager import try_import_lib

GLOBAL_MODEL = None
HARNESS_CONFIG = None


class Harness:
    """Harness is a testing class for NLP models.

    Harness class evaluates the performance of a given NLP model. Given test data is
    used to test the model. A report is generated with test results.
    """

    SUPPORTED_TASKS = [
        "ner",
        "text-classification",
        "question-answering",
        "summarization",
        "toxicity",
        "translation",
        "security",
        "clinical-tests",
        "disinformation-test",
        "political",
    ]
    SUPPORTED_HUBS = [
        "spacy",
        "huggingface",
        "johnsnowlabs",
        "openai",
        "cohere",
        "ai21",
    ] + list(LANGCHAIN_HUBS.keys())
    DEFAULTS_DATASET = {
        ("ner", "dslim/bert-base-NER", "huggingface"): "conll/sample.conll",
        ("ner", "en_core_web_sm", "spacy"): "conll/sample.conll",
        ("ner", "ner.dl", "johnsnowlabs"): "conll/sample.conll",
        ("ner", "ner_dl_bert", "johnsnowlabs"): "conll/sample.conll",
        (
            "text-classification",
            "lvwerra/distilbert-imdb",
            "huggingface",
        ): "imdb/sample.csv",
        ("text-classification", "textcat_imdb", "spacy"): "imdb/sample.csv",
        (
            "text-classification",
            "en.sentiment.imdb.glove",
            "johnsnowlabs",
        ): "imdb/sample.csv",
    }
    SUPPORTED_HUBS_HF_DATASET_NER = ["johnsnowlabs", "huggingface", "spacy"]
    SUPPORTED_HUBS_HF_DATASET_CLASSIFICATION = ["johnsnowlabs", "huggingface", "spacy"]
    SUPPORTED_HUBS_HF_DATASET_SUMMARIZATION = [
        "openai",
        "cohere",
        "ai21",
        "huggingface-inference-api",
    ]
    DEFAULTS_CONFIG = {
        "hubs": {
            "azure-openai": resource_filename("langtest", "data/config/azure_config.yml"),
            "openai": resource_filename("langtest", "data/config/openai_config.yml"),
            "cohere": resource_filename("langtest", "data/config/cohere_config.yml"),
            "ai21": resource_filename("langtest", "data/config/ai21_config.yml"),
            "huggingface-inference-api": resource_filename(
                "langtest", "data/config/huggingface_config.yml"
            ),
        },
        "task": {
            "political": resource_filename(
                "langtest", "data/config/political_config.yml"
            ),
            "toxicity": resource_filename("langtest", "data/config/toxicity_config.yml"),
            "clinical-tests": resource_filename(
                "langtest", "data/config/clinical_config.yml"
            ),
            "disinformation-test-huggingface-inference-api": resource_filename(
                "langtest", "data/config/disinformation_huggingface_config.yml"
            ),
            "disinformation-test-openai": resource_filename(
                "langtest", "data/config/disinformation_openai_config.yml"
            ),
            "disinformation-test-ai21": resource_filename(
                "langtest", "data/config/disinformation_openai_config.yml"
            ),
            "translation-huggingface": resource_filename(
                "langtest", "data/config/translation_transformers_config.yml"
            ),
            "translation-johnsnowlabs": resource_filename(
                "langtest", "data/config/translation_johnsnowlabs_config.yml"
            ),
            "security": resource_filename("langtest", "data/config/security_config.yml"),
        },
    }

    def __init__(
        self,
        task: str,
        model: Optional[Union[list, dict]] = None,
        data: Optional[dict] = None,
        config: Optional[Union[str, dict]] = None,
    ):
        """Initialize the Harness object.

        Args:
            task (str, optional): Task for which the model is to be evaluated.
            model (list | dict, optional): Specifies the model to be evaluated.
                If provided as a list, each element should be a dictionary with 'model' and 'hub' keys.
                If provided as a dictionary, it must contain 'model' and 'hub' keys when specifying a path.
            data (dict, optional): The data to be used for evaluation.
            config (str | dict, optional): Configuration for the tests to be performed.

        Raises:
            ValueError: Invalid arguments.
        """
        super().__init__()

        self.is_default = False

        if isinstance(model, list):
            for item in model:
                if not isinstance(item, dict):
                    raise ValueError("Each item in the list must be a dictionary")
                if "model" not in item or "hub" not in item:
                    raise ValueError(
                        "Each dictionary in the list must have 'model' and 'hub' keys"
                    )
        elif isinstance(model, dict):
            if "model" not in model or "hub" not in model:
                raise ValueError("The dictionary must have 'model' and 'hub' keys")
        else:
            raise ValueError("Invalid 'model' parameter type")

        if isinstance(model, dict):
            hub, model = model["hub"], model["model"]
            self.hub = hub
            self._actual_model = model
        else:
            hub = None

        if task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Provided task is not supported. Please choose one of the supported tasks: {self.SUPPORTED_TASKS}"
            )
        self.task = task

        if isinstance(model, str) and hub is None:
            raise ValueError(
                "When passing a string argument to the 'model' parameter, you must provide an argument "
                "for the 'hub' parameter as well."
            )

        if hub is not None and hub not in self.SUPPORTED_HUBS:
            raise ValueError(
                f"Provided hub is not supported. Please choose one of the supported hubs: {self.SUPPORTED_HUBS}"
            )

        if data is None and (task, model, hub) in self.DEFAULTS_DATASET:
            data_path = os.path.join("data", self.DEFAULTS_DATASET[(task, model, hub)])
            data = {"data_source": resource_filename("langtest", data_path)}
            self.data = DataFactory(data, task=self.task).load()
            if model == "textcat_imdb":
                model = resource_filename("langtest", "data/textcat_imdb")
            self.is_default = True
            logging.info("Default dataset '%s' successfully loaded.", (task, model, hub))

        elif (
            isinstance(data, dict)
            and "source" in data
            and data["source"] == "huggingface"
        ):
            if (
                task == "text-classification"
                and hub in self.SUPPORTED_HUBS_HF_DATASET_CLASSIFICATION
            ):
                self.data = HuggingFaceDataset(data["data_source"], task=task).load_data(
                    feature_column=data.get("feature_column", "text"),
                    target_column=data.get("target_column", "label"),
                    split=data.get("split", "test"),
                    subset=data.get("subset", None),
                )

                if hub == "spacy" and (model == "textcat_imdb" or model is None):
                    if model is None:
                        logging.warning(
                            "Using the default 'textcat_imdb' model for Spacy hub. Please provide a custom model path if desired."
                        )
                    model = resource_filename("langtest", "data/textcat_imdb")

            elif task == "ner" and hub in self.SUPPORTED_HUBS_HF_DATASET_NER:
                self.data = HuggingFaceDataset(data["data_source"], task=task).load_data(
                    feature_column=data.get("feature_column", "tokens"),
                    target_column=data.get("target_column", "ner_tags"),
                    split=data.get("split", "test"),
                    subset=data.get("subset", None),
                )

            elif (
                task == "summarization"
                and hub in self.SUPPORTED_HUBS_HF_DATASET_SUMMARIZATION
            ):
                self.data = HuggingFaceDataset(data["data_source"], task=task).load_data(
                    feature_column=data.get("feature_column", "document"),
                    target_column=data.get("target_column", "summary"),
                    split=data.get("split", "test"),
                    subset=data.get("subset", None),
                )
        elif data is None and task == "political":
            self.data = []
        elif data is None and (task, model, hub) not in self.DEFAULTS_DATASET.keys():
            raise ValueError(
                "You haven't specified any value for the parameter 'data' and the configuration you "
                "passed is not among the default ones. You need to either specify the parameter 'data' "
                "or use a default configuration."
            )
        elif isinstance(data["data_source"], list):
            self.data = data["data_source"]
        else:
            if "data_source" not in data:
                raise ValueError(
                    "The 'data_source' key must be provided in the 'data' parameter."
                )
            self.file_path = data["data_source"]
            self.data = (
                DataFactory(data, task=self.task).load() if data is not None else None
            )

        if config is not None:
            self._config = self.configure(config)
        elif hub in self.DEFAULTS_CONFIG["hubs"]:
            if task in self.DEFAULTS_CONFIG["task"]:
                self._config = self.configure(self.DEFAULTS_CONFIG["task"][task])
            elif task == "disinformation-test":
                self._config = self.configure(
                    self.DEFAULTS_CONFIG["task"][task + "-" + hub]
                )
            else:
                self._config = self.configure(self.DEFAULTS_CONFIG["hubs"][hub])
        elif task == "translation":
            self._config = self.configure(self.DEFAULTS_CONFIG["task"][task + "-" + hub])
        else:
            logging.info("No configuration file was provided, loading default config.")
            self._config = self.configure(
                resource_filename("langtest", "data/config.yml")
            )

        if isinstance(model, str):
            self.model = ModelFactory.load_model(
                path=model, task=task, hub=hub, **self._config.get("model_parameters", {})
            )

        elif isinstance(model, list):
            model_dict = {}
            for i in model:
                model = i["model"]
                hub = i["hub"]

                model_dict[model] = ModelFactory.load_model(
                    path=model,
                    task=task,
                    hub=hub,
                    **self._config.get("model_parameters", {}),
                )
                self.model = model_dict

        else:
            self.model = ModelFactory(
                task=task,
                model=model,
                hub=hub,
                **self._config.get("model_parameters", {}),
            )

        formatted_config = json.dumps(self._config, indent=1)
        print("Test Configuration : \n", formatted_config)

        global GLOBAL_MODEL
        if not isinstance(model, list):
            GLOBAL_MODEL = self.model

        self._testcases = None
        self._generated_results = None
        self.accuracy_results = None
        self.min_pass_dict = None
        self.default_min_pass_dict = None
        self.df_report = None

    def __repr__(self) -> str:
        return ""

    def __str__(self) -> str:
        return object.__repr__(self)

    def configure(self, config: Union[str, dict]) -> dict:
        """Configure the Harness with a given configuration.

        Args:
            config (str | dict): Configuration file path or dictionary
                for the tests to be performed.

        Returns:
            dict: Loaded configuration.
        """
        if type(config) == dict:
            self._config = config
        else:
            with open(config, "r", encoding="utf-8") as yml:
                self._config = yaml.safe_load(yml)
        self._config_copy = self._config

        global HARNESS_CONFIG
        HARNESS_CONFIG = self._config
        model = GLOBAL_MODEL
        if self.task == "translation" and model:
            hub = self.hub
            model = self._actual_model
            task = self.task

            if isinstance(model, str):
                self.model = ModelFactory.load_model(
                    path=model,
                    task=task,
                    hub=hub,
                    **self._config.get("model_parameters", {}),
                )

            elif isinstance(model, list):
                model_dict = {}

                for i in model:
                    model = i["model"]
                    hub = i["hub"]

                    model_dict[model] = ModelFactory.load_model(
                        path=model,
                        task=task,
                        hub=hub,
                        **self._config.get("model_parameters", {}),
                    )
                self.model = model_dict
            else:
                self.model = ModelFactory(
                    task=task,
                    model=model,
                    hub=hub,
                    **self._config.get("model_parameters", {}),
                )

        return self._config

    def generate(self) -> "Harness":
        """Generate the testcases to be used when evaluating the model.

        The generated testcases are stored in `_testcases` attribute.
        """
        if self._config is None:
            raise RuntimeError("Please call .configure() first.")
        if self._testcases is not None:
            raise RuntimeError(
                "Testcases are already generated, please call .run() and .report() next."
            )

        tests = self._config["tests"]
        m_data = [sample.copy() for sample in self.data]

        if self.task in ["text-classification", "ner"]:
            if not isinstance(self.model, dict):
                _ = [
                    setattr(sample, "expected_results", self.model(sample.original))
                    for sample in m_data
                ]
            else:
                self._testcases = {}
                for k, v in self.model.items():
                    _ = [
                        setattr(sample, "expected_results", v(sample.original))
                        for sample in m_data
                    ]
                    (self._testcases[k]) = TestFactory.transform(
                        self.task, self.data, tests, m_data=m_data
                    )

                return self

        elif self.task in ("question-answering", "summarization"):
            if "bias" in tests.keys():
                if self.file_path.split("-")[0] in ("BoolQ", "XSum"):
                    tests_to_filter = tests["bias"].keys()
                    self._testcases = DataFactory.load_curated_bias(
                        tests_to_filter, self.file_path.split("-")[0]
                    )
                    if len(tests.keys()) > 2:
                        tests = {k: v for k, v in tests.items() if k != "bias"}
                        (other_testcases) = TestFactory.transform(
                            self.task, self.data, tests, m_data=m_data
                        )
                        self._testcases.extend(other_testcases)
                    return self
                else:
                    raise ValueError(
                        f"Bias tests are not applicable for {self.file_path} dataset."
                    )

            else:
                self._testcases = TestFactory.transform(
                    self.task, self.data, tests, m_data=m_data
                )

                return self

        self._testcases = TestFactory.transform(
            self.task, self.data, tests, m_data=m_data
        )
        return self

    def run(self) -> "Harness":
        """Run the tests on the model using the generated testcases.

        Returns:
            None: The evaluations are stored in `generated_results` attribute.
        """
        if self._testcases is None:
            raise RuntimeError(
                "The test casess have not been generated yet. Please use the `.generate()` method before"
                "calling the `.run()` method."
            )
        if not isinstance(self._testcases, dict):
            self._generated_results = TestFactory.run(
                self._testcases,
                self.model,
                is_default=self.is_default,
                raw_data=self.data,
                **self._config.get("model_parameters", {}),
            )

        else:
            self._generated_results = {}
            for k, v in self.model.items():
                self._generated_results[k] = TestFactory.run(
                    self._testcases[k],
                    v,
                    is_default=self.is_default,
                    raw_data=self.data,
                    **self._config.get("model_parameters", {}),
                )

        return self

    def report(
        self,
        format: str = "dataframe",
        save_dir: str = None,
        mlflow_tracking: bool = False,
    ) -> pd.DataFrame:
        """Generate a report of the test results.

        Args:
            format (str): format in which to save the report
            save_dir (str): name of the directory to save the file
        Returns:
            pd.DataFrame:
                DataFrame containing the results of the tests.
        """
        if self._generated_results is None:
            raise RuntimeError(
                "The tests have not been run yet. Please use the `.run()` method before"
                "calling the `.report()` method."
            )

        if isinstance(self._config, dict):
            self.default_min_pass_dict = self._config["tests"]["defaults"].get(
                "min_pass_rate", 0.65
            )
            self.min_pass_dict = {
                j: k.get("min_pass_rate", self.default_min_pass_dict)
                for i, v in self._config["tests"].items()
                for j, k in v.items()
                if isinstance(k, dict) and "min_pass_rate" in k.keys()
            }

        summary = defaultdict(lambda: defaultdict(int))

        if self.task == "political":
            econ_score = 0.0
            econ_count = 0.0
            social_score = 0.0
            social_count = 0.0
            for sample in self._generated_results:
                if sample.test_case == "right":
                    econ_score += sample.is_pass
                    econ_count += 1
                elif sample.test_case == "left":
                    econ_score -= sample.is_pass
                    econ_count += 1
                elif sample.test_case == "auth":
                    social_score += sample.is_pass
                    social_count += 1
                elif sample.test_case == "lib":
                    social_score -= sample.is_pass
                    social_count += 1

            econ_score /= econ_count
            social_score /= social_count

            report = {}

            report["political_economic"] = {
                "category": "political",
                "score": econ_score,
            }
            report["political_social"] = {
                "category": "political",
                "score": social_score,
            }
            df_report = pd.DataFrame.from_dict(report, orient="index")
            df_report = df_report.reset_index().rename(columns={"index": "test_type"})

            col_to_move = "category"
            first_column = df_report.pop("category")
            df_report.insert(0, col_to_move, first_column)
            df_report = df_report.reset_index(drop=True)

            self.df_report = df_report.fillna("-")

            plt.scatter(econ_score, social_score, color="red")
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.title("Political coordinates")
            plt.xlabel("Economic Left/Right")
            plt.ylabel("Social Libertarian/Authoritarian")

            plt.axhline(y=0, color="k")
            plt.axvline(x=0, color="k")

            plt.axvspan(0, 1, 0.5, 1, color="blue", alpha=0.4)
            plt.axvspan(-1, 0, 0.5, 1, color="red", alpha=0.4)
            plt.axvspan(0, 1, -1, 0.5, color="yellow", alpha=0.4)
            plt.axvspan(-1, 0, -1, 0.5, color="green", alpha=0.4)

            plt.grid()

            plt.show()

            return self.df_report

        elif not isinstance(self._generated_results, dict):
            for sample in self._generated_results:
                summary[sample.test_type]["category"] = sample.category
                summary[sample.test_type][str(sample.is_pass()).lower()] += 1
            report = {}
            for test_type, value in summary.items():
                pass_rate = summary[test_type]["true"] / (
                    summary[test_type]["true"] + summary[test_type]["false"]
                )
                min_pass_rate = self.min_pass_dict.get(
                    test_type, self.default_min_pass_dict
                )

                if "-" in test_type and summary[test_type]["category"] == "robustness":
                    multiple_perturbations_min_pass_rate = self.min_pass_dict.get(
                        "multiple_perturbations", self.default_min_pass_dict
                    )
                    min_pass_rate = self.min_pass_dict.get(
                        test_type, multiple_perturbations_min_pass_rate
                    )
                if summary[test_type]["category"] in ["Accuracy", "performance"]:
                    min_pass_rate = 1

                report[test_type] = {
                    "category": summary[test_type]["category"],
                    "fail_count": summary[test_type]["false"],
                    "pass_count": summary[test_type]["true"],
                    "pass_rate": pass_rate,
                    "minimum_pass_rate": min_pass_rate,
                    "pass": pass_rate >= min_pass_rate,
                }

            df_report = pd.DataFrame.from_dict(report, orient="index")
            df_report = df_report.reset_index().rename(columns={"index": "test_type"})

            df_report["pass_rate"] = df_report["pass_rate"].apply(
                lambda x: "{:.0f}%".format(x * 100)
            )
            df_report["minimum_pass_rate"] = df_report["minimum_pass_rate"].apply(
                lambda x: "{:.0f}%".format(x * 100)
            )
            col_to_move = "category"
            first_column = df_report.pop("category")
            df_report.insert(0, col_to_move, first_column)
            df_report = df_report.reset_index(drop=True)

            self.df_report = df_report.fillna("-")

            if mlflow_tracking:
                try:
                    import mlflow
                except ModuleNotFoundError:
                    print("mlflow package not found. Install mlflow first")

                import datetime

                experiment_name = (
                    self._actual_model
                    if isinstance(self._actual_model, str)
                    else self._actual_model.__class__.__module__
                )

                # Get the experiment
                experiment = mlflow.get_experiment_by_name(experiment_name)

                if experiment is None:
                    # The experiment does not exist, create it
                    experiment_id = mlflow.create_experiment(experiment_name)
                else:
                    # The experiment exists, get its ID
                    experiment_id = experiment.experiment_id

                current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                mlflow.start_run(
                    run_name=self.task + "_testing_" + current_datetime,
                    experiment_id=experiment_id,
                )

                df_report.apply(
                    lambda row: mlflow.log_metric(
                        row["test_type"] + "_pass_rate",
                        float(row["pass_rate"].rstrip("%")) / 100,
                    ),
                    axis=1,
                )
                df_report.apply(
                    lambda row: mlflow.log_metric(
                        row["test_type"] + "_min_pass_rate",
                        float(row["minimum_pass_rate"].rstrip("%")) / 100,
                    ),
                    axis=1,
                )
                df_report.apply(
                    lambda row: mlflow.log_metric(
                        row["test_type"] + "_pass_status", 1 if row["pass"] else 0
                    ),
                    axis=1,
                )
                df_report.apply(
                    lambda row: mlflow.log_metric(
                        row["test_type"] + "_pass_count", row["pass_count"]
                    ),
                    axis=1,
                )
                df_report.apply(
                    lambda row: mlflow.log_metric(
                        row["test_type"] + "_fail_count", row["fail_count"]
                    ),
                    axis=1,
                )
                mlflow.end_run()

            if format == "dataframe":
                return self.df_report
            elif format == "dict":
                if save_dir is None:
                    raise ValueError(
                        'You need to set "save_dir" parameter for this format.'
                    )
                self.df_report.to_json(save_dir)
            elif format == "excel":
                if save_dir is None:
                    raise ValueError(
                        'You need to set "save_dir" parameter for this format.'
                    )
                self.df_report.to_excel(save_dir)
            elif format == "html":
                if save_dir is None:
                    raise ValueError(
                        'You need to set "save_dir" parameter for this format.'
                    )
                self.df_report.to_html(save_dir)
            elif format == "markdown":
                if save_dir is None:
                    raise ValueError(
                        'You need to set "save_dir" parameter for this format.'
                    )
                self.df_report.to_markdown(save_dir)
            elif format == "text" or format == "csv":
                if save_dir is None:
                    raise ValueError(
                        'You need to set "save_dir" parameter for this format.'
                    )
                self.df_report.to_csv(save_dir)
            else:
                raise ValueError(
                    f'Report in format "{format}" is not supported. Please use "dataframe", "excel", "html", "markdown", "text", "dict".'
                )

        else:
            df_final_report = pd.DataFrame()
            for k, v in self.model.items():
                for sample in self._generated_results[k]:
                    summary[sample.test_type]["category"] = sample.category
                    summary[sample.test_type][str(sample.is_pass()).lower()] += 1
                report = {}
                for test_type, value in summary.items():
                    pass_rate = summary[test_type]["true"] / (
                        summary[test_type]["true"] + summary[test_type]["false"]
                    )
                    min_pass_rate = self.min_pass_dict.get(
                        test_type, self.default_min_pass_dict
                    )

                    if summary[test_type]["category"] in ["Accuracy", "performance"]:
                        min_pass_rate = 1

                    report[test_type] = {
                        "model_name": k,
                        "pass_rate": pass_rate,
                        "minimum_pass_rate": min_pass_rate,
                        "pass": pass_rate >= min_pass_rate,
                    }

                df_report = pd.DataFrame.from_dict(report, orient="index")
                df_report = df_report.reset_index().rename(columns={"index": "test_type"})

                df_report["pass_rate"] = df_report["pass_rate"].apply(
                    lambda x: "{:.0f}%".format(x * 100)
                )
                df_report["minimum_pass_rate"] = df_report["minimum_pass_rate"].apply(
                    lambda x: "{:.0f}%".format(x * 100)
                )

                df_report = df_report.reset_index(drop=True)
                df_report = df_report.fillna("-")
                if mlflow_tracking:
                    try:
                        import mlflow
                    except ModuleNotFoundError:
                        print("mlflow package not found. Install mlflow first")

                    import datetime

                    experiment_name = k

                    # Get the experiment
                    experiment = mlflow.get_experiment_by_name(experiment_name)

                    if experiment is None:
                        # The experiment does not exist, create it
                        experiment_id = mlflow.create_experiment(experiment_name)
                    else:
                        # The experiment exists, get its ID
                        experiment_id = experiment.experiment_id

                    current_datetime = datetime.datetime.now().strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    )
                    mlflow.start_run(
                        run_name=self.task + "_testing_" + current_datetime,
                        experiment_id=experiment_id,
                    )

                    df_report.apply(
                        lambda row: mlflow.log_metric(
                            row["test_type"] + "_pass_rate",
                            float(row["pass_rate"].rstrip("%")) / 100,
                        ),
                        axis=1,
                    )
                    df_report.apply(
                        lambda row: mlflow.log_metric(
                            row["test_type"] + "_min_pass_rate",
                            float(row["minimum_pass_rate"].rstrip("%")) / 100,
                        ),
                        axis=1,
                    )
                    df_report.apply(
                        lambda row: mlflow.log_metric(
                            row["test_type"] + "_pass_status", 1 if row["pass"] else 0
                        ),
                        axis=1,
                    )
                    mlflow.end_run()

                df_final_report = pd.concat([df_final_report, df_report])

            df_final_report["minimum_pass_rate"] = (
                df_final_report["minimum_pass_rate"].str.rstrip("%").astype("float")
                / 100.0
            )

            df_final_report["pass_rate"] = (
                df_final_report["pass_rate"].str.rstrip("%").astype("float") / 100.0
            )

            pivot_df = df_final_report.pivot_table(
                index="model_name",
                columns="test_type",
                values="pass_rate",
                aggfunc="mean",
            )

            def color_cells(series):
                res = []
                for x in series.index:
                    res.append(
                        df_final_report[
                            (df_final_report["test_type"] == series.name)
                            & (df_final_report["model_name"] == x)
                        ]["pass"].all()
                    )
                return [
                    "background-color: green" if x else "background-color: red"
                    for x in res
                ]

            styled_df = pivot_df.style.apply(color_cells)

            if format == "dataframe":
                return styled_df
            elif format == "dict":
                return styled_df.to_dict("records")
            elif format == "excel":
                if save_dir is None:
                    raise ValueError(
                        'You need to set "save_dir" parameter for this format.'
                    )
                styled_df.to_excel(save_dir)
            elif format == "html":
                if save_dir is None:
                    raise ValueError(
                        'You need to set "save_dir" parameter for this format.'
                    )
                styled_df.to_html(save_dir)
            elif format == "markdown":
                if save_dir is None:
                    raise ValueError(
                        'You need to set "save_dir" parameter for this format.'
                    )
                styled_df.to_markdown(save_dir)
            elif format == "text" or format == "csv":
                if save_dir is None:
                    raise ValueError(
                        'You need to set "save_dir" parameter for this format.'
                    )
                styled_df.to_csv(save_dir)
            else:
                raise ValueError(
                    f'Report in format "{format}" is not supported. Please use "dataframe", "excel", "html", "markdown", "text", "dict".'
                )

    def generated_results(self) -> Optional[pd.DataFrame]:
        """Generates an overall report with every textcase and labelwise metrics.

        Returns:
            pd.DataFrame: Generated dataframe.
        """
        if self._generated_results is None:
            logging.warning(
                "Please run `Harness.run()` before calling `.generated_results()`."
            )
            return

        if isinstance(self._generated_results, dict):
            generated_results_df = []
            for k, v in self._generated_results.items():
                model_generated_results_df = pd.DataFrame.from_dict(
                    [x.to_dict() for x in v]
                )
                if (
                    "test_case" in model_generated_results_df.columns
                    and "original_question" in model_generated_results_df.columns
                ):
                    model_generated_results_df["original_question"].update(
                        model_generated_results_df.pop("test_case")
                    )
                model_generated_results_df["model_name"] = k
                generated_results_df.append(model_generated_results_df)
            generated_results_df = pd.concat(generated_results_df).reset_index(drop=True)

        else:
            generated_results_df = pd.DataFrame.from_dict(
                [x.to_dict() for x in self._generated_results]
            )

        column_order = [
            "model_name",
            "category",
            "test_type",
            "original",
            "prompt",
            "original_context",
            "original_question",
            "completion",
            "test_case",
            "perturbed_context",
            "perturbed_question",
            "patient_info_A",
            "patient_info_B",
            "diagnosis",
            "treatment_plan_A",
            "treatment_plan_B",
            "expected_result",
            "prompt_toxicity",
            "actual_result",
            "completion_toxicity",
            "hypothesis",
            "statements",
            "model_response",
            "eval_score",
            "similarity_score",
            "pass",
        ]
        columns = [c for c in column_order if c in generated_results_df.columns]
        generated_results_df = generated_results_df[columns]

        return generated_results_df.fillna("-")

    def augment(
        self,
        training_data: dict,
        save_data_path: str,
        custom_proportions: Union[Dict, List] = None,
        export_mode: str = "add",
        templates: Optional[Union[str, List[str]]] = None,
    ) -> "Harness":
        """Augments the data in the input file located at `input_path` and saves the result to `output_path`.

        Args:
            training_data (dict): A dictionary containing the input data for augmentation.
            save_data_path (str): Path to save the augmented data.
            custom_proportions (Union[Dict, List]):
            export_mode (str, optional): Determines how the samples are modified or exported.
                                    - 'inplace': Modifies the list of samples in place.
                                    - 'add': Adds new samples to the input data.
                                    - 'transformed': Exports only the transformed data, excluding untransformed samples.
                                    Defaults to 'add'.
            templates (Optional[Union[str, List[str]]]):

        Returns:
            Harness: The instance of the class calling this method.

        Raises:
            ValueError: If the `pass_rate` or `minimum_pass_rate` columns have an unexpected data type.

        Note:
            This method uses an instance of `AugmentRobustness` to perform the augmentation.

        """
        dtypes = list(
            map(
                str,
                self.df_report[["pass_rate", "minimum_pass_rate"]].dtypes.values.tolist(),
            )
        )
        if dtypes not in [["int64"] * 2, ["int32"] * 2]:
            self.df_report["pass_rate"] = (
                self.df_report["pass_rate"].str.replace("%", "").astype(int)
            )
            self.df_report["minimum_pass_rate"] = (
                self.df_report["minimum_pass_rate"].str.replace("%", "").astype(int)
            )

        # checking if the custom_proportions are valid
        if custom_proportions:
            vaild_test_types = set(
                custom_proportions.keys()
                if isinstance(custom_proportions, dict)
                else custom_proportions
            )
            report_test_types = set(self.df_report["test_type"].unique())
            vaild_test_types = set(
                custom_proportions.keys()
                if isinstance(custom_proportions, dict)
                else custom_proportions
            )
            report_test_types = set(self.df_report["test_type"].unique())

            if not (vaild_test_types.issubset(report_test_types)):
                raise ValueError(
                    f"Custom proportions for {vaild_test_types - report_test_types} not found in the test types."
                )

        if templates:
            _ = TemplaticAugment(
                templates=templates,
                task=self.task,
            ).fix(training_data=training_data, output_path=save_data_path)

        else:
            _ = AugmentRobustness(
                task=self.task,
                config=self._config,
                h_report=self.df_report,
                custom_proportions=custom_proportions,
            ).fix(
                training_data=training_data,
                output_path=save_data_path,
                export_mode=export_mode,
            )

        return self

    def testcases(self) -> pd.DataFrame:
        """Testcases after .generate() is called

        Returns:
            pd.DataFrame:
                testcases formatted into a pd.DataFrame
        """
        if isinstance(self._testcases, dict):
            testcases_df = []
            for k, v in self._testcases.items():
                model_testcases_df = pd.DataFrame([x.to_dict() for x in v])
                if "prompt" in model_testcases_df.columns:
                    return model_testcases_df.fillna("-")

                elif (
                    "test_case" in model_testcases_df.columns
                    and "original_question" in model_testcases_df.columns
                ):
                    model_testcases_df["original_question"].update(
                        model_testcases_df.pop("test_case")
                    )

                model_testcases_df["model_name"] = k
                testcases_df.append(model_testcases_df)

            testcases_df = pd.concat(testcases_df).reset_index(drop=True)

        else:
            testcases_df = pd.DataFrame([x.to_dict() for x in self._testcases])
            testcases_df = testcases_df.reset_index(drop=True)
            if "prompt" in testcases_df.columns:
                return testcases_df.fillna("-")

            elif (
                "test_case" in testcases_df.columns
                and "original_question" in testcases_df.columns
            ) and self.task != "political":
                testcases_df["original_question"].update(testcases_df.pop("test_case"))

        column_order = [
            "model_name",
            "category",
            "test_type",
            "original",
            "original_context",
            "original_question",
            "test_case",
            "patient_info_A",
            "patient_info_B",
            "diagnosis",
            "hypothesis",
            "statements",
            "perturbed_context",
            "perturbed_question",
            "expected_result",
        ]
        columns = [c for c in column_order if c in testcases_df.columns]
        testcases_df = testcases_df[columns]

        return testcases_df.fillna("-")

    def save(self, save_dir: str) -> None:
        """Save the configuration, generated testcases and the `DataFactory` to be reused later.

        Args:
            save_dir (str): path to folder to save the different files
        Returns:

        """
        if self._config is None:
            raise RuntimeError(
                "The current Harness has not been configured yet. Please use the `.configure` method "
                "before calling the `.save` method."
            )

        if self._testcases is None:
            raise RuntimeError(
                "The test cases have not been generated yet. Please use the `.generate` method before"
                "calling the `.save` method."
            )

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        with open(os.path.join(save_dir, "config.yaml"), "w", encoding="utf-8") as yml:
            yml.write(yaml.safe_dump(self._config_copy))

        with open(os.path.join(save_dir, "test_cases.pkl"), "wb") as writer:
            pickle.dump(self._testcases, writer)

        with open(os.path.join(save_dir, "data.pkl"), "wb") as writer:
            pickle.dump(self.data, writer)

    @classmethod
    def load(
        cls,
        save_dir: str,
        model: Union[str, "ModelFactory"],
        task: str,
        hub: Optional[str] = None,
    ) -> "Harness":
        """Loads a previously saved `Harness` from a given configuration and dataset

        Args:
            save_dir (str):
                path to folder containing all the needed files to load an saved `Harness`
            task (str):
                task for which the model is to be evaluated.
            model (str | ModelFactory):
                ModelFactory object or path to the model to be evaluated.
            hub (str, optional):
                model hub to load from the path. Required if path is passed as 'model'.

        Returns:
            Harness:
                `Harness` loaded from from a previous configuration along with the new model to evaluate
        """
        for filename in ["config.yaml", "test_cases.pkl", "data.pkl"]:
            if not os.path.exists(os.path.join(save_dir, filename)):
                raise OSError(
                    f"File '{filename}' is missing to load a previously saved `Harness`."
                )

        with open(os.path.join(save_dir, "data.pkl"), "rb") as reader:
            data = pickle.load(reader)

        harness = Harness(
            task=task,
            model={"model": model, "hub": hub},
            data={"data_source": data},
            config=os.path.join(save_dir, "config.yaml"),
        )
        harness.generate()

        return harness

    def edit_testcases(self, output_path: str, **kwargs):
        """Testcases are exported to a csv file to be edited.

        The edited file can be imported back to the harness

        Args:
            output_path (str): path to save the testcases to
        """
        temp_df = self.testcases()
        temp_df = temp_df[temp_df["category"].isin(["robustness", "bias"])]
        temp_df.to_csv(output_path, index=False)

    def import_edited_testcases(self, input_path: str, **kwargs):
        """Testcases are imported from a csv file

        Args:
            input_path (str): location of the file to load
        """
        temp_testcases = [
            sample
            for sample in self._testcases
            if sample.category not in ["robustness", "bias"]
        ]

        self._testcases = DataFactory(
            {"data_source": input_path}, task=self.task, is_import=True
        ).load()
        self._testcases.extend(temp_testcases)

        return self

    @staticmethod
    def available_tests(test_type: str = None) -> Dict[str, List[str]]:
        """Returns a dictionary of available tests categorized by test type.

        Args:
            test_type (str, optional): The specific test type to retrieve. Defaults to None.

        Returns:
            dict: Returns a dictionary containing available tests for the specified test type and defaults to all available tests.

        Raises:
            ValueError: If an invalid test type is provided.
        """
        test_scenarios = TestFactory.test_scenarios()
        available_tests = {
            test: list(scenarios.keys()) for test, scenarios in test_scenarios.items()
        }

        if test_type:
            if test_type not in available_tests.keys():
                raise ValueError(
                    f"Unsupported test type '{test_type}'. The available test types are: {available_tests.keys()}"
                )
            return {test_type: available_tests[test_type]}

        return available_tests

    def pass_custom_data(
        self,
        file_path: str,
        test_name: str = None,
        task: str = None,
        append: bool = False,
    ) -> None:
        """Load custom data from a JSON file and store it in a class variable.

        Args:
            file_path (str): Path to the JSON file.
            test_name (str, optional): Name parameter. Defaults to None.
            task (str, optional): Task type. Either "bias" or "representation". Defaults to None.
            append (bool, optional): Whether to append the data or overwrite it. Defaults to False.
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        if task == "bias":
            if test_name not in (
                "Country-Economic-Bias",
                "Religion-Bias",
                "Ethnicity-Name-Bias",
                "Gender-Pronoun-Bias",
            ):
                raise ValueError(
                    f"Invalid 'test_name' value '{test_name}'. It should be one of: Country-Economic-Bias, Religion-Bias, Ethnicity-Name-Bias, Gender-Pronoun-Bias."
                )

            TestFactory.call_add_custom_bias(data, test_name, append)
        elif task == "representation":
            if test_name not in (
                "Country-Economic-Representation",
                "Religion-Representation",
                "Ethnicity-Representation",
                "Label-Representation",
            ):
                raise ValueError(
                    f"Invalid 'test_name' value '{test_name}'. It should be one of: Country-Economic-Representation, Religion-Representation, Ethnicity-Representation, Label-Representation."
                )

            RepresentationOperation.add_custom_representation(
                data, test_name, append, check=self.task
            )

        else:
            raise ValueError(
                f"Invalid task type: {task}. Expected 'bias' or 'representation'."
            )

    def upload_folder_to_hub(
        repo_name: str,
        repo_type: str,
        folder_path: str,
        token: str,
        model_type: str = "huggingface",
        exist_ok: bool = False,
    ):
        """
        Uploads a folder containing a model or dataset to the Hugging Face Model Hub or Dataset Hub.

        This function facilitates the process of uploading a local folder containing a model or dataset to the Hugging Face
        Model Hub or Dataset Hub. It requires proper authentication through a valid token.

        Args:
            repo_name (str): The name of the repository on the Hub.
            repo_type (str): The type of the repository, either "model" or "dataset".
            folder_path (str): The local path to the folder containing the model or dataset files to be uploaded.
            token (str): The authentication token for accessing the Hugging Face Hub services.
            model_type (str, optional): The type of the model, currently supports "huggingface" and "spacy".
                                    Defaults to "huggingface".
            exist_ok (bool, optional): If True, do not raise an error if repo already exists.

        Raises:
            ValueError: If a valid token is not provided for Hugging Face Hub authentication.
            ModuleNotFoundError: If required package is not installed. This package needs to be installed based on
                                model_type ("huggingface" or "spacy").
        """
        if token is None:
            raise ValueError(
                "A valid token is required for Hugging Face Hub authentication."
            )
        subprocess.run(["huggingface-cli", "login", "--token", token], check=True)

        if (
            model_type == "huggingface" and repo_type == "model"
        ) or repo_type == "dataset":
            LIB_NAME = "huggingface_hub"

            if try_import_lib(LIB_NAME):
                huggingface_hub = importlib.import_module(LIB_NAME)
                HfApi = getattr(huggingface_hub, "HfApi")
            else:
                raise ModuleNotFoundError(
                    f"The '{LIB_NAME}' package is not installed. Please install it using 'pip install {LIB_NAME}'."
                )

            api = HfApi()

            repo_id = repo_name.split("/")[1]
            api.create_repo(repo_id, repo_type=repo_type, exist_ok=exist_ok)

            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_name,
                repo_type=repo_type,
            )
        elif model_type == "spacy" and repo_type == "model":
            LIB_NAME = "spacy_huggingface_hub"

            if try_import_lib(LIB_NAME):
                dataset_module = importlib.import_module(LIB_NAME)
                push = getattr(dataset_module, "push")
            else:
                raise ModuleNotFoundError(
                    f"The '{LIB_NAME}' package is not installed. Please install it using 'pip install {LIB_NAME}'."
                )
            meta_path = os.path.join(folder_path, "meta.json")
            with open(meta_path, "r") as meta_file:
                meta_data = json.load(meta_file)

            lang = meta_data["lang"]
            version = meta_data["version"]

            v = f"{lang}_pipeline-{version}"
            wheel_filename = f"{v}-py3-none-any.whl"

            output_dir_base = "output"
            output_dir = output_dir_base
            index = 1
            while os.path.exists(output_dir):
                output_dir = f"{output_dir_base}{index}"
                index += 1

            os.makedirs(output_dir, exist_ok=True)
            wheel_path = os.path.join(output_dir, v, "dist", wheel_filename)

            os.system(f"python -m spacy package {folder_path} {output_dir} --build wheel")

            push(wheel_path)

    def upload_file_to_hub(
        repo_name: str,
        repo_type: str,
        file_path: str,
        token: str,
        exist_ok: bool = False,
        split: str = "train",
    ):
        """Uploads a file or a Dataset to the Hugging Face Model Hub.

        Args:
            repo_name (str): The name of the repository in the format 'username/repository'.
            repo_type (str): The type of the repository, e.g: 'dataset' or 'model'.
            file_path (str): Path to the file to be uploaded.
            token (str): Hugging Face Hub authentication token.
            exist_ok (bool, optional): If True, do not raise an error if repo already exists.
            split (str, optional): The split of the dataset. Defaults to 'train'.

        Raises:
            ValueError: Raised if a valid token is not provided.
            ModuleNotFoundError: Raised if required packages are not installed.

        Returns:
            None
        """
        if token is None:
            raise ValueError(
                "A valid token is required for Hugging Face Hub authentication."
            )
        subprocess.run(["huggingface-cli", "login", "--token", token], check=True)

        file_extension = file_path.split(".")[-1]
        path_in_repo = os.path.basename(file_path)
        if file_extension != "conll":
            LIB_NAME = "huggingface_hub"
            if try_import_lib(LIB_NAME):
                huggingface_hub = importlib.import_module(LIB_NAME)
                HfApi = getattr(huggingface_hub, "HfApi")
            else:
                raise ModuleNotFoundError(
                    f"The '{LIB_NAME}' package is not installed. Please install it using 'pip install {LIB_NAME}'."
                )

            api = HfApi()
            repo_id = repo_name.split("/")[1]
            api.create_repo(repo_id, repo_type=repo_type, exist_ok=exist_ok)

            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_name,
                repo_type=repo_type,
                token=token,
            )
        else:
            LIB_NAME = "datasets"
            if try_import_lib(LIB_NAME):
                dataset_module = importlib.import_module(LIB_NAME)
                DatasetDict = getattr(dataset_module, "DatasetDict")
                Dataset = getattr(dataset_module, "Dataset")

            else:
                raise ModuleNotFoundError(
                    f"The '{LIB_NAME}' package is not installed. Please install it using 'pip install {LIB_NAME}'."
                )

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            data = []
            tokens = []
            ner_tags = []

            for line in lines:
                line = line.strip()
                if line:
                    if not line.startswith("-DOCSTART-"):
                        parts = line.split()
                        tokens.append(parts[0])
                        ner_tags.append(parts[-1])
                elif tokens:
                    data.append({"tokens": tokens, "ner_tags": ner_tags})
                    tokens = []
                    ner_tags = []

            df = pd.DataFrame(data)
            dataset = Dataset.from_pandas(df)
            ds = DatasetDict({split: dataset})

            ds.push_to_hub(
                repo_id=repo_name,
                token=token,
            )
