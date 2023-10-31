import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union
from ..errors import Errors
from langtest.tasks import TaskManager


def political_report(generated_results: List) -> pd.DataFrame:
    """
    Generate a political report based on the given results.

    This function calculates the political economic and social scores from the input results
    based on whether the test cases are 'right', 'left', 'auth', or 'lib'. The results are
    then presented in a DataFrame format and visualized using a scatter plot to show the
    political coordinates. The X-axis represents the economic score (right-positive, left-negative),
    and the Y-axis represents the social score (auth-positive, lib-negative).

    Parameters:
    - generated_results (List): A list of samples, where each sample has properties
                                'test_case' and 'is_pass' to indicate the type of test case
                                and whether the test passed, respectively.

    """
    econ_score = 0.0
    econ_count = 0.0
    social_score = 0.0
    social_count = 0.0
    for sample in generated_results:
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

    df_report = df_report.fillna("-")

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

    return df_report


def model_report(
    summary: Dict,
    min_pass_dict: Dict,
    default_min_pass_dict: float,
    generated_results: List,
) -> pd.DataFrame:
    """
    Generate a report summarizing the performance of a model based on provided results.

    This function computes the pass rate of each test type, compares it against a specified minimum pass rate,
    and creates a detailed report with pass counts, fail counts, pass rates, and whether the pass rate meets the minimum threshold.

    Parameters:
    - summary (Dict): A dictionary to store and accumulate results by test type.
    - min_pass_dict (Dict): A dictionary specifying the minimum pass rate for each test type.
    - default_min_pass_dict (float): Default minimum pass rate if not specified in `min_pass_dict`.
    - generated_results (List): A list of objects where each object should have a `test_type` attribute indicating the type,
                                a `category` attribute indicating the category (e.g., "robustness", "Accuracy"),
                                and an `is_pass` attribute indicating whether the test passed or not (1 for pass, 0 for fail).

    Returns:
    - pd.DataFrame: A DataFrame containing detailed reporting for each test type. The columns include "test_type",
                    "category", "fail_count", "pass_count", "pass_rate", "minimum_pass_rate", and "pass".

    """

    report = {}
    for sample in generated_results:
        summary[sample.test_type]["category"] = sample.category
        summary[sample.test_type][str(sample.is_pass()).lower()] += 1
        for test_type, value in summary.items():
            pass_rate = summary[test_type]["true"] / (
                summary[test_type]["true"] + summary[test_type]["false"]
            )
            min_pass_rate = min_pass_dict.get(test_type, default_min_pass_dict)

            if "-" in test_type and summary[test_type]["category"] == "robustness":
                multiple_perturbations_min_pass_rate = min_pass_dict.get(
                    "multiple_perturbations", default_min_pass_dict
                )
                min_pass_rate = min_pass_dict.get(
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

    df_report = df_report.fillna("-")

    return df_report


def multi_model_report(
    summary: Dict,
    min_pass_dict: Dict,
    default_min_pass_dict: float,
    generated_results: Dict,
    model_name: str,
) -> pd.DataFrame:
    """
    Generate a report summarizing the performance of a specific model from multiple models based on provided results.

    This function computes the pass rate of each test type for the specified model, compares it against a specified minimum
    pass rate, and creates a detailed report with pass rates and whether the pass rate meets the minimum threshold.

    Parameters:
    - summary (Dict): A dictionary to store and accumulate results by test type.
    - min_pass_dict (Dict): A dictionary specifying the minimum pass rate for each test type.
    - default_min_pass_dict (float): Default minimum pass rate if not specified in `min_pass_dict`.
    - generated_results (Dict): A dictionary with model names as keys and a list of test results as values.
                               Each test result should have a `test_type` attribute indicating the type,
                               a `category` attribute, and an `is_pass` attribute.
    - model_name (str): The name of the model for which the report should be generated.

    Returns:
    - pd.DataFrame: A DataFrame containing a detailed report for the specified model. The columns include "test_type",
                    "model_name", "pass_rate", "minimum_pass_rate", and "pass".
    """

    for sample in generated_results[model_name]:
        summary[sample.test_type]["category"] = sample.category
        summary[sample.test_type][str(sample.is_pass()).lower()] += 1
    report = {}
    for test_type, value in summary.items():
        pass_rate = summary[test_type]["true"] / (
            summary[test_type]["true"] + summary[test_type]["false"]
        )
        min_pass_rate = min_pass_dict.get(test_type, default_min_pass_dict)

        if summary[test_type]["category"] in ["Accuracy", "performance"]:
            min_pass_rate = 1

        report[test_type] = {
            "model_name": model_name,
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
    return df_report


def color_cells(series: pd.Series, df_final_report: pd.DataFrame):
    """
    Apply background coloring to cells based on the "pass" value for a given model and test type.

    This function determines if the test passed or failed for each model based on the `pass` column in the final report.
    Cells are colored green if the test passed and red if the test failed.

    Parameters:
    - series (pd.Series): A Series object containing model names as its index.
    - df_final_report (pd.DataFrame): The final report DataFrame with columns "test_type", "model_name", and "pass"
                                      where "pass" indicates if the test passed (True) or failed (False).

    """

    res = []
    for x in series.index:
        res.append(
            df_final_report[
                (df_final_report["test_type"] == series.name)
                & (df_final_report["model_name"] == x)
            ]["pass"].all()
        )
    return ["background-color: green" if x else "background-color: red" for x in res]


def mlflow_report(
    experiment_name: str,
    task: Union[str, TaskManager],
    df_report: pd.DataFrame,
    multi_model_comparison: bool = False,
):
    """
    Logs metrics and details from a given report to an MLflow experiment.

    This function uses MLflow to record various metrics (e.g., pass rate, pass status) from a given report into a specified experiment.
    If the experiment does not already exist, it's created. If it does exist, metrics are logged under a new run.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment where metrics will be logged.
    - task (str): A descriptor or identifier for the current testing task, used to name the run.
    - df_report (pd.DataFrame): DataFrame containing the report details. It should have columns like "pass_rate", "minimum_pass_rate", "pass",
                                and optionally "pass_count" and "fail_count".
    - multi_model_comparison (bool, optional): Indicates whether the report pertains to a comparison between multiple models.
                                               If `True`, certain metrics like "pass_count" and "fail_count" are not logged.
                                               Default is `False`.
    """

    try:
        import mlflow
    except ModuleNotFoundError:
        print("mlflow package not found. Install mlflow first")

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
        run_name=str(task) + "_testing_" + current_datetime,
        experiment_id=experiment_id,
    )

    metrics_to_log = {
        "_pass_rate": lambda row: float(row["pass_rate"].rstrip("%")) / 100,
        "_min_pass_rate": lambda row: float(row["minimum_pass_rate"].rstrip("%")) / 100,
        "_pass_status": lambda row: 1 if row["pass"] else 0,
    }

    if not multi_model_comparison:
        metrics_to_log["_pass_count"] = lambda row: row["pass_count"]
        metrics_to_log["_fail_count"] = lambda row: row["fail_count"]

    for suffix, func in metrics_to_log.items():
        df_report.apply(
            lambda row: mlflow.log_metric(row["test_type"] + suffix, func(row)), axis=1
        )

    mlflow.end_run()


def save_format(format: str, save_dir: str, df_report: pd.DataFrame):
    """
    Save the provided report DataFrame into a specified format at a given directory.

    This function supports saving the report in multiple formats such as JSON (dict), Excel, HTML, Markdown, Text, and CSV. The user
    needs to specify the desired format and the directory to which the report should be saved. If a format that isn't supported is
    provided, an error will be raised.

    Parameters:
    - format (str): The desired format to save the report in. Supported values are "dict", "excel", "html", "markdown", "text", and "csv".
    - save_dir (str): The directory path where the report should be saved. This must be provided for all formats.
    - df_report (pd.DataFrame): The report DataFrame containing the data to be saved.
    """

    if format == "dataframe":
        return

    elif format == "dict":
        if save_dir is None:
            raise ValueError(Errors.E012)

        df_report.to_json(save_dir)

    elif format == "excel":
        if save_dir is None:
            raise ValueError(Errors.E012)
        df_report.to_excel(save_dir)
    elif format == "html":
        if save_dir is None:
            raise ValueError(Errors.E012)
        df_report.to_html(save_dir)
    elif format == "markdown":
        if save_dir is None:
            raise ValueError(Errors.E012)
        df_report.to_markdown(save_dir)
    elif format == "text" or format == "csv":
        if save_dir is None:
            raise ValueError(Errors.E012)
        df_report.to_csv(save_dir)
    else:
        raise ValueError(Errors.E013)
