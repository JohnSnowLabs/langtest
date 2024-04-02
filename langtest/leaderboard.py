import os
import click
import yaml
import json
import logging
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from langtest.config import cli
from langtest import Harness
from langtest.utils.custom_types.helpers import create_dirs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

desired_order = [
    "timestamp",
    "parms_dir",
    "model",
    "hub",
    "data_source",
    "split",
    "subset",
    "task",
]


@cli.command("eval")
@click.option("--harness-config-path", "-c", type=str, required=True)
@click.option(
    "--output-dir",
    "-o",
    type=str,
    required=False,
    default=os.path.expanduser("~/.langtest/"),
)
@click.option("--model", "-m", type=str, required=False)
@click.option("--hub", "-h", type=str, required=False)
def init_leaderboard(harness_config_path, output_dir, model, hub):
    """Initialize a new langtest leaderboard."""
    logger.info("Initializing new langtest leaderboard...")

    print(output_dir)
    store_dir = create_dirs(get_store_path(output_dir))

    params, model, task, config, data = get_parameters(
        harness_config_path,
        update_model_dict={"model": model, "hub": hub} if model and hub else None,
    )

    testcases_folder_key, timestamp = generate_folder_key(model, task, data, config)
    testcases_folder_path, is_exists_testcases = create_folder(
        store_dir["testcases"], testcases_folder_key
    )
    report_folder_path, _ = create_folder(store_dir["reports"], timestamp)

    # Save the parameters file
    save_file(
        os.path.join(report_folder_path, os.path.basename(harness_config_path)), params
    )

    if is_exists_testcases:
        logger.info(f"Testcases already exist at: {testcases_folder_path}")
        harness = load_old_testcases(
            task=task,
            model=model,
            data=data,
            config=config,
            testcases_folder_path=testcases_folder_path,
        )
    else:
        harness = generate_store_testcases(
            task=task,
            model=model,
            data=data,
            config=config,
            testcases_folder_path=testcases_folder_path,
        )

    harness.run()
    generated_results = harness.generated_results()
    # save the generated results
    generated_results.to_csv(
        os.path.join(report_folder_path, "generated_results.csv"), index=False
    )
    if "accuracy" in generated_results["category"].unique().tolist():
        harness.model_response("accuracy").to_csv(
            os.path.join(report_folder_path, "accuracy_model_responses.csv"), index=False
        )
    report = harness.report(
        format="csv", save_dir=os.path.join(report_folder_path, "report.csv")
    )

    if isinstance(data, list):
        report.columns = [v for col, v in report.columns]
        report.reset_index(inplace=True)
    logger.info("Updating leaderboard...")

    if isinstance(data, list):
        report_dict = {name: group for name, group in report.groupby("dataset_name")}
        generated_results_dict = {
            name: group for name, group in generated_results.groupby("dataset_name")
        }
    else:
        report_dict = {
            "": report
        }  # If data is not a list, group everything under an empty string key
        generated_results_dict = {"": generated_results}

    for name in report_dict.keys():
        create_leaderboard(
            report=report_dict.get(name, report),
            generated_results=generated_results_dict.get(name, generated_results),
            model=(
                model
                if model["hub"] != "lm-studio"
                else {
                    "model": get_lm_studio_model_name(model["model"]),
                    "hub": "lm-studio",
                }
            ),
            task=task if isinstance(task, dict) else {"task": task},
            data={"data_source": name} if name else data,
            save_dir=store_dir["leaderboard"],
            parms_dir=os.path.join(
                report_folder_path, os.path.basename(harness_config_path)
            ),
            timestamp=timestamp,
        )
    # print "leaderboard"
    for key in report["category"].unique().tolist():
        # print horizontal line
        print(f"\n\n{'':=^80}\n{key:^80}\n{'':=^80}")
        logger.info(f"{key} Leaderboard")
        pivot_df = pd.read_csv(
            os.path.join(store_dir["leaderboard"], f"{key}_leaderboard.csv")
        )
        pivot_df.sort_values(by="avg", ascending=False, inplace=True)
        pivot_df.reset_index(drop=True, inplace=True)
        pivot_df.index += 1

        print(pivot_df.to_markdown())

        print(f"{'':-^80}\n")


@cli.command("show-leaderboard")
@click.option(
    "--output-dir",
    "-o",
    type=str,
    required=False,
    default=os.path.expanduser("."),
)
def show_leaderboard(output_dir):
    # check if the store_dir pickle exists
    print(os.path.expanduser(f"{output_dir}/.langtest"))
    if not os.path.exists(os.path.expanduser(f"{output_dir}/.langtest")):
        output_dir = os.path.expanduser("~/")

    import pickle

    if not os.path.exists(os.path.expanduser(f"{output_dir}/.langtest")):
        logger.info("Store directory not found. Please run 'init-leaderboard' first.")
        return

    with open(os.path.expanduser(f"{output_dir}/.langtest/store_dir.pkl"), "rb") as file:
        store_dir = pickle.load(file)

    leaderboard_files = [
        file
        for file in os.listdir(store_dir["leaderboard"])
        if file.endswith("_leaderboard.csv")
    ]
    for file in leaderboard_files:
        # print horizontal line
        key = file.split("_")[0]
        print(f"\n\n{'':=^80}\n{key:^80}\n{'':=^80}")
        logger.info(f"{key} Leaderboard")
        pivot_df = pd.read_csv(os.path.join(store_dir["leaderboard"], f"{file}"))
        pivot_df.sort_values(by="avg", ascending=False, inplace=True)
        pivot_df.reset_index(drop=True, inplace=True)
        pivot_df.index += 1

        print(pivot_df.to_markdown())

        print(f"{'':-^80}\n")


def get_parameters(
    params_file: str,
    update_model_dict: dict = None,
):
    """Get the parameters from the configuration file."""
    # Check file extension
    if params_file.endswith(".yml") or params_file.endswith(".yaml"):
        loader = yaml.safe_load
    elif params_file.endswith(".json"):
        loader = json.load
    else:
        raise ValueError(
            "Unsupported file format. Supported formats are YAML (.yml) and JSON (.json)."
        )

    with open(params_file, "r", encoding="utf-8") as file:
        params = loader(file)

    required_keys = ["model", "task", "data"]
    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        raise ValueError(
            f"Required key(s) {', '.join(missing_keys)} not found in the configuration file."
        )
    if update_model_dict:
        params["model"].update(update_model_dict)
    model = params.get("model")
    task = params.get("task")
    config = params.get("config")
    data = params.get("data")

    return params, model, task, config, data


def load_old_testcases(
    task, model, data: dict, testcases_folder_path: str, config=None, *args, **kwargs
) -> Harness:
    """Generate the testcases."""
    old_config_path = os.path.join(testcases_folder_path, "config.yaml")
    try:
        with open(old_config_path, "r", encoding="utf-8") as file:
            old_config = yaml.safe_load(file)
    except FileNotFoundError:
        # If the config file doesn't exist, generate and store new testcases
        logger.info(
            f"Generating and storing new testcases because the old config present in dir: {testcases_folder_path} is missing."
        )
        return generate_store_testcases(
            task=task,
            model=model,
            data=data,
            config=config,
            testcases_folder_path=testcases_folder_path,
        )

    # Check if the old config matches the provided config
    if old_config == config:
        # Load testcases if config matches
        harness = Harness.load(
            save_dir=testcases_folder_path,
            task=task,
            model=model,
        )
        logger.info(f"Loading testcases from {testcases_folder_path}.")
        return harness

    else:
        logger.info(
            f"Generating and storing new testcases because the old config present in dir: {testcases_folder_path} differs from the existing one."
        )
        return generate_store_testcases(
            task=task,
            model=model,
            data=data,
            config=config,
            testcases_folder_path=testcases_folder_path,
        )


def generate_store_testcases(
    task, model, data: dict, testcases_folder_path: str, config=None, *args, **kwargs
) -> Harness:
    harness = Harness(
        task=task,
        model=model,
        data=data,
        config=config,
    )
    # Generate the testcases
    harness.generate(seed=42)

    # Save the testcases

    harness.save(testcases_folder_path, *args, **kwargs)
    logger.info(f"Testcases saved to {testcases_folder_path}.")

    return harness


def run_store_checkpoints(
    harness: Harness, checkpoints_dir: str, model_dict, *args, **kwargs
):
    """Run the testcases on the checkpoints."""
    # Create the folder
    folder_path, is_exists = create_folder(checkpoints_dir, model_dict)

    if is_exists:
        # Load the testcases
        logger.info(f"Loading testcases from {folder_path}.")
        harness = Harness.load(
            save_dir=folder_path,
            task=harness.task,
            model=harness.model,
        )
    else:
        # Run the testcases
        harness.run(*args, **kwargs)

        # Save the testcases
        harness.save(folder_path, *args, **kwargs)
        logger.info(f"Testcases saved to {folder_path}.")

    return harness


def generate_folder_key(model, task, data, config):
    """Generate report folder key."""

    if isinstance(data, list):
        data_str = ",".join(
            "+".join(
                item.get(key, "")
                for key in ["data_source", "split", "subset"]
                if key in item
            )
            for item in data
        )

    else:
        data_str = "+".join(
            [data[key] for key in ["data_source", "subset", "split"] if key in data]
        )

    task_str = "+".join(task.values()) if isinstance(task, dict) else task

    test_categories = [category for category in config["tests"] if category != "defaults"]
    test_categories_str = "+".join(test_categories)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    data_unique_key = f"{task_str}&{data_str}&{test_categories_str}"

    return data_unique_key, timestamp


def get_store_path(output_dir):
    if output_dir == os.path.expanduser("~/.langtest/"):
        return output_dir
    return os.path.expanduser(f"{output_dir}/.langtest/")


def create_folder(default_location: str, folder_name: str) -> str:
    """Create the folder based on the data_dict."""

    folder_dir = os.path.join(default_location, folder_name)

    if os.path.exists(folder_dir):
        return folder_dir, True

    os.makedirs(folder_dir, exist_ok=True)
    return folder_dir, False


def get_lm_studio_model_name(endpoint: str):
    import requests

    modified_endpoint = endpoint.replace("chat/completions", "models")
    r = requests.get(modified_endpoint)
    data = r.json()["data"][0]
    return os.path.basename(data.get("id"))


def create_leaderboard(
    report: pd.DataFrame,
    generated_results: pd.DataFrame,
    model: dict,
    task: dict,
    data: dict,
    save_dir: str,
    **keywords,
):
    # Define a dictionary to map category to score key
    category_score_mapping = ["accuracy", "robustness"]

    test_categories = report["category"].unique().tolist()

    for category in test_categories:
        if category in category_score_mapping:
            if category == "accuracy":
                filtered_report = generated_results[
                    generated_results["category"] == category
                ]
            elif category == "robustness":
                filtered_report = report[report["category"] == category]
                filtered_report["pass_rate"] = (
                    filtered_report["pass_rate"].str.rstrip("%").astype(float)
                )

            summary_data = getattr(sys.modules[__name__], f"prepare_{category}_summary")(
                filtered_report, model, task, data, **keywords
            )

            summary_file_path = update_summary(summary_data, category, save_dir)
            update_leaderboard(summary_file_path, category)


def prepare_accuracy_summary(
    report: pd.DataFrame, model: dict, task: dict, data: dict, **keywords
):
    if "test_case" in report.columns:
        report["key"] = [
            f"{test_type}-{test_case}"
            for test_type, test_case in zip(report["test_type"], report["test_case"])
        ]
    else:
        report["key"] = report["test_type"].values
    overall_accuracy = report["actual_result"].mean()
    result_dict = report.set_index("key")["actual_result"].to_dict()
    result_dict.update(
        {**model, "task": task, **data, **keywords, "overall_accuracy": overall_accuracy}
    )
    return result_dict


def prepare_robustness_summary(
    report: pd.DataFrame, model: dict, task: dict, data: dict, **keywords
):
    overall_robustness = report["pass_rate"].mean()
    result_dict = report.set_index("test_type")["pass_rate"].to_dict()
    result_dict.update(
        {
            **model,
            "task": task,
            **data,
            **keywords,
            "overall_robustness": overall_robustness,
        }
    )
    return result_dict


def update_summary(summary_data: dict, category: str, save_dir: str) -> str:
    summary_file_path = os.path.join(save_dir, f"{category}_summary.csv")
    if not os.path.exists(summary_file_path):
        df = pd.DataFrame([summary_data])
        df = reorder_columns(df, desired_order)
        df.to_csv(summary_file_path, index=False)
    else:
        df = pd.read_csv(summary_file_path)
        for key in summary_data.keys():
            if key not in df.columns:
                df[key] = np.nan

        df = pd.concat([df, pd.DataFrame([summary_data])], ignore_index=True)
        df = reorder_columns(df, desired_order)
        df.to_csv(summary_file_path, index=False)

    return summary_file_path


def update_leaderboard(summary_file_path: str, category: str):
    metric = f"overall_{category}"
    df = pd.read_csv(summary_file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d-%H-%M-%S")
    df = df.sort_values(by="timestamp", ascending=False)
    unique_records = df.drop_duplicates(
        subset=["model", "hub", "data_source", "split", "subset", "task"]
    )
    unique_records.reset_index(drop=True, inplace=True)
    average = (
        unique_records.groupby(
            [
                "model",
                "hub",
                "data_source",
                "task",
            ]
        )[[metric, "timestamp", "split", "subset"]]
        .agg(
            {
                metric: "mean",
                "timestamp": list,
                "split": list,
                "subset": list,
            }
        )
        .reset_index()
    )
    pivot_df = average.pivot_table(
        index="model", columns="data_source", values=metric, aggfunc="first"
    )
    numeric_cols = pivot_df.select_dtypes(include=[float]).columns

    pivot_df["avg"] = pivot_df[numeric_cols].mean(axis=1)
    pivot_df.insert(0, "avg", pivot_df.pop("avg"))

    pivot_df["std"] = pivot_df[numeric_cols].std(axis=1)
    pivot_df.insert(1, "std", pivot_df.pop("std"))

    pivot_df.to_csv(
        os.path.join(os.path.dirname(summary_file_path), f"{category}_leaderboard.csv"),
    )


def reorder_columns(df: pd.DataFrame, desired_order: list) -> pd.DataFrame:
    """Reorders columns in the DataFrame according to the desired order."""
    return df.reindex(
        columns=desired_order + [col for col in df.columns if col not in desired_order]
    )


def save_file(file_path, data):
    """
    Save data to a file based on the file extension.

    Args:
        file_path (str): The path to the file to save.
        data (dict): The data to save.

    Raises:
        ValueError: If the file format is not supported.
    """
    if file_path.endswith(".yml"):
        dumper = yaml.safe_dump
    elif file_path.endswith(".json"):
        dumper = json.dump
    else:
        raise ValueError(
            "Unsupported file format. Supported formats are YAML (.yml) and JSON (.json)."
        )

    with open(file_path, "w", encoding="utf-8") as file:
        dumper(data, file)
