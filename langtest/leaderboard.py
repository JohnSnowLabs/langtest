import os
import click
import yaml
import json
import logging
from langtest.config import cli
from langtest import Harness
from langtest.utils.custom_types.helpers import create_dirs, create_folder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@cli.command("eval")
@click.option("--harness-config", "-c", type=str, required=True)
@click.option(
    "--output-dir",
    "-o",
    type=str,
    required=False,
    default=os.path.expanduser("~/.langtest/"),
)
def init_leaderboard(harness_config, output_dir):
    """Initialize a new langtest leaderboard."""
    logger.info("Initializing new langtest leaderboard...")

    store_dir = create_dirs(get_store_path(output_dir))

    model, task, config, data = get_parameters(harness_config)

    # prepare the harness parameters
    if isinstance(model, list):
        logger.warning("Handling multiple models is currently not supported.")
    if isinstance(data, list):
        logger.warning("Handling multiple datasets is currently not supported.")
    else:
        testcases_folder_key, report_folder_key = generate_folder_key(model, task, data)
        testcases_folder_path, is_exists_testcases = create_folder(
            store_dir["testcases"], testcases_folder_key
        )
        report_folder_path, is_exists_report = create_folder(
            store_dir["reports"], report_folder_key
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
        report = harness.report(
            format="csv", save_dir=os.path.join(report_folder_path, "report.csv")
        )
        logger.info("Generated report:")
        print(report.to_markdown(index=False))


def get_parameters(params_file: str):
    """Get the parameters from the configuration file."""
    # Check file extension
    if params_file.endswith(".yml"):
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
    model = params.get("model")
    task = params.get("task")
    config = params.get("config")
    data = params.get("data")

    return model, task, config, data


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
            f"Generating and storing new testcases because the old config present in dir: {testcases_folder_path} differs from the existing one."
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
    harness.data = harness.data[:10]
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


def generate_folder_key(model, task, data):
    """Generate report folder key."""

    model_str = "-".join(model.values())
    data_str = "-".join(
        [data[key] for key in ["data_source", "subset", "split"] if key in data]
    )
    task_str = "-".join(task.values()) if isinstance(task, dict) else task

    return f"{task_str}-{data_str}", f"{model_str}-{task_str}-{data_str}"


def get_store_path(output_dir):
    return os.path.expanduser(output_dir)
