import os
import click
from langtest.config import cli
from langtest import Harness
from langtest.utils.custom_types.helpers import create_dirs, create_folder

STORE_PATH = os.path.expanduser("~/.langtest/")

store_dir = create_dirs(STORE_PATH)


@cli.command("eval")
@click.option("--model", "-m", type=str, required=True)
@click.option("--hub", "-h", type=str, required=True)
@click.option("--datasets", "-d", type=str, required=True)
@click.option("--output", "-o", type=str, required=False, default="./langtest")
def init_leaderboard(model, hub, datasets, output):
    """Initialize a new langtest leaderboard."""
    print("Initializing new langtest leaderboard...")

    # prepare the harness parameters
    task = "question-answering"
    model_dict = {
        "model": model,
        "hub": hub,
    }
    datasets = datasets.split(",")
    data_dict = []
    for dataset in datasets:
        data_dict.append({"data_source": dataset, "split": "test-tiny"})
    harness = generate_store_testcases(task, model_dict, data_dict)
    harness.run()
    report = harness.report()

    print(report.to_markdown(index=False))


def generate_store_testcases(
    task, model, data_dict: dict, config=None, *args, **kwargs
) -> Harness:
    """Generate the testcases."""

    # Create the folder
    global store_dir
    if "testcases" not in store_dir:
        global STORE_PATH
        store_dir = create_dirs(STORE_PATH)

    if isinstance(data_dict, list) and len(data_dict) <= 1:
        # Create a folder for each dataset
        data_dict = data_dict[0]

    # create tbe folder for storing the testcases
    testcases_dir = store_dir["testcases"]
    folder_path, is_exists = create_folder(testcases_dir, data_dict)

    if is_exists:
        # Load the testcases
        print(is_exists, folder_path)
        harness = Harness.load(
            save_dir=folder_path,
            task=task,
            model=model,
        )
        print(f"Loading testcases from {folder_path}.")
        return harness
    else:
        # Initialize the harness
        harness = Harness(
            task=task,
            model=model,
            data=data_dict,
            config=config,
        )
        # Generate the testcases
        harness.generate(seed=42)

        # Save the testcases
        harness.save(folder_path, *args, **kwargs)
        print(f"Testcases saved to {folder_path}.")

    return harness


def run_store_checkpoints(
    harness: Harness, checkpoints_dir: str, model_dict, *args, **kwargs
):
    """Run the testcases on the checkpoints."""
    # Create the folder
    folder_path, is_exists = create_folder(checkpoints_dir, model_dict)

    if is_exists:
        # Load the testcases
        print(is_exists, folder_path)
        harness = Harness.load(
            save_dir=folder_path,
            task=harness.task,
            model=harness.model,
        )
        print(f"Loading testcases from {folder_path}.")
    else:
        # Run the testcases
        harness.run(*args, **kwargs)

        # Save the testcases
        harness.save(folder_path, *args, **kwargs)
        print(f"Testcases saved to {folder_path}.")

    return harness
