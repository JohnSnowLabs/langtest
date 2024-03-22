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
    harness = Harness(
        task="question-answering",
        model={
            "model": model,
            "hub": hub,
        },
        data={
            "data_source": datasets,
            "split": "test",
        },
    )
    # harness.generate(seed=42)
    generate_store_testcases(harness, {"data_source": datasets, "split": "test"})


def generate_store_testcases(harness: Harness, data_dict: dict, *args, **kwargs) -> str:
    """Generate the testcases."""

    # Create the folder
    global store_dir
    if "testcases" not in store_dir:
        global STORE_PATH
        store_dir = create_dirs(STORE_PATH)

    # create tbe folder for storing the testcases
    testcases_dir = store_dir["testcases"]
    folder_path, is_exists = create_folder(testcases_dir, data_dict)

    if is_exists:
        # Load the testcases
        print(is_exists, folder_path)
        harness = Harness.load(
            save_dir=folder_path,
            task=harness.task,
            model=harness.model,
        )
        print(f"Loading testcases from {folder_path}.")
        return harness
    else:
        # Generate the testcases
        harness.generate(seed=42)

        # Save the testcases
        harness.save(folder_path, *args, **kwargs)
        print(f"Testcases saved to {folder_path}.")

    return harness
