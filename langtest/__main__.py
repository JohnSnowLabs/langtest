import click
import os
import json
import sys
from langtest import Harness
from langtest.config import cli
from langtest.pipelines.embedding import benchmark
from langtest.leaderboard import *  # noqa

click.CommandCollection(sources=[cli, benchmark], help="LangTest CLI")


@cli.command("init")
@click.option("--task", "-t", type=str, required=True)
@click.option("--model", "-m", type=str, required=True)
@click.option("--hub", "-h", type=str, required=True)
@click.option("--dataset", "-d", type=str, required=False)
@click.option("--config", "-c", type=str, required=False)
@click.option("--output", "-o", type=str, required=False, default="./langtest")
def init(task, model, hub, dataset, config, output):
    """Initialize a new langtest project."""
    print("Initializing new langtest project...")

    # get abspath for output
    output = os.path.abspath(output)
    # do required directories in cwd for project
    os.makedirs(output, exist_ok=True)
    _ = Harness(
        task=task,
        model={"model": model, "hub": hub},
        data={"data_source": dataset},
        config=config,
    )
    with open("./harness.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "task": task,
                    "model": {"model": model, "hub": hub},
                    "data": {"data_source": dataset},
                    "config": config,
                    "save_dir": os.path.abspath(output),
                }
            )
        )

    # harness.save(output)


@cli.command("generate")
def generate():
    """Generate the testcases."""
    print("Generating the testcases...")
    if os.path.exists("./harness.json"):
        params = json.load(open("./harness.json", "r"))
        harness = Harness(
            task=params["task"],
            model=params["model"],
            data=params["data"],
            config=params["config"],
        )
        # generate testcases
        harness.generate()

        # save harness
        harness.save(params["save_dir"])
    else:
        sys.exit(
            "No harness.json found in current directory. Please run `langtest init` first."
        )


@cli.command("run")
@click.option("--task", "-t", type=str, required=False)
@click.option("--model", "-m", type=str, required=False)
@click.option("--hub", "-h", type=str, required=False)
@click.option("--batch_size", "-b", type=int, required=False, default=500)
@click.option("--checkpoint", "-c", type=bool, required=False, default=True)
def run(task, model, hub, batch_size, checkpoint):
    """Run the testcases."""
    harness = None
    params = None
    if os.path.exists("./harness.json"):
        params = json.load(open("./harness.json", "r"))
        checkpoints_dir = params["save_dir"] + "/checkpoints"

    else:
        sys.exit(
            "No harness.json found in current directory. Please run `langtest init` first."
        )

    # check if checkpoints exist and load them
    if os.path.exists(checkpoints_dir):
        harness = Harness.load_checkpoints(
            task,
            model={"model": model, "hub": hub},
            save_checkpoints_dir=checkpoints_dir,
        )
    # check if harness.json exists and load it
    elif model and hub:
        # different model, hub, or task
        harness = Harness.load(
            save_dir=params["save_dir"],
            task=task,
            model={"model": model, "hub": hub},
        )
    else:
        harness = Harness.load(
            save_dir=params["save_dir"],
            task=params["task"],
            model=params["model"],
            load_testcases=True,
        )

    # check if harness is not None then run it
    if harness is not None:
        harness.run(
            batch_size=batch_size,
            checkpoint=checkpoint,
            save_checkpoints_dir=checkpoints_dir,
        )

        generated_results = harness.generated_results()
        generated_results.to_csv(
            f"{params['save_dir']}/generated_results.csv", index=False
        )
        # save harness
        harness.save(params["save_dir"], include_generated_results=True)


@cli.command("report")
def report():
    """Generate the report."""
    if os.path.exists("./harness.json"):
        params = json.load(open("./harness.json", "r"))
    else:
        sys.exit(
            "No harness.json found in current directory. Please run `langtest init` first."
        )

    harness = Harness.load_checkpoints(
        task=params["task"],
        model=params["model"],
        save_checkpoints_dir=f"{params['save_dir']}/checkpoints",
    )
    harness.run()
    # generated results and report
    report = harness.report()

    # print and save
    print(report)
    report.to_csv(f"{params['save_dir']}/report.csv", index=False)


if __name__ == "__main__":
    cli()
