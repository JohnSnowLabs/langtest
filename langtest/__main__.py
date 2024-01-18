import click
from langtest import Harness
from langtest.config import cli
from langtest.pipelines.embedding import benchmark

click.CommandCollection(sources=[cli, benchmark], help="LangTest CLI")
harness = None


@cli.command("init")
@click.option("--task", "-t", type=str, required=True)
@click.option("--model", "-m", type=str, required=True)
@click.option("--hub", "-h", type=str, required=True)
@click.option("--dataset", "-d", type=str, required=True)
@click.option("--config", "-c", type=str, required=True)
def init(task, model, hub, dataset, config):
    """Initialize a new langtest project."""
    print(f"Initializing new langtest project with {task}, {model}, {dataset}, {config}.")
    global harness
    harness = Harness(
        task=task,
        model={"model": model, "hub": hub},
        dataset=dataset,
        config=config,
    )


@cli.command("generate")
def generate():
    """Generate a new langtest project."""
    print("Generating new langtest project.")
    global harness
    # harness.generate()


if __name__ == "__main__":
    cli()
