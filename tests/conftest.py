import json
import os
from pathlib import Path

import pytest


def pytest_sessionstart():
    """Called after the Session object has been created"""
    homewd = str(Path.home())
    metaflow_config = os.path.join(homewd, ".metaflowconfig")
    os.makedirs(metaflow_config, exist_ok=True)

    with open(os.path.join(metaflow_config, "config_test.json"), "w") as writer:
        json.dump({"METAFLOW_DEFAULT_DATASTORE": "local"}, writer)


@pytest.fixture(scope="session", autouse=True)
def create_summarization_data():
    """Creates fake data files for summarization task"""
    samples = {
        "summarization_1": [
            {"text": "Hello my name is John", "summary": "John"},
            {"text": "Hello my name is Jules", "summary": "Jules"},
        ]
    }
    for key, value in samples.items():
        with open(f"/tmp/{key}.jsonl", "w") as writer:
            for entry in value:
                json.dump(entry, writer)
                writer.write("\n")
