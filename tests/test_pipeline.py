import os
import pytest
import json
import subprocess
from langtest.pipelines.transformers import ner_pipeline

os.environ["METAFLOW_PROFILE"] = "test"


class TestPipelineNERHf:
    """Test class for NER HF pipelines"""

    @pytest.mark.parametrize(
        "data_path,feature_col,target_col",
        [
            ("tests/fixtures/tner.csv", "tokens", "ner_tags"),
            ("tests/fixtures/test.conll", None, None),
        ],
    )
    def test_workflow(self, data_path: str, feature_col: str, target_col: str):
        """"""
        training_args = {"per_device_train_batch_size": 4, "max_steps": 3}
        flow_path = os.path.abspath(ner_pipeline.__file__)
        cmd = [
            "python",
            flow_path,
            "run",
            "--model-name=dslim/bert-base-NER",
            f"--train-data={data_path}",
            f"--eval-data={data_path}",
            f"--training-args={json.dumps(training_args)}",
            "--run-id-file=test_id",
        ]
        if feature_col is not None:
            cmd.append(f"--feature-col={feature_col}"),
        if target_col is not None:
            cmd.append(f"--target-col={target_col}")

        subprocess.check_call(cmd)
