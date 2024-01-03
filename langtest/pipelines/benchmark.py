from metaflow import FlowSpec, JSONType, Parameter, step

from langtest import Harness
from langtest.pipelines.constant import (
    DEFAULT_CONFIG,
    # BENCHMARK_DATASETS,
)


class LLMBenchmarkPipeline(FlowSpec):
    model_name = Parameter(
        "model-name", help="Name of the pretrained model to load", type=str, required=True
    )

    hub = Parameter("hub", help="Name of the hub to use", type=str, required=True)

    task = Parameter("task", help="Name of the task to use", type=str, required=True)

    output_dir = Parameter(
        "output-dir", help="Name of the output directory", type=str, required=True
    )

    test_config = Parameter(
        "test-config",
        help="Tests configuration in JSON",
        type=JSONType,
        required=False,
        default=DEFAULT_CONFIG,
    )

    @step
    def start(self):
        """Starting step of the flow (required by Metaflow)"""
        self.next(self.setup)

    @step
    def setup(self):
        """Performs all the necessary set up steps"""
        self.harness = Harness(
            task=self.task,
            model={
                "model": self.model_name,
                "hub": self.hub,
            },
            data={"data_source": "BoolQ", "split": "test-tiny"},
            config=self.test_config,
        )
        self.harness.configure(self.test_config)
        self.next(self.generate)

    @step
    def generate(self):
        """Generate the dataset"""
        # self.harness.generate()
        # self.next(self.evaluate)
        self.next(self.end)

    @step
    def end(self):
        """End of the flow"""
        pass


if __name__ == "__main__":
    LLMBenchmarkPipeline()
