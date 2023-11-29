import os
from IPython.display import display
from transformers import TrainerCallback, pipeline
from langtest import Harness


class LangTestCallback(TrainerCallback):
    def __init__(
        self,
        task,
        config=None,
        data=None,
        print_reports=True,
        save_reports=False,
        run_each_epoch=False,
    ) -> None:
        super().__init__()
        self.task = task
        self.config = config
        self.data = data
        self.print_reports = print_reports
        self.save_reports = save_reports
        self.run_each_epoch = run_each_epoch
        self.epoch_number = 0
        self.harness = None

    def on_init_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]

        if self.task == "ner":
            model = pipeline("ner", model=model, tokenizer=tokenizer, ignore_labels=[])
        elif self.task == "text-classification":
            model = pipeline("text-classification", model=model, tokenizer=tokenizer)

        self.harness = Harness(
            self.task,
            {
                "model": model,
                "hub": "huggingface",
            },
            self.data,
            self.config,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_number += 1

        if self.run_each_epoch:
            self.harness._generated_results = None
            self.harness.run()
            if self.print_reports:
                display(self.harness.report())

            if self.save_reports:
                if not os.path.exists("reports"):
                    os.mkdir("reports")
                self.harness.report(
                    format="markdown",
                    save_dir=f"reports/report{self.epoch_number:03d}.md",
                )

    def on_train_begin(self, args, state, control, **kwargs):
        self.harness.generate()

    def on_train_end(self, args, state, control, **kwargs):
        if self.run_each_epoch:
            return

        if self.print_reports:
            display(self.harness.run().report())
        if self.save_reports:
            if not os.path.exists("reports"):
                os.mkdir("reports")
            self.harness.report(
                format="markdown", save_dir=f"reports/report{self.epoch_number:03d}.md"
            )
