import json
import math
from pathlib import Path
from transformers.trainer_callback import DefaultFlowCallback, TrainingArguments, TrainerState, TrainerControl


class CharmerCallback(DefaultFlowCallback):
    def __init__(self):
        self.record_loss = True
        self.args = None
        self.losses = []
        self.perf = {}
        self.train_perf = {}

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        self.args = args
        control.should_log = False

        if logs and "loss" in logs.keys() and self.record_loss:
            self.losses.append(logs)
        elif logs and "train_runtime" in logs.keys():
            self.perf["samples_per_second"] = logs["train_samples_per_second"]
            self.perf["steps_per_second"] = logs["train_steps_per_second"]
            self.perf["loss"] = "{:.2f}".format(logs["train_loss"])
            self.perf["max_memory"] = logs["max_memory_allocated (GB)"]
            self.train_perf = logs
            self.record_loss = False
        elif logs and "eval_loss" in logs.keys():
            try:
                perplexity = math.exp(logs["eval_loss"])
            except OverflowError:
                perplexity = float("inf")

            self.perf["perplexity"] = "{:.2f}".format(perplexity)

        return control

    def write_to_disk(self):
        output_dir = Path("/log")
        output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / f"performance.json").open("w", encoding="utf-8") as f:
            json.dump(self.perf, f, ensure_ascii=False, indent=4)
        with (output_dir / f"train_results.json").open("w", encoding="utf-8") as f:
            json.dump(self.train_perf, f, ensure_ascii=False, indent=4)
        with (output_dir / f"losses.json").open("w", encoding="utf-8") as f:
            json.dump(self.losses, f, ensure_ascii=False, indent=4)

        if self.args:
            with (output_dir / "training_arguments.json").open("w", encoding="utf-8") as f:
                json.dump(self.args.to_dict(), f, ensure_ascii=False, indent=4)
