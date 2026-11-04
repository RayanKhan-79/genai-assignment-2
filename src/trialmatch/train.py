"""LoRA fine-tuning of the Qwen base model with completion-only loss.

Only a ``--smoke`` run (2 optimisation steps, 8 examples, tiny random subset) is executed by the
agent; the real training job runs on Colab T4 (see README, "Long-running jobs").
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from trialmatch import constants as c
from trialmatch.data import Annotation, AnnotationLoader, PromptBuilder, PromptEncoder, SplitManager
from trialmatch.modeling import LoadedModel, ModelLoader


class TrainingDataBuilder:
    """Turn annotations into a tokenised dataset with prompt tokens masked from the loss."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Store the tokenizer and build the encoder once."""
        self._encoder = PromptEncoder(tokenizer, PromptBuilder())

    def build(self, annotations: Sequence[Annotation]) -> Dataset:
        """Return a dataset with ``input_ids``, ``attention_mask`` and ``labels`` columns."""
        records = [self._encoder.encode_train(PromptBuilder().build(a)) for a in annotations]
        return Dataset.from_list(records)


class TrainCurvePlotter:
    """Render training/validation loss curves from ``Trainer.state.log_history``."""

    @staticmethod
    def plot(history: Sequence[dict[str, Any]], path: Path = c.TRAIN_CURVES_PATH) -> None:
        """Write a PDF with training loss per step and validation loss per epoch."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train = [(entry["step"], entry["loss"]) for entry in history if "loss" in entry]
        val = [(entry["step"], entry["eval_loss"]) for entry in history if "eval_loss" in entry]
        figure, axis = plt.subplots(figsize=(6, 3.5))
        if train:
            axis.plot([s for s, _ in train], [v for _, v in train], label="train loss")
        if val:
            axis.plot([s for s, _ in val], [v for _, v in val], marker="o", label="val loss")
        axis.set_xlabel("optimisation step")
        axis.set_ylabel("cross-entropy")
        axis.set_title("LoRA fine-tuning")
        axis.legend()
        axis.grid(alpha=0.3)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)


class FineTuner:
    """Configure and run LoRA fine-tuning with the Hugging Face ``Trainer``."""

    def __init__(
        self,
        output_dir: Path = c.ADAPTER_DIR,
        config: c.TrainConfig = c.TRAINING,
        model_name: str = c.BASE_MODEL_NAME,
        smoke: bool = False,
    ) -> None:
        """Store hyper-parameters; ``smoke`` switches to the 0.5B model with 2 steps."""
        self._output_dir = output_dir
        self._config = config
        self._model_name = c.SMOKE_MODEL_NAME if smoke else model_name
        self._smoke = smoke

    def run(
        self,
        train_annotations: Sequence[Annotation],
        val_annotations: Sequence[Annotation],
        resume: bool = False,
    ) -> Path:
        """Fine-tune and save the adapter, returning its directory."""
        config = self._config
        if self._smoke:
            config = dataclasses.replace(
                config, per_device_batch_size=1, gradient_accumulation_steps=1
            )
        train_items = list(train_annotations)
        val_items = list(val_annotations)
        if self._smoke:
            train_items = train_items[: config.smoke_num_examples]
            val_items = val_items[: config.smoke_num_examples]

        # Smoke runs use the CPU: MPS shares system memory and the fp32 LM head pushes a
        # 0.5B model's backward pass past the MPS watermark on this machine.
        device = torch.device("cpu") if self._smoke else None
        loaded = ModelLoader(model_name=self._model_name, device=device).load()
        tokenizer = loaded.tokenizer
        lora = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(loaded.model, lora)
        if config.gradient_checkpointing and loaded.device.type == "cuda":
            model.enable_input_require_grads()

        builder = TrainingDataBuilder(tokenizer)
        train_dataset = builder.build(train_items)
        eval_dataset = builder.build(val_items)
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100
        )
        args = self._training_arguments(len(train_dataset), loaded)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            processing_class=tokenizer,
        )
        trainer.train(resume_from_checkpoint=resume or None)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self._output_dir))
        tokenizer.save_pretrained(self._output_dir)
        self._write_log(trainer.state.log_history)
        return self._output_dir

    def _training_arguments(self, n_train: int, loaded: LoadedModel) -> TrainingArguments:
        """Build ``TrainingArguments``, including the warm-up step count (no warmup_ratio in v5)."""
        config = self._config
        steps_per_epoch = max(
            1,
            math.ceil(
                n_train / (config.per_device_batch_size * config.gradient_accumulation_steps)
            ),
        )
        total_steps = config.smoke_max_steps if self._smoke else steps_per_epoch * config.num_epochs
        use_cuda = loaded.device.type == "cuda"
        common: dict[str, Any] = {
            "output_dir": str(self._output_dir),
            "per_device_train_batch_size": config.per_device_batch_size,
            "per_device_eval_batch_size": config.per_device_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "lr_scheduler_type": config.lr_scheduler,
            "warmup_steps": max(1, int(config.warmup_ratio * total_steps)),
            "weight_decay": config.weight_decay,
            "max_grad_norm": config.max_grad_norm,
            "fp16": config.fp16 and use_cuda,
            "gradient_checkpointing": config.gradient_checkpointing and use_cuda,
            "logging_steps": 1 if self._smoke else config.logging_steps,
            "save_total_limit": config.save_total_limit,
            "report_to": [],
            "seed": c.SEED,
            "data_seed": c.SEED,
            "remove_unused_columns": False,
            "use_cpu": self._smoke,
        }
        if self._smoke:
            return TrainingArguments(
                **common,
                max_steps=config.smoke_max_steps,
                eval_strategy="no",
                save_strategy="no",
                load_best_model_at_end=False,
            )
        return TrainingArguments(
            **common,
            num_train_epochs=config.num_epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    def _write_log(self, history: Sequence[dict[str, Any]]) -> None:
        """Persist the trainer log history and the loss figure.

        Smoke runs keep their artefacts inside the (git-ignored) output directory so that
        ``results/`` only ever contains real Colab runs.
        """
        log_path = c.TRAIN_LOG_PATH
        figure_path = c.TRAIN_CURVES_PATH
        if self._smoke:
            log_path = self._output_dir / "train_log.json"
            figure_path = self._output_dir / "train_curves.pdf"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(list(history), indent=2) + "\n", encoding="utf-8")
        TrainCurvePlotter.plot(history, figure_path)


class Sweeper:
    """Optional 4-configuration sweep selected by validation macro-F1 only."""

    def __init__(
        self,
        config: c.TrainConfig = c.TRAINING,
        model_name: str = c.BASE_MODEL_NAME,
        root: Path = c.CHECKPOINTS_DIR / "sweep",
    ) -> None:
        """Store the base configuration and the sweep output root."""
        self._config = config
        self._model_name = model_name
        self._root = root

    def run(
        self, train_annotations: Sequence[Annotation], val_annotations: Sequence[Annotation]
    ) -> Path:
        """Train every configuration, score it on validation, and return the best adapter path."""
        from trialmatch.evaluate import Evaluator

        scores: list[dict[str, Any]] = []
        for rank, (rank_value, learning_rate) in enumerate(self._config.sweep_configs):
            tuned = dataclasses.replace(
                self._config, lora_r=rank_value, learning_rate=learning_rate
            )
            output = self._root / f"r{rank_value}_lr{learning_rate:.0e}"
            FineTuner(output_dir=output, config=tuned, model_name=self._model_name).run(
                train_annotations, val_annotations
            )
            metrics = Evaluator(split="val", model_tag=c.MODEL_TAG_FINETUNED, adapter=output).run()
            scores.append(
                {
                    "rank": rank,
                    "lora_r": rank_value,
                    "learning_rate": learning_rate,
                    "val_macro_f1": metrics["classification"]["overall"]["macro avg"]["f1-score"],
                    "adapter": str(output),
                }
            )
        c.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (c.RESULTS_DIR / "sweep.json").write_text(json.dumps(scores, indent=2) + "\n", "utf-8")
        best = max(scores, key=lambda item: float(item["val_macro_f1"]))
        return Path(str(best["adapter"]))


def main() -> None:
    """CLI: ``python -m trialmatch.train [--smoke] [--resume] [--sweep] [--model base|draft]``."""
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for criterion matching.")
    parser.add_argument("--output-dir", type=Path, default=c.ADAPTER_DIR)
    parser.add_argument("--model", choices=["base", "draft"], default="base")
    parser.add_argument("--resume", action="store_true", help="resume from the last checkpoint")
    parser.add_argument("--smoke", action="store_true", help="2 steps, 8 examples, 0.5B model")
    parser.add_argument("--sweep", action="store_true", help="run the 4-config sweep")
    args = parser.parse_args()

    annotations = AnnotationLoader().load()
    manager = SplitManager()
    manifest = manager.load() if c.SPLIT_MANIFEST_PATH.exists() else manager.build(annotations)
    grouped = manager.assign(annotations, manifest)
    model_name = c.DRAFT_MODEL_NAME if args.model == "draft" else c.BASE_MODEL_NAME

    if args.sweep:
        best = Sweeper(model_name=model_name).run(grouped["train"], grouped["val"])
        print(f"best adapter: {best}")
        return
    output_dir = c.CHECKPOINTS_DIR / "smoke" if args.smoke else args.output_dir
    output = FineTuner(output_dir=output_dir, model_name=model_name, smoke=args.smoke).run(
        grouped["train"], grouped["val"], resume=args.resume
    )
    print(f"adapter written to {output}")


if __name__ == "__main__":
    main()
