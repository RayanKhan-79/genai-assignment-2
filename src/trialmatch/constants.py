"""Every project-wide constant (rule R4).

Nothing numeric or string-valued that carries meaning may be hard-coded elsewhere:
import it from here. Frozen dataclasses group related defaults so they can be passed
into classes as configuration objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "report" / "figures"
SPLITS_DIR: Final[Path] = PROJECT_ROOT / "splits"
CHECKPOINTS_DIR: Final[Path] = PROJECT_ROOT / "checkpoints"
ADAPTER_DIR: Final[Path] = CHECKPOINTS_DIR / "adapter"
SPLIT_MANIFEST_PATH: Final[Path] = SPLITS_DIR / "split_manifest.json"
TRUNCATION_REPORT_PATH: Final[Path] = RESULTS_DIR / "truncation_report.json"
TRAIN_LOG_PATH: Final[Path] = RESULTS_DIR / "train_log.json"
TRAIN_CURVES_PATH: Final[Path] = FIGURES_DIR / "train_curves.pdf"

# --------------------------------------------------------------------------------------
# Seeds and splits
# --------------------------------------------------------------------------------------
SEED: Final[int] = 42
N_TRAIN_PATIENTS: Final[int] = 37
N_VAL_PATIENTS: Final[int] = 8
N_TEST_PATIENTS: Final[int] = 8
SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "val", "test")
RESERVED_SPLIT: Final[str] = "test"

# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
DATASET_NAME: Final[str] = "ncbi/TrialGPT-Criterion-Annotations"
#: Fields that leak GPT-4 answers and must never reach an ``Annotation`` or a prompt.
FORBIDDEN_FIELDS: Final[tuple[str, ...]] = (
    "gpt4_explanation",
    "gpt4_eligibility",
    "gpt4_correct",
    "gpt4_score",
    "gpt4_sentences",
    "explanation_correctness",
)

# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------
CRITERION_TYPE_INCLUSION: Final[str] = "inclusion"
CRITERION_TYPE_EXCLUSION: Final[str] = "exclusion"
INCLUSION_LABELS: Final[tuple[str, ...]] = (
    "included",
    "not included",
    "not enough information",
    "not applicable",
)
EXCLUSION_LABELS: Final[tuple[str, ...]] = (
    "excluded",
    "not excluded",
    "not enough information",
    "not applicable",
)
#: Label assigned when the model output cannot be parsed or is not a valid label.
LABEL_INVALID: Final[str] = "invalid"

#: Patient-level screening outcomes produced by :class:`OverallAggregator` (a documented
#: rule, not a ground-truth label).
OUTCOME_ELIGIBLE: Final[str] = "Likely eligible"
OUTCOME_INELIGIBLE: Final[str] = "Likely ineligible"
OUTCOME_NEEDS_INFO: Final[str] = "Needs more information"

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
BASE_MODEL_NAME: Final[str] = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL_NAME: Final[str] = "Qwen/Qwen2.5-0.5B-Instruct"
SMOKE_MODEL_NAME: Final[str] = "Qwen/Qwen2.5-0.5B-Instruct"
#: Hugging Face Hub repository that hosts the fine-tuned LoRA adapter.
ADAPTER_REPO: Final[str] = "RayanKhan-79/genai-assignment-2"
#: Decoding modes accepted by the evaluator and the app.
DECODE_MODES: Final[tuple[str, ...]] = ("hf", "uncached", "cached")
MODEL_TAG_BASE: Final[str] = "base"
MODEL_TAG_FINETUNED: Final[str] = "finetuned"

# --------------------------------------------------------------------------------------
# Prompt templates
# --------------------------------------------------------------------------------------
SYSTEM_PROMPT: Final[str] = (
    "You are a clinical trial screening assistant. You decide whether one patient "
    "satisfies a single eligibility criterion, using only the numbered patient "
    "sentences. Reply with one JSON object and nothing else."
)
PROMPT_TEMPLATE: Final[str] = (
    "Patient note, sentences numbered in square brackets:\n"
    "{sentences}\n\n"
    "Criterion type: {criterion_type}\n"
    "Criterion: {criterion_text}\n\n"
    "Allowed labels: {labels}\n"
    "Answer with exactly one JSON object of the form "
    '{{"label": "<one of the allowed labels>", "evidence_ids": [<sentence numbers>]}} '
    "and nothing else. List the sentence numbers that directly support the label; leave the "
    "evidence list empty when no sentence supports it."
)
#: One numbered sentence inside :data:`PROMPT_TEMPLATE`.
SENTENCE_TEMPLATE: Final[str] = "[{index}] {text}"
EXPLANATION_WITH_EVIDENCE: Final[str] = (
    'Labelled "{label}": the patient statement(s) {ids} read "{evidence}".'
)
EXPLANATION_NO_EVIDENCE: Final[str] = (
    'Labelled "{label}"; no supporting evidence cited; this is not proof of eligibility '
    "or ineligibility."
)


# --------------------------------------------------------------------------------------
# Dataclasses holding grouped defaults
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    """LoRA fine-tuning hyper-parameters (values fixed in PLAN.md section 5)."""

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    fp16: bool = True
    gradient_checkpointing: bool = True
    save_total_limit: int = 2
    logging_steps: int = 5
    smoke_max_steps: int = 2
    smoke_num_examples: int = 8
    sweep_configs: tuple[tuple[int, float], ...] = (
        (8, 1e-4),
        (8, 2e-4),
        (16, 1e-4),
        (16, 2e-4),
    )


@dataclass(frozen=True)
class GenerationConfig:
    """Greedy generation defaults shared by evaluation, benchmarks and the app."""

    max_new_tokens: int = 128
    max_seq_len: int = 2048


@dataclass(frozen=True)
class BenchmarkConfig:
    """Grids and repetition counts for the three benchmark suites."""

    decoding_input_lengths: tuple[int, ...] = (128, 512, 1024)
    decoding_output_lengths: tuple[int, ...] = (32, 64, 128)
    attention_lengths: tuple[int, ...] = (256, 512, 1024, 2048)
    attention_block_q: int = 64
    attention_block_k: int = 64
    speculative_gammas: tuple[int, ...] = (2, 4, 6)
    warmup_runs: int = 2
    timed_runs: int = 5


# --------------------------------------------------------------------------------------
# Numerical tolerances
# --------------------------------------------------------------------------------------
FP32_ATOL: Final[float] = 1e-5
FP16_ATOL: Final[float] = 1e-2

#: Ready-made default configuration instances (frozen, so safe to share).
TRAINING: Final[TrainConfig] = TrainConfig()
GENERATION: Final[GenerationConfig] = GenerationConfig()
BENCHMARK: Final[BenchmarkConfig] = BenchmarkConfig()

# --------------------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------------------
APP_TITLE: Final[str] = "TrialMatch: Educational Trial Screening Prototype"
APP_DISCLAIMER: Final[str] = "Educational screening prototype, not a clinical enrollment decision."
APP_MAX_CRITERIA: Final[int] = 40
