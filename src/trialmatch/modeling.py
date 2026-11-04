"""Model loading: base weights, optional LoRA adapter, optional merge, device/dtype selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from trialmatch import constants as c


@dataclass(frozen=True)
class LoadedModel:
    """A model/tokenizer pair ready for evaluation or serving.

    Attributes:
        model: The causal LM (possibly a merged adapter).
        tokenizer: Its tokenizer with a defined padding token.
        device: Device the weights live on.
        dtype: Weight dtype (fp16 on CUDA, fp32 elsewhere).
        base_name: Hugging Face name of the base checkpoint.
        adapter: Adapter path or Hub ID that was applied, if any.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype
    base_name: str
    adapter: str | None = None

    def generate(self, **kwargs: Any) -> torch.Tensor:
        """Call the Hugging Face ``generate`` method and return the generated token IDs.

        ``transformers`` 5.x annotates the ``generate`` attribute as ``Tensor | Module``, so it is
        fetched dynamically for the type checker and the result cast back to a tensor.
        """
        generator = cast(Callable[..., Any], self.model.generate)
        return cast(torch.Tensor, generator(**kwargs))


class ModelLoader:
    """Load a Qwen checkpoint on the best available device, optionally with a LoRA adapter."""

    def __init__(
        self,
        model_name: str = c.BASE_MODEL_NAME,
        adapter: str | Path | None = None,
        merge_adapter: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """Configure the loader; no weights are read until :meth:`load`.

        Args:
            model_name: Hugging Face checkpoint name (base model).
            adapter: Local adapter directory or Hub repository ID to apply.
            merge_adapter: Merge the adapter into the base weights (needed for cache work).
            device: Force a device; ``None`` selects CUDA, then MPS, then CPU.
        """
        self._model_name = model_name
        self._adapter = str(adapter) if adapter is not None else None
        self._merge_adapter = merge_adapter
        self._device = device

    @staticmethod
    def resolve_device() -> tuple[torch.device, torch.dtype]:
        """Return ``(device, dtype)``: CUDA in fp16, otherwise MPS/CPU in fp32.

        MPS is used when available (Apple silicon) but stays in fp32 because half precision
        there is numerically fragile; the T4 target runs fp16 on CUDA.
        """
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.float16
        if torch.backends.mps.is_available():
            return torch.device("mps"), torch.float32
        return torch.device("cpu"), torch.float32

    @staticmethod
    def prepare_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
        """Give the tokenizer a padding token and left padding for batched generation."""
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def load(self) -> LoadedModel:
        """Load the base weights, apply the adapter if configured, and return the bundle."""
        resolved_device, dtype = self.resolve_device()
        device = self._device or resolved_device
        tokenizer = self.prepare_tokenizer(AutoTokenizer.from_pretrained(self._model_name))

        # One GPU (T4, 16 GB) holds the 1.5B fp16 weights with room to spare, so the model is
        # loaded whole and moved explicitly; ``device_map="auto"`` would only add accelerate
        # sharding that the Trainer does not need.
        model: PreTrainedModel = cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(self._model_name, dtype=dtype),
        )
        # transformers 5.x annotates ``.to`` in a way mypy cannot use; cast through Any.
        model = cast(PreTrainedModel, cast(Any, model).to(device))
        if self._adapter is not None:
            peft_model = PeftModel.from_pretrained(model, self._adapter, is_trainable=False)
            merged = peft_model.merge_and_unload() if self._merge_adapter else peft_model
            model = cast(PreTrainedModel, merged)
        cast(Any, model).eval()
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            base_name=self._model_name,
            adapter=self._adapter,
        )
