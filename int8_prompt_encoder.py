"""Gemma text encoder with int8 weight-only quantization on a single XPU.

Quantizes all nn.Linear weights in the Gemma-3-12B model to int8 with
per-output-channel scales at load time. At forward time, weights are
dequantized to bf16 and run through standard F.linear (bf16 GEMM,
hardware-accelerated on Battlemage XPU).

Storage: ~13 GB (int8) vs ~26 GB (bf16) — fits a single 24 GB B60.
Compute: bf16-speed (same as fp8-cast pattern).
Accuracy: per-channel scaling, max reconstruction error ~0.0004.

This eliminates the 45 s dispatch overhead of DP sharding (xpu:2+xpu:3)
at the cost of a small quantization error in the text encoder output.
"""
from __future__ import annotations

import logging

import torch
from torch import nn

from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder

logger = logging.getLogger(__name__)


class Int8WeightOnlyLinear(nn.Linear):
    """nn.Linear storing weights as int8 + per-channel scale.

    Weight is dequantized to input dtype at forward time and run through
    standard F.linear. Uses __class__ reassignment (like Fp8CastLinear) so
    the weight Parameter slot is reused with int8 data.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002, type: ignore[override]
        w_bf16 = self.weight.to(input.dtype) * self.weight_scale.to(input.dtype)
        b_bf16 = self.bias if self.bias is None else self.bias.to(input.dtype)
        return torch.nn.functional.linear(input, w_bf16, b_bf16)


def _quantize_tensor_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight tensor to int8 with per-output-row scaling.

    Returns (int8_tensor, scale) where scale is (out_features, 1).
    Dequantize: tensor_bf16 = int8_tensor.to(bf16) * scale.to(bf16)
    """
    assert tensor.ndim == 2
    abs_max = tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = (abs_max / 127.0).to(torch.float32)
    int8_tensor = (tensor.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return int8_tensor, scale


def quantize_linear_to_int8(layer: nn.Linear) -> None:
    """Replace an nn.Linear's weight with int8 + per-channel scale, in-place.

    Overwrites the original bf16 weight Parameter with int8 data (same slot),
    adds a weight_scale Parameter, and reassigns __class__ to
    Int8WeightOnlyLinear. Bias stays in its original dtype (small, negligible).
    """
    w = layer.weight.data  # (out_features, in_features) bf16
    int8_w, scale_w = _quantize_tensor_int8(w)

    # Overwrite the weight Parameter slot with int8 data
    layer.weight = nn.Parameter(int8_w, requires_grad=False)
    # Add per-channel scale as a new Parameter
    layer.weight_scale = nn.Parameter(scale_w, requires_grad=False)
    # Bias stays as-is (small, negligible memory)

    layer.__class__ = Int8WeightOnlyLinear


def quantize_model_int8(model: nn.Module, skip_keywords: tuple[str, ...] = ()) -> int:
    """Replace all nn.Linear layers in the model with Int8WeightOnlyLinear.

    Skips layers whose module name contains any of the skip_keywords
    (e.g. 'embed_tokens', 'lm_head' to keep them in higher precision).

    Returns the number of layers quantized.
    """
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, Int8WeightOnlyLinear):
            continue
        if any(kw in name for kw in skip_keywords):
            continue
        quantize_linear_to_int8(module)
        count += 1
    return count


class Int8PromptEncoder:
    """Prompt encoder that runs Gemma in int8 weight-only on a single XPU.

    Builds Gemma on CPU (bf16), quantizes all Linear weights to int8
    (per-channel), moves to a single XPU, runs the forward pass, then frees.
    The embeddings processor runs on proc_device.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        dtype: torch.dtype,
        gemma_device: torch.device,
        proc_device: torch.device | None = None,
        registry=None,
    ) -> None:
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
        from ltx_core.loader.registry import DummyRegistry
        from ltx_core.text_encoders.gemma import (
            EMBEDDINGS_PROCESSOR_KEY_OPS,
            GEMMA_LLM_KEY_OPS,
            GEMMA_MODEL_OPS,
            EmbeddingsProcessorConfigurator,
            GemmaTextEncoderConfigurator,
            module_ops_from_gemma_root,
        )
        from ltx_core.utils import find_matching_file

        self._gemma_root = gemma_root
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._gemma_device = gemma_device
        self._proc_device = proc_device or gemma_device
        self._registry = registry or DummyRegistry()

        module_ops = module_ops_from_gemma_root(gemma_root)
        model_folder = find_matching_file(gemma_root, "model*.safetensors").parent
        weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]

        self._text_encoder_builder = Builder(
            model_path=tuple(weight_paths),
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_sd_ops=GEMMA_LLM_KEY_OPS,
            module_ops=(GEMMA_MODEL_OPS, *module_ops),
            registry=self._registry,
        )
        self._embeddings_processor_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            registry=self._registry,
        )

    def _build_and_quantize_gemma(self) -> torch.nn.Module:
        """Build Gemma on CPU (bf16), quantize to int8, move to XPU."""
        logger.info("Building text encoder on CPU from %s", self._gemma_root)
        model = self._text_encoder_builder.build(
            device=torch.device("cpu"), dtype=self._dtype
        ).eval()

        logger.info("Quantizing Linear weights to int8 (weight-only, per-channel)...")
        n_quantized = quantize_model_int8(
            model,
            skip_keywords=("embed_tokens", "lm_head", "vision_tower", "multi_modal_projector"),
        )
        logger.info("Quantized %d Linear layers to int8", n_quantized)

        # Measure size before/after
        int8_bytes = 0
        other_bytes = 0
        for p in model.parameters():
            if p.dtype == torch.int8:
                int8_bytes += p.numel()
            else:
                other_bytes += p.numel() * p.element_size()
        logger.info(
            "int8 params: %.2f GB | other params: %.2f GB | total: ~%.2f GB",
            int8_bytes / 1024**3,
            other_bytes / 1024**3,
            (int8_bytes + other_bytes) / 1024**3,
        )

        logger.info("Moving int8 model to %s...", self._gemma_device)
        model = model.to(self._gemma_device)
        alloc = torch.xpu.memory_allocated(self._gemma_device) / 1024**3
        logger.info("  %s allocated: %.2f GB", self._gemma_device, alloc)
        return model

    def _free_gemma(self, model: torch.nn.Module) -> None:
        from ltx_pipelines.utils.helpers import cleanup_memory

        model.to("meta")
        del model
        cleanup_memory()
        torch.xpu.empty_cache()

    def _build_embeddings_processor(self) -> torch.nn.Module:
        return self._embeddings_processor_builder.build(
            device=self._proc_device, dtype=self._dtype
        ).eval()

    def __call__(self, prompts, *, enhance_first_prompt=False, enhance_prompt_image=None, enhance_prompt_seed=42):
        """Encode prompts through int8 Gemma -> embeddings processor."""
        from ltx_pipelines.utils.helpers import cleanup_memory
        from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
        from ltx_core.utils import find_matching_file

        text_encoder = self._build_and_quantize_gemma()
        try:
            # Call the model directly (bypassing GemmaTextEncoder.encode which
            # uses self.model.device — unreliable when weights are int8).
            inner_model = text_encoder.model  # Gemma3ForConditionalGeneration
            tokenizer = text_encoder.tokenizer

            tokenized = [tokenizer.tokenize_with_weights(t)["gemma"] for t in prompts]
            input_ids = torch.tensor(
                [[tok for tok, _ in pairs] for pairs in tokenized],
                device=self._gemma_device,
            )
            attention_mask = torch.tensor(
                [[w for _, w in pairs] for pairs in tokenized],
                device=self._gemma_device,
            )
            logger.info("Running int8 Gemma forward on %s (input_ids %s on %s)...",
                        self._gemma_device, tuple(input_ids.shape), input_ids.device)
            outputs = inner_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states
            raw_outputs = [
                (tuple(h[i : i + 1] for h in hidden_states), attention_mask[i : i + 1])
                for i in range(len(prompts))
            ]
            del outputs
        finally:
            self._free_gemma(text_encoder)

        logger.info("Building embeddings processor from %s", self._checkpoint_path)
        proc = self._build_embeddings_processor()
        try:
            results = []
            for hs, mask in raw_outputs:
                hs_on_proc = tuple(h.to(self._proc_device) for h in hs)
                mask_on_proc = mask.to(self._proc_device)
                results.append(proc.process_hidden_states(hs_on_proc, mask_on_proc))
        finally:
            proc.to("meta")
            del proc
            cleanup_memory()
        logger.info("Prompt encoding complete")
        return results
