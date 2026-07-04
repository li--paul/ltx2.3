"""Gemma text encoder with model-parallel dispatch across 2 XPUs.

Shards the Gemma-3-12B model across xpu:2 + xpu:3 (NUMA group B) using
accelerate's device_map dispatch, keeping it off xpu:0/xpu:1 (group A,
which hosts the transformer and VAEs respectively).

Gemma-3-12B in bf16 is ~26 GB — too large for a single 24 GB B60. Sharded
across 2 XPUs it fits comfortably (~16.5 GB + ~8.2 GB) and the forward pass
runs in ~0.8 s instead of ~17-22 s on CPU.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import remove_hook_from_module

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_core.utils import find_matching_file
from ltx_pipelines.utils.helpers import cleanup_memory, generate_enhanced_prompt

logger = logging.getLogger(__name__)
T = TypeVar("T")


class DPPromptEncoder:
    """Prompt encoder that shards Gemma across two XPUs via accelerate dispatch.

    Mirrors ``ltx_pipelines.utils.blocks.PromptEncoder`` but builds Gemma on CPU,
    dispatches it across ``gemma_devices`` (default xpu:2+xpu:3), runs the forward
    pass, then frees it and builds the small embeddings processor on ``proc_device``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        dtype: torch.dtype,
        gemma_devices: tuple[int, ...] = (2, 3),
        proc_device: torch.device | None = None,
        registry: Registry | None = None,
        max_memory_per_gpu_gb: float = 20.0,
    ) -> None:
        self._gemma_root = gemma_root
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._gemma_devices = gemma_devices
        self._proc_device = proc_device or torch.device("xpu", gemma_devices[0])
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

        # accelerate max_memory dict: integer keys for XPU devices
        self._max_memory = {d: f"{max_memory_per_gpu_gb}GB" for d in gemma_devices}
        self._max_memory["cpu"] = "2000GB"

    def _build_and_dispatch_gemma(self) -> torch.nn.Module:
        """Build Gemma on CPU, infer device map, dispatch across XPUs."""
        logger.info("Building text encoder on CPU from %s", self._gemma_root)
        model = self._text_encoder_builder.build(
            device=torch.device("cpu"), dtype=self._dtype
        ).eval()

        logger.info("Inferring device map for xpu:%s", ",".join(str(d) for d in self._gemma_devices))
        device_map = infer_auto_device_map(
            model,
            max_memory=self._max_memory,
            no_split_module_classes=["Gemma3DecoderLayer"],
        )
        devs = sorted(set(str(v) for v in device_map.values()))
        logger.info("Device map: %d entries across devices %s", len(device_map), devs)

        logger.info("Dispatching (sharding weights)...")
        model = dispatch_model(model, device_map)
        for d in self._gemma_devices:
            alloc = torch.xpu.memory_allocated(d) / 1024**3
            logger.info("  xpu:%d allocated: %.2f GB", d, alloc)
        return model

    def _free_gemma(self, model: torch.nn.Module) -> None:
        """Remove accelerate hooks, move to meta, free memory."""
        for name, module in model.named_modules():
            remove_hook_from_module(module, recurse=False)
        model.to("meta")
        del model
        cleanup_memory()
        for d in self._gemma_devices:
            torch.xpu.empty_cache()

    def _build_embeddings_processor(self) -> torch.nn.Module:
        return self._embeddings_processor_builder.build(
            device=self._proc_device, dtype=self._dtype
        ).eval()

    def __call__(
        self,
        prompts: list[str],
        *,
        enhance_first_prompt: bool = False,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
    ) -> list[EmbeddingsProcessorOutput]:
        """Encode prompts through sharded Gemma -> embeddings processor."""
        text_encoder = self._build_and_dispatch_gemma()
        try:
            if enhance_first_prompt:
                prompts = list(prompts)
                prompts[0] = generate_enhanced_prompt(
                    text_encoder, prompts[0], enhance_prompt_image, seed=enhance_prompt_seed
                )
            raw_outputs = text_encoder.encode(prompts)
        finally:
            self._free_gemma(text_encoder)

        logger.info("Building embeddings processor from %s", self._checkpoint_path)
        proc = self._build_embeddings_processor()
        try:
            results = []
            for hs, mask in raw_outputs:
                # Move hidden states from Gemma's last device to proc device
                hs_on_proc = tuple(h.to(self._proc_device) for h in hs)
                mask_on_proc = mask.to(self._proc_device)
                results.append(proc.process_hidden_states(hs_on_proc, mask_on_proc))
        finally:
            proc.to("meta")
            del proc
            cleanup_memory()
        logger.info("Prompt encoding complete")
        return results
