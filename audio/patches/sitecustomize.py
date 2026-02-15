"""
vLLM runtime monkeypatches for Strix Halo.

This repository is meant to be published, so keep patches:
- small
- well-scoped
- easy to revert

Patch: Voxtral Realtime (Whisper block-pooling attention) on ROCm

vLLM's Whisper block-pooling wrapper currently only supports backends that
subclass `FlashAttentionBackend`. On ROCm, the AITER attention backends use the
same KV-cache layout and builder/impl interface, but they do not inherit from
`FlashAttentionBackend`, so vLLM rejects them with NotImplementedError.

This patch relaxes the check to allow ROCm AITER backends as well.

Upstream source (as of vLLM v0.16.0rc1.dev* in kyuz0/vllm-therock-gfx1151):
  vllm/model_executor/models/whisper_causal.py:create_whisper_attention_backend_with_block_pooling
"""

from __future__ import annotations

import copy
import functools
from dataclasses import replace

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    CommonAttentionMetadata,
    subclass_attention_backend_with_overrides,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.kv_cache_interface import AttentionSpec


def _try_import(name: str):
    try:
        module_name, attr = name.rsplit(".", 1)
        module = __import__(module_name, fromlist=[attr])
        return getattr(module, attr)
    except Exception:
        return None


_AiterFlashAttentionBackend = _try_import(
    "vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionBackend"
)
_RocmAiterUnifiedAttentionBackend = _try_import(
    "vllm.v1.attention.backends.rocm_aiter_unified_attn.RocmAiterUnifiedAttentionBackend"
)

_SUPPORTED_BASES: tuple[type, ...] = tuple(
    base
    for base in (
        FlashAttentionBackend,
        _AiterFlashAttentionBackend,
        _RocmAiterUnifiedAttentionBackend,
    )
    if base is not None
)


@functools.lru_cache
def create_whisper_attention_backend_with_block_pooling(
    underlying_attn_backend: type[AttentionBackend], block_pool_size: int
) -> type[AttentionBackend]:
    prefix = "WhisperCausalAttentionWithBlockPooling_"
    underlying_builder = underlying_attn_backend.get_builder_cls()
    underlying_impl = underlying_attn_backend.get_impl_cls()

    class WhisperCausalAttentionWithBlockPoolingBuilder(underlying_builder):  # type: ignore
        def __init__(
            self,
            kv_cache_spec: AttentionSpec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            assert kv_cache_spec.num_kv_heads % block_pool_size == 0
            kv_cache_spec = replace(
                kv_cache_spec,
                block_size=kv_cache_spec.block_size * block_pool_size,
                num_kv_heads=kv_cache_spec.num_kv_heads // block_pool_size,
            )
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            new_common_attn_metadata = copy.deepcopy(common_attn_metadata)
            new_common_attn_metadata.query_start_loc *= block_pool_size
            new_common_attn_metadata.query_start_loc_cpu *= block_pool_size
            new_common_attn_metadata.seq_lens *= block_pool_size
            new_common_attn_metadata._seq_lens_cpu *= block_pool_size
            new_common_attn_metadata._num_computed_tokens_cpu *= block_pool_size
            new_common_attn_metadata.num_actual_tokens *= block_pool_size
            new_common_attn_metadata.max_query_len *= block_pool_size
            new_common_attn_metadata.max_seq_len *= block_pool_size
            original_slot_mapping = common_attn_metadata.slot_mapping
            common_prefix_len *= block_pool_size
            new_common_attn_metadata.slot_mapping = (
                (
                    original_slot_mapping.unsqueeze(1) * block_pool_size
                    + torch.arange(block_pool_size, device=original_slot_mapping.device)
                )
                .flatten()
                .clamp(min=-1)
            )
            return super().build(
                common_prefix_len, new_common_attn_metadata, fast_build
            )

    # NOTE: We need a custom impl so we can use the transformed slot_mapping
    # computed by `WhisperCausalAttentionWithBlockPoolingBuilder` instead of
    # the one from `forward_context.slot_mapping` (gpu_model_runner).
    # This follows the same pattern as CrossAttentionImpl.
    class WhisperCausalAttentionWithBlockPoolingImpl(underlying_impl):  # type: ignore[valid-type,misc]
        def forward(
            self,
            layer: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            output: torch.Tensor | None = None,
            output_scale: torch.Tensor | None = None,
            output_block_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if (
                not underlying_attn_backend.forward_includes_kv_cache_update
                and attn_metadata is not None
            ):
                self.do_kv_cache_update(
                    layer, key, value, kv_cache, attn_metadata.slot_mapping
                )

            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

    if _SUPPORTED_BASES and not issubclass(underlying_attn_backend, _SUPPORTED_BASES):
        raise NotImplementedError(
            f"{underlying_attn_backend} is not yet supported."
            " Contributions to support more backends are much appreciated."
        )

    attn_backend = subclass_attention_backend_with_overrides(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        overrides={
            "get_builder_cls": lambda: WhisperCausalAttentionWithBlockPoolingBuilder,
            "get_impl_cls": lambda: WhisperCausalAttentionWithBlockPoolingImpl,
            # All AITER/FlashAttention-like backends in vLLM currently share the same
            # (2, num_blocks, block_size, num_kv_heads, head_size) KV-cache layout.
            "get_kv_cache_shape": lambda num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str: (
                2,
                num_blocks,
                # we stretch each block by `block_pool_size`
                block_size * block_pool_size,
                num_kv_heads // block_pool_size,
                head_size,
            ),
            "forward_includes_kv_cache_update": True,
        },
    )

    return attn_backend


def _apply_patch() -> None:
    # Import late: importing vLLM modules may perform ROCm platform detection.
    from vllm.model_executor.models import whisper_causal as whisper_causal_mod

    whisper_causal_mod.create_whisper_attention_backend_with_block_pooling = (
        create_whisper_attention_backend_with_block_pooling
    )


_apply_patch()
