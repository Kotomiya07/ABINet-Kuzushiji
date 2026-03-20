import logging
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import torch

_CURRENT_ATTENTION_BACKEND = ContextVar("current_attention_backend", default="auto")


def apply_runtime_overrides(base_config, runtime_cfg):
    if runtime_cfg is None:
        return

    runtime_items = [
        "attention_backend",
        "compile_enabled",
        "compile_backend",
        "compile_mode",
        "compile_fullgraph",
        "compile_dynamic",
    ]
    for key in runtime_items:
        if hasattr(runtime_cfg, key):
            setattr(base_config, f"runtime_{key}", getattr(runtime_cfg, key))


def maybe_compile_module(module, config):
    if not getattr(config, "runtime_compile_enabled", False):
        return module
    if not hasattr(torch, "compile"):
        raise RuntimeError("`torch.compile` is not available in this PyTorch build.")

    logging.info(
        "Compile model with backend=%s mode=%s fullgraph=%s dynamic=%s",
        getattr(config, "runtime_compile_backend", "inductor"),
        getattr(config, "runtime_compile_mode", "reduce-overhead"),
        getattr(config, "runtime_compile_fullgraph", False),
        getattr(config, "runtime_compile_dynamic", False),
    )
    return torch.compile(
        module,
        backend=getattr(config, "runtime_compile_backend", "inductor"),
        mode=getattr(config, "runtime_compile_mode", "reduce-overhead"),
        fullgraph=getattr(config, "runtime_compile_fullgraph", False),
        dynamic=getattr(config, "runtime_compile_dynamic", False),
    )


def warn_if_runtime_mismatch(trainer_cfg, config):
    backend = getattr(config, "runtime_attention_backend", "auto")
    precision = getattr(trainer_cfg, "precision", 32)
    if backend in {"auto", "flash_only"} and str(precision) in {"32", "32-true"}:
        logging.warning(
            "attention_backend=%s but trainer.precision=%s. FlashAttention kernels usually require "
            "fp16/bf16, so PyTorch may fall back to non-flash SDPA.",
            backend,
            precision,
        )


def _resolve_sdpa_backends(backend_name):
    from torch.nn.attention import SDPBackend

    if backend_name == "flash_only":
        return [SDPBackend.FLASH_ATTENTION]
    if backend_name == "math_only":
        return [SDPBackend.MATH]
    if backend_name == "auto":
        return [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.MATH,
        ]
    raise ValueError(f"Unsupported attention backend: {backend_name}")


def get_runtime_attention_backend():
    return _CURRENT_ATTENTION_BACKEND.get()


@contextmanager
def sdpa_context(config):
    backend_name = getattr(config, "runtime_attention_backend", "auto")
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        with nullcontext():
            yield
        return

    from torch.nn.attention import sdpa_kernel

    backends = _resolve_sdpa_backends(backend_name)
    token = _CURRENT_ATTENTION_BACKEND.set(backend_name)
    try:
        with sdpa_kernel(backends, set_priority=True):
            yield
    finally:
        _CURRENT_ATTENTION_BACKEND.reset(token)
