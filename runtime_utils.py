import logging
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import torch

_CURRENT_ATTENTION_BACKEND = ContextVar("current_attention_backend", default="auto")


def _ifnone(value, default):
    return default if value is None else value


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
        "fullgraph_train",
        "fast_train_backend",
    ]
    for key in runtime_items:
        if hasattr(runtime_cfg, key):
            setattr(base_config, f"runtime_{key}", getattr(runtime_cfg, key))


def maybe_compile_module(module, config):
    compile_enabled = _ifnone(getattr(config, "runtime_compile_enabled", False), False)
    compile_backend = _ifnone(getattr(config, "runtime_compile_backend", "inductor"), "inductor")
    compile_mode = _ifnone(getattr(config, "runtime_compile_mode", "reduce-overhead"), "reduce-overhead")
    compile_fullgraph = _ifnone(getattr(config, "runtime_compile_fullgraph", False), False)
    compile_dynamic = _ifnone(getattr(config, "runtime_compile_dynamic", False), False)

    if getattr(config, "runtime_fullgraph_train", False):
        compile_enabled = True
        compile_fullgraph = True
        compile_dynamic = False
        if compile_mode == "reduce-overhead":
            compile_mode = "default"

    if not compile_enabled:
        return module
    if not hasattr(torch, "compile"):
        raise RuntimeError("`torch.compile` is not available in this PyTorch build.")

    logging.info(
        "Compile model with backend=%s mode=%s fullgraph=%s dynamic=%s",
        compile_backend,
        compile_mode,
        compile_fullgraph,
        compile_dynamic,
    )
    return torch.compile(
        module,
        backend=compile_backend,
        mode=compile_mode,
        fullgraph=compile_fullgraph,
        dynamic=compile_dynamic,
    )


def warn_if_runtime_mismatch(trainer_cfg, config):
    backend = _ifnone(getattr(config, "runtime_attention_backend", "auto"), "auto")
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
    backend_name = _ifnone(getattr(config, "runtime_attention_backend", "auto"), "auto")
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
