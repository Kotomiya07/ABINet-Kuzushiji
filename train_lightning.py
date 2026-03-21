import os
import logging
from pathlib import Path
from typing import Any

import hydra
from hydra import utils as hydra_utils
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lit_datamodule import ABINetDataModule
from lit_module import ABINetLightningModule, _load_model
from lit_module_language import LanguageLightningModule
from runtime_utils import apply_runtime_overrides, maybe_compile_module, warn_if_runtime_mismatch
from utils import Config

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
# 畳み込みの最適アルゴリズムを動的選択し高速化
import torch

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def _validate_checkpoint(path_value, name):
    if path_value is None:
        return
    checkpoint_path = Path(path_value)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{name} checkpoint not found: {checkpoint_path}")


def _validate_training_config(config):
    stage = config.global_stage
    vision_ckpt = getattr(config, "model_vision_checkpoint", None)
    language_ckpt = getattr(config, "model_language_checkpoint", None)

    _validate_checkpoint(vision_ckpt, "vision")
    _validate_checkpoint(language_ckpt, "language")

    if stage == "train-super" and getattr(config, "training_require_pretrained", False):
        missing = []
        if vision_ckpt is None:
            missing.append("model.vision.checkpoint")
        if language_ckpt is None:
            missing.append("model.language.checkpoint")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                "Pretrained checkpoints are required for this config. "
                f"Set the following values explicitly: {joined}"
            )


def _log_training_summary(config):
    logging.info(
        "Training summary: stage=%s image_size=%sx%s multiscales=%s rotate_if_vertical=%s rotate_direction=%s",
        config.global_stage,
        getattr(config, "dataset_image_height", None),
        getattr(config, "dataset_image_width", None),
        getattr(config, "dataset_multiscales", None),
        getattr(config, "dataset_rotate_if_vertical", False),
        getattr(config, "dataset_rotate_direction", "ccw"),
    )
    logging.info(
        "Checkpoint summary: vision=%s language=%s require_pretrained=%s",
        getattr(config, "model_vision_checkpoint", None),
        getattr(config, "model_language_checkpoint", None),
        getattr(config, "training_require_pretrained", False),
    )


def _to_serializable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    return value


def _flatten_dict(data, prefix=""):
    flat = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, full_key))
        else:
            flat[full_key] = _to_serializable(value)
    return flat


def _build_wandb_config(base_config, hydra_cfg):
    base_items = {
        key: value
        for key, value in vars(base_config).items()
        if not key.startswith("_")
    }
    hydra_items = OmegaConf.to_container(hydra_cfg, resolve=True)
    return {
        **_flatten_dict({"train": base_items}),
        **_flatten_dict({"hydra": hydra_items}),
    }


@hydra.main(config_path="configs", config_name="lightning.yaml", version_base=None)
def main(cfg):
    # Hydraはデフォルトで作業ディレクトリを移動するので元に戻す
    os.chdir(hydra_utils.get_original_cwd())

    # 従来のYAMLを読み込み（テンプレートも適用）
    base_config = Config(cfg.config_path)
    apply_runtime_overrides(base_config, getattr(cfg, "runtime", None))

    # Hydra側からepochを上書き可能に
    if cfg.trainer.max_epochs is not None:
        base_config.training_epochs = cfg.trainer.max_epochs

    _validate_training_config(base_config)
    _log_training_summary(base_config)

    pl.seed_everything(base_config.global_seed or 42, workers=True)

    datamodule = ABINetDataModule(base_config)
    # stage に応じて Module を切り替える
    if base_config.global_stage == 'pretrain-language':
        model = LanguageLightningModule(base_config, _load_model(base_config))
    else:
        model = ABINetLightningModule(base_config)
    model.model = maybe_compile_module(model.model, base_config)

    # 監視メトリクスはステージごとに切り替える
    if base_config.global_stage == "pretrain-language":
        monitor_metric = f"val/top{model.top_k}_acc"
        ckpt_name = f"lang-{{epoch:02d}}-{{val_top{model.top_k}_acc:.4f}}"
    else:
        monitor_metric = "val/cwr"
        ckpt_name = "abinet-{epoch:02d}-{val_cwr:.4f}"

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor=monitor_metric,
            mode="max",
            save_top_k=1,
            filename=ckpt_name,
        ),
    ]

    wandb_logger = None
    if cfg.wandb.enable:
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
            log_model=True,
        )
        wandb_config = _build_wandb_config(base_config, cfg)
        wandb_logger.experiment.config.update(wandb_config, allow_val_change=True)
        wandb_logger.log_hyperparams(wandb_config)

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=base_config.training_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=base_config.optimizer_clip_grad,
        val_check_interval=getattr(cfg.trainer, "val_check_interval", 1.0),
        num_sanity_val_steps=getattr(cfg.trainer, "num_sanity_val_steps", 2),
        callbacks=callbacks,
        logger=wandb_logger,
    )
    warn_if_runtime_mismatch(cfg.trainer, base_config)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
