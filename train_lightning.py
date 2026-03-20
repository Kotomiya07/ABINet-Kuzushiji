import os

import hydra
from hydra import utils as hydra_utils
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
torch.backends.cudnn.benchmark = True
# Ampere以降でTF32を許可し、行列演算を高速化
if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
    torch.backends.cuda.matmul.fp32_precision = "tf32"
else:
    torch.backends.cuda.matmul.allow_tf32 = True

if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
    torch.backends.cudnn.conv.fp32_precision = "tf32"
if hasattr(torch.backends.cudnn, "rnn") and hasattr(torch.backends.cudnn.rnn, "fp32_precision"):
    torch.backends.cudnn.rnn.fp32_precision = "tf32"
if hasattr(torch.backends.cudnn, "fp32_precision"):
    torch.backends.cudnn.fp32_precision = "tf32"
elif hasattr(torch.backends.cudnn, "allow_tf32"):
    torch.backends.cudnn.allow_tf32 = True


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
