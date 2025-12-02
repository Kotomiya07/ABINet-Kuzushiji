import importlib
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler

from losses import MultiLosses
from utils import CharsetMapper, ifnone

# schedulefreeのサポート（オプショナル）
try:
    from schedulefree import RAdamScheduleFree, AdamWScheduleFree
    SCHEDULEFREE_AVAILABLE = True
except ImportError:
    SCHEDULEFREE_AVAILABLE = False


def _load_model(config):
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(config)


class ABINetLightningModule(pl.LightningModule):
    """LightningModule for ABINet (vision + language)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = _load_model(config)
        self.criterion = MultiLosses(one_hot=config.dataset_one_hot_y)
        self.charset = CharsetMapper(
            config.dataset_charset_path,
            max_length=config.dataset_max_length + 1
        )
        # メトリクス計算間隔（デフォルトは1で毎ステップ）
        self.metric_interval = getattr(config, 'training_metric_interval', 1) or 1
        # Config は独自クラスで deepcopy に失敗するため除外
        self.save_hyperparameters(ignore=['model', 'config'])

    def forward(self, images):
        return self.model(images)

    # --- metrics ---
    def _pick_output(self, outputs: Any) -> Dict:
        # handle nested outputs from iterative model
        if isinstance(outputs, (list, tuple)):
            first = outputs[0]
            if isinstance(first, (list, tuple)):  # ([a_res...], [l_res...], v_res)
                return first[-1]
            for res in outputs:
                if isinstance(res, dict) and 'logits' in res:
                    return res
        return outputs

    def _word_accuracy(self, outputs, labels, lengths):
        res = self._pick_output(outputs)
        logits = res['logits']
        pred_ids = logits.argmax(dim=-1)
        pred_lengths = self.model._get_length(logits)

        if labels.dim() == 3:
            labels = labels.argmax(dim=-1)

        total_word = labels.size(0)
        correct_word = 0
        total_char = 0
        correct_char = 0
        for i in range(total_word):
            l = lengths[i].item()
            p_len = pred_lengths[i].item()
            l = min(l, p_len)
            gt = labels[i][:l]
            pr = pred_ids[i][:l]
            correct_char += (gt == pr).sum().item()
            total_char += l
            if torch.equal(gt, pr):
                correct_word += 1
        char_acc = correct_char / max(total_char, 1)
        word_acc = correct_word / max(total_word, 1)
        return char_acc, word_acc

    # --- steps ---
    def training_step(self, batch, batch_idx):
        images, target = batch
        if getattr(self.config, 'dataset_channels_last', False):
            images = images.to(memory_format=torch.channels_last)
        gt_labels, gt_lengths = target
        outputs = self(images)
        loss = self.criterion(outputs, gt_labels, gt_lengths)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        # メトリクスは設定間隔ごとに計算し、CPU側の後処理負荷を抑制
        if (self.global_step % self.metric_interval) == 0:
            with torch.no_grad():
                ccr, cwr = self._word_accuracy(outputs, gt_labels, gt_lengths)
            self.log("train/ccr", ccr, on_step=True, prog_bar=False)
            self.log("train/cwr", cwr, on_step=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch
        if getattr(self.config, 'dataset_channels_last', False):
            images = images.to(memory_format=torch.channels_last)
        gt_labels, gt_lengths = target
        outputs = self(images)
        loss = self.criterion(outputs, gt_labels, gt_lengths)
        ccr, cwr = self._word_accuracy(outputs, gt_labels, gt_lengths)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/ccr", ccr, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/cwr", cwr, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, target = batch
        if getattr(self.config, 'dataset_channels_last', False):
            images = images.to(memory_format=torch.channels_last)
        gt_labels, gt_lengths = target
        outputs = self(images)
        loss = self.criterion(outputs, gt_labels, gt_lengths)
        ccr, cwr = self._word_accuracy(outputs, gt_labels, gt_lengths)
        self.log("test/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/ccr", ccr, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/cwr", cwr, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def _is_schedulefree_optimizer(self, optimizer):
        """schedulefreeオプティマイザーかどうかを判定"""
        return SCHEDULEFREE_AVAILABLE and hasattr(optimizer, 'train') and hasattr(optimizer, 'eval')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """オプティマイザーステップ時にtrainモードに設定"""
        if self._is_schedulefree_optimizer(optimizer):
            optimizer.train()
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def on_train_epoch_start(self):
        """訓練エポック開始時にtrainモードに設定"""
        super().on_train_epoch_start()
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for opt in optimizers:
            if self._is_schedulefree_optimizer(opt):
                opt.train()

    def on_validation_epoch_start(self):
        """検証エポック開始時にevalモードに設定"""
        super().on_validation_epoch_start()
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for opt in optimizers:
            if self._is_schedulefree_optimizer(opt):
                opt.eval()

    # --- optimizers ---
    def configure_optimizers(self):
        optimizer_type = self.config.optimizer_type
        
        # schedulefreeのoptimizerをサポート
        if SCHEDULEFREE_AVAILABLE and optimizer_type in ['RAdamScheduleFree', 'AdamWScheduleFree']:
            if optimizer_type == 'RAdamScheduleFree':
                opt_cls = RAdamScheduleFree
            elif optimizer_type == 'AdamWScheduleFree':
                opt_cls = AdamWScheduleFree
            opt = opt_cls(self.parameters(),
                          lr=self.config.optimizer_lr,
                          **(self.config.optimizer_args or {}))
            # schedulefreeのoptimizerは学習率スケジューラが不要
            return opt
        else:
            # 標準のtorch.optimから取得
            opt_cls = getattr(torch.optim, optimizer_type)
            opt = opt_cls(self.parameters(),
                          lr=self.config.optimizer_lr,
                          **(self.config.optimizer_args or {}))

            periods = self.config.optimizer_scheduler_periods
            gamma = self.config.optimizer_scheduler_gamma
            milestones = list(np.cumsum(periods)[:-1])
            sched = lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "interval": "epoch",
                    "monitor": "val/cwr"
                }
            }
