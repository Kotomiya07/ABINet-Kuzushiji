import pytorch_lightning as pl
import torch
import jiwer

from losses import MultiLosses
from utils import ifnone

# schedulefreeのサポート（オプショナル）
try:
    from schedulefree import RAdamScheduleFree, AdamWScheduleFree

    SCHEDULEFREE_AVAILABLE = True
except ImportError:
    SCHEDULEFREE_AVAILABLE = False


class LanguageLightningModule(pl.LightningModule):
    """Language-only pretrain LightningModule (BCNLanguage)."""

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self.criterion = MultiLosses(one_hot=config.dataset_one_hot_y)
        # Top-K精度のK値（デフォルトは5）
        self.top_k = ifnone(getattr(config, "model_k", None), 5)
        # Config は独自クラスで deepcopy に失敗するため除外
        self.save_hyperparameters(ignore=["model", "config"])

    def forward(self, tokens, lengths):
        return self.model(tokens, lengths)

    def _character_accuracy(self, outputs, labels, lengths):
        """文字精度とCERを計算

        Returns:
            top_k_acc: Top-K文字精度
            cer: Character Error Rate
        """
        logits = outputs["logits"]  # (N, T, C)
        pred_ids = logits.argmax(dim=-1)  # (N, T)
        pred_lengths = self.model._get_length(logits)  # (N,)

        if labels.dim() == 3:
            labels = labels.argmax(dim=-1)  # one-hotから通常のラベルに変換

        batch_size = labels.size(0)
        total_chars = 0
        top_k_correct = 0

        # Top-K精度の計算用
        top_k_values, top_k_indices = logits.topk(self.top_k, dim=-1)  # (N, T, K)

        # CER計算用の文字列リスト
        gt_texts = []
        pred_texts = []

        for i in range(batch_size):
            gt_length = lengths[i].item()
            pred_length = pred_lengths[i].item()

            # 実際の長さに合わせて切り取り
            gt_labels_seq = labels[i][:gt_length].cpu()
            pred_labels_seq = pred_ids[i][:pred_length].cpu()

            # ラベルIDを文字列に変換
            gt_text = self.model.charset.get_text(gt_labels_seq, padding=False, trim=True)
            pred_text = self.model.charset.get_text(pred_labels_seq, padding=False, trim=True)

            gt_texts.append(gt_text)
            pred_texts.append(pred_text)

            # Top-K文字精度の計算
            # 正解の長さ分だけ評価
            for j in range(gt_length):
                if j < pred_length:
                    gt_char = labels[i][j].item()
                    top_k_chars = top_k_indices[i][j].cpu().numpy().tolist()
                    if gt_char in top_k_chars:
                        top_k_correct += 1
                total_chars += 1

        top_k_acc = top_k_correct / max(total_chars, 1)

        # jiwerでCERを計算
        cer = jiwer.cer(gt_texts, pred_texts)

        return top_k_acc, cer

    def training_step(self, batch, batch_idx):
        tokens, target = batch  # tokens: [label_x, length_x], target: [label_y, length_y]
        tokens_x, lengths_x = tokens
        gt_labels, gt_lengths = target
        outputs = self(tokens_x, lengths_x)
        loss = self.criterion(outputs, gt_labels, gt_lengths)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        # 精度評価（勾配計算を無効化）
        with torch.no_grad():
            top_k_acc, cer = self._character_accuracy(outputs, gt_labels, gt_lengths)
        self.log(f"train/top{self.top_k}_acc", top_k_acc, on_step=True, prog_bar=False)
        self.log("train/cer", cer, on_step=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        tokens, target = batch
        tokens_x, lengths_x = tokens
        gt_labels, gt_lengths = target
        outputs = self(tokens_x, lengths_x)
        loss = self.criterion(outputs, gt_labels, gt_lengths)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # 精度評価
        with torch.no_grad():
            top_k_acc, cer = self._character_accuracy(outputs, gt_labels, gt_lengths)
        self.log(f"val/top{self.top_k}_acc", top_k_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/cer", cer, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        tokens, target = batch
        tokens_x, lengths_x = tokens
        gt_labels, gt_lengths = target
        outputs = self(tokens_x, lengths_x)
        loss = self.criterion(outputs, gt_labels, gt_lengths)
        self.log("test/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # 精度評価
        with torch.no_grad():
            top_k_acc, cer = self._character_accuracy(outputs, gt_labels, gt_lengths)
        self.log(f"test/top{self.top_k}_acc", top_k_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/cer", cer, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def _is_schedulefree_optimizer(self, optimizer):
        """schedulefreeオプティマイザーかどうかを判定"""
        return SCHEDULEFREE_AVAILABLE and hasattr(optimizer, "train") and hasattr(optimizer, "eval")

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
        if SCHEDULEFREE_AVAILABLE and optimizer_type in ["RAdamScheduleFree", "AdamWScheduleFree"]:
            if optimizer_type == "RAdamScheduleFree":
                opt_cls = RAdamScheduleFree
            elif optimizer_type == "AdamWScheduleFree":
                opt_cls = AdamWScheduleFree
            opt = opt_cls(self.parameters(), lr=self.config.optimizer_lr, **(self.config.optimizer_args or {}))
            # schedulefreeのoptimizerは学習率スケジューラが不要
            return opt
        else:
            # 標準のtorch.optimから取得
            opt_cls = getattr(torch.optim, optimizer_type)
            opt = opt_cls(self.parameters(), lr=self.config.optimizer_lr, **(self.config.optimizer_args or {}))

            periods = self.config.optimizer_scheduler_periods
            gamma = self.config.optimizer_scheduler_gamma
            milestones = list(torch.cumsum(torch.tensor(periods), dim=0)[:-1].tolist())
            sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch", "monitor": "val/loss"}}
