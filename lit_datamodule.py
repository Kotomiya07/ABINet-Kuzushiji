import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import ImageDataset, TextDataset, HuggingFaceTextDataset


class ABINetDataModule(pl.LightningDataModule):
    """LightningDataModule wrapping既存のデータセット実装."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        if self.config.global_stage == 'pretrain-language':
            kwargs = dict(
                max_length=self.config.dataset_max_length,
                case_sensitive=self.config.dataset_case_sensitive,
                charset_path=self.config.dataset_charset_path,
                smooth_label=self.config.dataset_smooth_label,
                smooth_factor=self.config.dataset_smooth_factor,
                one_hot_x=self.config.dataset_one_hot_y,
                one_hot_y=self.config.dataset_one_hot_y,
                use_sm=self.config.dataset_use_sm,
            )
            # Hugging Faceデータセットが指定されている場合はそれを使用
            huggingface_train = getattr(self.config, 'dataset_huggingface_train', None)
            if huggingface_train is not None and len(huggingface_train) > 0:
                text_column = getattr(self.config, 'dataset_huggingface_text_column', 'text')
                train_split = getattr(self.config, 'dataset_huggingface_train_split', 'train')
                self.train_ds = HuggingFaceTextDataset(
                    dataset_names=huggingface_train,
                    text_column=text_column,
                    split=train_split,
                    is_training=True,
                    **kwargs
                )
            else:
                self.train_ds = TextDataset(self.config.dataset_train_roots[0], is_training=True, **kwargs)
            
            # 検証用データセット
            huggingface_test = getattr(self.config, 'dataset_huggingface_test', None)
            if huggingface_test is not None and len(huggingface_test) > 0:
                text_column = getattr(self.config, 'dataset_huggingface_text_column', 'text')
                test_split = getattr(self.config, 'dataset_huggingface_test_split', 'validation')
                self.val_ds = HuggingFaceTextDataset(
                    dataset_names=huggingface_test,
                    text_column=text_column,
                    split=test_split,
                    is_training=False,
                    **kwargs
                )
            else:
                self.val_ds = TextDataset(self.config.dataset_test_roots[0], is_training=False, **kwargs)
        else:
            common_kwargs = dict(
                img_h=self.config.dataset_image_height,
                img_w=self.config.dataset_image_width,
                max_length=self.config.dataset_max_length,
                case_sensitive=self.config.dataset_case_sensitive,
                charset_path=self.config.dataset_charset_path,
                data_aug=self.config.dataset_data_aug,
                deteriorate_ratio=self.config.dataset_deteriorate_ratio,
                multiscales=self.config.dataset_multiscales,
                one_hot_y=self.config.dataset_one_hot_y,
                readahead=self.config.dataset_readahead,
            )
            train_list = [ImageDataset(p, is_training=True, **common_kwargs) for p in self.config.dataset_train_roots]
            val_list = [ImageDataset(p, is_training=False, **common_kwargs) for p in self.config.dataset_test_roots]
            if len(train_list) == 1:
                self.train_ds = train_list[0]
            else:
                from utils import MyConcatDataset
                self.train_ds = MyConcatDataset(train_list)
            if len(val_list) == 1:
                self.val_ds = val_list[0]
            else:
                from utils import MyConcatDataset
                self.val_ds = MyConcatDataset(val_list)
            # テスト用も同じリストを再利用（Lightningのtest_dataloaderで使用）
            self.test_ds = self.val_ds

    def train_dataloader(self):
        # DataLoader設定を強化してデータ供給の揺らぎを減らす
        persistent_workers = False
        prefetch_factor = None
        if self.config.dataset_num_workers and self.config.dataset_num_workers > 0:
            persistent_workers = self.config.dataset_persistent_workers
            prefetch_factor = self.config.dataset_prefetch_factor

        return DataLoader(
            self.train_ds,
            batch_size=self.config.dataset_train_batch_size,
            shuffle=True,
            num_workers=self.config.dataset_num_workers,
            pin_memory=self.config.dataset_pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=True,
        )

    def val_dataloader(self):
        persistent_workers = False
        prefetch_factor = None
        if self.config.dataset_num_workers and self.config.dataset_num_workers > 0:
            persistent_workers = self.config.dataset_persistent_workers
            prefetch_factor = self.config.dataset_prefetch_factor

        return DataLoader(
            self.val_ds,
            batch_size=self.config.dataset_test_batch_size,
            shuffle=False,
            num_workers=self.config.dataset_num_workers,
            pin_memory=self.config.dataset_pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def test_dataloader(self):
        persistent_workers = False
        prefetch_factor = None
        if self.config.dataset_num_workers and self.config.dataset_num_workers > 0:
            persistent_workers = self.config.dataset_persistent_workers
            prefetch_factor = self.config.dataset_prefetch_factor

        return DataLoader(
            self.test_ds,
            batch_size=self.config.dataset_test_batch_size,
            shuffle=False,
            num_workers=self.config.dataset_num_workers,
            pin_memory=self.config.dataset_pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
