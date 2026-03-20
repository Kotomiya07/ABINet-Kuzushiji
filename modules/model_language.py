import logging
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from utils import ifnone

from modules.model import _default_tfmer_cfg
from modules.model import Model
from modules.transformer import (PositionalEncoding, 
                                 TransformerDecoder,
                                 TransformerDecoderLayer)


class BCNLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            # 文字集合サイズが異なる場合、分類ヘッド（proj/cls）は破棄して本体のみ読み込む
            state = torch.load(config.model_language_checkpoint, map_location=None, weights_only=False)
            if "model" in state:
                model_state = state["model"]
            elif "state_dict" in state:
                model_state = {}
                for key, value in state["state_dict"].items():
                    if key.startswith("model._orig_mod."):
                        new_key = key[len("model._orig_mod."):]
                    elif key.startswith("model."):
                        new_key = key[len("model."):]
                    else:
                        new_key = key
                    model_state[new_key] = value
            else:
                model_state = state
            drop_prefixes = ("proj.", "cls.")
            removed = [k for k in list(model_state.keys()) if k.startswith(drop_prefixes)]
            for k in removed:
                model_state.pop(k)
            missing, unexpected = self.load_state_dict(model_state, strict=False)
            if removed:
                logging.info(f"Drop head params due to vocab mismatch: {removed}")
            if missing:
                logging.debug(f"Missing keys (expected for dropped head): {missing}")
            if unexpected:
                logging.debug(f"Unexpected keys: {unexpected}")

    def _build_memory_flex_attention(self, lengths, device):
        valid_lengths = lengths.to(device=device, dtype=torch.int32)
        batch_size = int(valid_lengths.shape[0])

        def score_mod(score, batch, head, q_idx, k_idx):
            valid_k = k_idx < valid_lengths[batch]
            not_self = q_idx != k_idx
            keep = valid_k & not_self
            return torch.where(keep, score, score.new_full((), float("-inf")))

        def padding_mask_mod(batch, head, q_idx, k_idx):
            return k_idx < valid_lengths[batch]

        block_mask = create_block_mask(
            padding_mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=self.max_length,
            KV_LEN=self.max_length,
            device=device,
            _compile=False,
        )
        return block_mask, score_mod

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)
        memory_flex_block_mask = None
        memory_flex_score_mod = None
        if tokens.is_cuda:
            memory_flex_block_mask, memory_flex_score_mod = self._build_memory_flex_attention(lengths, tokens.device)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)
        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask,
                memory_flex_block_mask=memory_flex_block_mask,
                memory_flex_score_mod=memory_flex_score_mod)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res
