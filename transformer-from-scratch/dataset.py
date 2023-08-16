from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(
        self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len
    ) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id("[SOS]")]).long()
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id("[EOS]")]).long()
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id("[PAD]")]).long()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sequence length too small")

        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(enc_input_tokens).long(),
                self.eos_token,
                torch.Tensor([self.pad_token] * enc_num_padding_tokens).long(),
            ]
        )

        # Add SOS to the decodder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(dec_input_tokens).long(),
                torch.Tensor([self.pad_token] * dec_num_padding_tokens).long(),
            ]
        )

        # Add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.Tensor(dec_input_tokens).long(),
                self.eos_token,
                torch.Tensor([self.pad_token] * dec_num_padding_tokens).long(),
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (Seq_len)
            "decoder_input": decoder_input,  # (Seq_len)
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, Seq_len)
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & causal_mask(
                decoder_input.size(0)
            ),  # (1, Seq_len) & (1, Seq_len, Seq_len)
            "label": label,  # (Seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.int)
    return mask == 0
