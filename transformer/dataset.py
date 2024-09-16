import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, data_source, source_tokenizer, target_tokenizer, source_lang, target_lang, max_length):
        self.data = data_source
        self.src_tokenizer = source_tokenizer
        self.tgt_tokenizer = target_tokenizer
        self.src_lang = source_lang
        self.tgt_lang = target_lang
        self.max_len = max_length

        self.start_token = torch.tensor([target_tokenizer.token_to_id("[START]")], dtype=torch.int64)
        self.end_token = torch.tensor([target_tokenizer.token_to_id("[END]")], dtype=torch.int64)
        self.pad_token = torch.tensor([target_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pair = self.data[index]
        source_text = pair['translation'][self.src_lang]
        target_text = pair['translation'][self.tgt_lang]

        source_tokens = self.tokenize_and_pad(self.src_tokenizer, source_text, include_end=True)
        target_tokens = self.tokenize_and_pad(self.tgt_tokenizer, target_text, include_end=False)

        target_input = torch.cat([self.start_token, target_tokens])
        target_output = torch.cat([target_tokens, self.end_token])

        source_mask = self.create_padding_mask(source_tokens)
        target_mask = self.create_causal_mask(target_input)

        return {
            "encoder_input": source_tokens,
            "decoder_input": target_input,
            "encoder_mask": source_mask,
            "decoder_mask": target_mask,
            "label": target_output,
            "source_text": source_text,
            "target_text": target_text
        }

    def tokenize_and_pad(self, tokenizer, text, include_end=True):
        tokens = torch.tensor(tokenizer.encode(text).ids, dtype=torch.int64)
        if include_end:
            tokens = torch.cat([tokens, self.end_token])
        padding_length = self.max_len - len(tokens)
        if padding_length < 0:
            raise ValueError("Sequence exceeds maximum length")
        return torch.cat([tokens, self.pad_token.repeat(padding_length)])

    def create_padding_mask(self, sequence):
        return (sequence != self.pad_token).unsqueeze(0).unsqueeze(0).int()

    def create_causal_mask(self, sequence):
        seq_len = sequence.size(0)
        mask = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).type(torch.int)
        return (sequence != self.pad_token).unsqueeze(0).int() & (mask == 0)
