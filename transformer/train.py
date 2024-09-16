import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
import torchmetrics
from tqdm import tqdm
import os
from pathlib import Path
import warnings

from model import create_transformer
from dataset import TranslationDataset
from config import TranslationConfigManager

def beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, beam_size=3):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.full((1, 1), sos_idx, dtype=torch.long, device=device)
    beam = [(decoder_input, 0.0)]

    for _ in range(max_len):
        candidates = []
        for seq, score in beam:
            if seq[0, -1].item() == eos_idx:
                candidates.append((seq, score))
                continue

            decoder_mask = torch.ones((1, seq.size(1), seq.size(1)), dtype=torch.bool, device=device)
            decoder_mask[:, :, :] = torch.triu(torch.ones((seq.size(1), seq.size(1)), dtype=torch.bool, device=device), diagonal=1)

            out = model.decode(encoder_output, source_mask, seq, decoder_mask)
            prob = model.project(out[:, -1])
            top_prob, top_idx = torch.topk(prob, beam_size)

            for i in range(beam_size):
                new_seq = torch.cat([seq, top_idx[:, i].unsqueeze(0)], dim=1)
                new_score = score - top_prob[0, i].item()
                candidates.append((new_seq, new_score))

        beam = sorted(candidates, key=lambda x: x[1])[:beam_size]
        if all(seq[0, -1].item() == eos_idx for seq, _ in beam):
            break

    return beam[0][0].squeeze(0)

def evaluate_model(model, val_dataloader, tokenizer_src, tokenizer_tgt, max_len, device, writer, global_step, num_examples=2):
    model.eval()
    metric_cer = torchmetrics.CharErrorRate()
    metric_wer = torchmetrics.WordErrorRate()
    metric_bleu = torchmetrics.BLEUScore()

    with torch.no_grad():
        for idx, batch in enumerate(val_dataloader):
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            model_out = beam_search_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["source_text"][0]
            target_text = batch["target_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            metric_cer.update([model_out_text], [target_text])
            metric_wer.update([model_out_text], [target_text])
            metric_bleu.update([model_out_text], [[target_text]])

            if idx < num_examples:
                print(f"Source: {source_text}")
                print(f"Target: {target_text}")
                print(f"Predicted: {model_out_text}")
                print()

    cer = metric_cer.compute()
    wer = metric_wer.compute()
    bleu = metric_bleu.compute()

    writer.add_scalar('validation/cer', cer, global_step)
    writer.add_scalar('validation/wer', wer, global_step)
    writer.add_scalar('validation/bleu', bleu, global_step)
    writer.flush()

    model.train()

def create_tokenizer(config, ds, lang):
    tokenizer_path = Path(config.settings['tokenizer_template'].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator((item['translation'][lang] for item in ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def prepare_data(config):
    ds_raw = load_dataset(config.settings['data_corpus'], f"{config.settings['source_language']}-{config.settings['target_language']}", split='train')

    tokenizer_src = create_tokenizer(config, ds_raw, config.settings['source_language'])
    tokenizer_tgt = create_tokenizer(config, ds_raw, config.settings['target_language'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = TranslationDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config.settings['source_language'], config.settings['target_language'], config.settings['sequence_length'])
    val_ds = TranslationDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config.settings['source_language'], config.settings['target_language'], config.settings['sequence_length'])

    train_dataloader = DataLoader(train_ds, batch_size=config.settings['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def train_transformer(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config.settings['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = prepare_data(config)
    model = create_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config.settings['sequence_length'],
        config.settings['embedding_dim']
    ).to(device)

    writer = SummaryWriter(config.settings['log_directory'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config.settings['learning_rate'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config.settings['load_checkpoint'] == 'most_recent':
        checkpoint_path = config.find_most_recent_checkpoint()
        if checkpoint_path:
            print(f'Loading checkpoint: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initial_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config.settings['training_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        evaluate_model(model, val_dataloader, tokenizer_src, tokenizer_tgt, config.settings['sequence_length'], device, writer, global_step)

        checkpoint_path = config.checkpoint_path(f"epoch_{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, checkpoint_path)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config_manager = TranslationConfigManager()
    train_transformer(config_manager)
