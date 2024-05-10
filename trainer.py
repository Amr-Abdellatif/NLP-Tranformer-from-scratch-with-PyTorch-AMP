import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from dataset import BilingualDataset, casual_mask
from model_architecture import build_transformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from config import *
from tqdm.autonotebook import tqdm
import warnings
import torch.cuda.amp as amp
import torchmetrics
import os
from torch.optim.lr_scheduler import LambdaLR



def run_validation(model, validation_ds,
                    tokenizer_src, tokenizer_tgt,
                    max_len, device, print_msg, global_step, writer,
                    num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    # size of control window
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80
    
    with torch.inference_mode():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        print(f"Character error rate : {cer.item()}")
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        print(f"Word error rate : {wer.item()}")
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        print(f"Bleu score : {bleu.item()}")
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def greedy_decode(model, source, source_mask,
                tokenizer_src, tokenizer_tgt, max_len, device):
    
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    with amp.autocast():
        # Precompute the encoder output and reuse it for every step
        encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        with amp.autocast():
            # calculate output
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset(config['data_set_name_from_HF'],f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')

    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])
    
    # Subset the data (if data_subset_ratio is provided)
    """
    None -> full dataset
    [0,1] subset of the dataset
    """
    if config.get('data_subset_ratio') is not None and 0 <= config['data_subset_ratio'] <= 1:
        subset_ratio = config['data_subset_ratio']
        indices = torch.randperm(len(ds_raw))[:int(subset_ratio * len(ds_raw))]
        ds_raw = torch.utils.data.Subset(ds_raw, indices=indices.tolist())


    # data split 90 % train 
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw , val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src,
                                tokenizer_tgt, config['lang_src'],
                                config['lang_tgt'], config['seq_len'])
    
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src,
                                tokenizer_tgt, config['lang_src'],
                                config['lang_tgt'], config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))
        
    print(f"max length of source sentence : {max_len_src}")
    print(f"max length of target sentence : {max_len_tgt}")
    
    

    train_dataloader = DataLoader(train_ds, batch_size=config['Batch_size'],shuffle=True,num_workers=config['num_workers'],pin_memory=config['pin_memory'])
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True,num_workers=config['num_workers'],pin_memory=config['pin_memory'])

    return train_dataloader , val_dataloader , tokenizer_src , tokenizer_tgt

def get_model(config, vocab_src_length, vocab_tgt_len):
    model = build_transformer(vocab_src_length,vocab_tgt_len,config['seq_len'],config['seq_len'])
    return model



### Training
def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device : {device}")
    print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

    Path(f"{config['data_set_name_from_HF']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # model = torch.compile(model,mode="reduce-overhead") # works with python lower than 3.12
    
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'], eps=1e-9,fused=True)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    scaler = amp.GradScaler()  # Initialize GradScaler

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # training loop 
    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader,desc=f'Processing batch {epoch:02d}')
        
        for batch in batch_iterator:
            
            model.train()

            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1 ,1 ,seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1 , seq_len, seq_len)


            with amp.autocast():  # Use autocast for mixed precision training
                encoder_output = model.encode(encoder_input,encoder_mask) # (B, seq_len, d_model)
                decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask) # (B,seq_len, d_model)
                proj_output = model.project(decoder_output) # (B, seq_len, tgt_vocab_size)

                label = batch['label'].to(device) # (B, seq_len)

                # (B, seq_len, tgt_vocab_size) --> (B * seq_len, tgt_vocab_size)
                loss = loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()), label.view(-1))
                batch_iterator.set_postfix({'loss': f"{loss.item():6.3f}"})


                # log the loss
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.flush()

            # backward propagation
            scaler.scale(loss).backward()  # Scale the loss and perform backward pass

            # update the weights
            scaler.step(optimizer)  # Perform optimizer step with scaled gradients
            scaler.update()  # Update the scale for next iteration
            
            # zero out the gradients
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        run_validation(model, val_dataloader,
                        tokenizer_src, tokenizer_tgt,
                        config['seq_len'], device,
                        lambda msg: batch_iterator.write(msg), global_step, writer)

        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            },model_filename # save the name of the model saved states
        )

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)