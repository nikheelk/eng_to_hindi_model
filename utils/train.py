import torch
import torch.nn as nn

import time
import math
import os

from .test import evaluate
from .optimizer import get_optimizer
from .loss import get_criterion
from models import *

best_valid_loss = float('inf')
start_epoch = 0

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def model_train(model_name, model_type, device, SRC, TRG, resume, train_iterator, valid_iterator, training = True, n_epochs = 10, ENC_EMB_DIM = 256, DEC_EMB_DIM = 256, HID_DIM = 512, ENC_HID_DIM = 512, DEC_HID_DIM = 512, N_LAYERS = 2, ENC_DROPOUT = 0.5, DEC_DROPOUT = 0.5, CLIP = 1):
    global start_epoch, best_valid_loss

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    if model_type == "seq2seqV1":
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    elif model_type == "seq2seqV2":
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)
    elif model_type == "attention":
        attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, device).to(device)

    # def init_weights(m):


    def init_weights(m):
        if model_type == "seq2seqV1":
            for name, param in m.named_parameters():
                nn.init.uniform_(param.data, -0.08, 0.08)
        if model_type == "seq2seqV2":
            for name, param in m.named_parameters():
                nn.init.normal_(param.data, mean=0, std=0.01)
        elif model_type == "attention":
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0, std=0.01)
                else:
                    nn.init.constant_(param.data, 0)

    model.apply(init_weights)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n****************************************************************************\n")
    print("*****Model Details*****\n")
    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("\n****************************************************************************\n")
    
    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+ model_name +'.pth')
        model.load_state_dict(checkpoint['net'])
        print('==> Model loaded from checkpoint..')
        best_valid_loss = checkpoint['loss']
        print("best_valid_loss", best_valid_loss)
        start_epoch = checkpoint['epoch']
        print("start_epoch", start_epoch)

    optimizer = get_optimizer(model)

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    criterion = get_criterion(TRG_PAD_IDX)
    if training:
        for epoch in range(start_epoch + 1, start_epoch + n_epochs + 1):
            print(epoch)
            start_time = time.time()
            
            train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
            valid_loss = evaluate(model, valid_iterator, criterion)
            
            end_time = time.time()
            
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            if valid_loss < best_valid_loss:
                print("\n*****Saving Model*****")
                state = {
                    'net': model.state_dict(),
                    'loss': valid_loss,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/'+ model_name +'.pth')
                best_valid_loss = valid_loss
            
            print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        
    return model
