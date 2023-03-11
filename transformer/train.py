import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from dataset import TranslationDataset, vocabs
from transformer import Transformer
import time
import random

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {params:,} trainable parameters')

lang1 = 'cn'
lang2 = 'en'
max_lines = None

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion, lang2_vocab):
    model.train()
    epoch_loss = 0
    for i, (enc_inputs, target) in enumerate(iterator):
        dec_inputs = torch.cat([torch.ones_like(target[:, :1]).fill_(lang2_vocab['<sos>']), target], dim=1)
        dec_outputs = torch.cat([target, torch.ones_like(target[:, :1]).fill_(lang2_vocab['<eos>'])], dim=1)

        optimizer.zero_grad()
         # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, lang2_vocab):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (enc_inputs, target) in enumerate(iterator):
            dec_inputs = torch.cat([torch.ones_like(target[:, :1]).fill_(lang2_vocab['<sos>']), target], dim=1)
            dec_outputs = torch.cat([target, torch.ones_like(target[:, :1]).fill_(lang2_vocab['<eos>'])], dim=1)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)



if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--save_dir", default='./models/', type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--num_epochs", default=15, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=41, type=int, help="Random seed for initialization")
    parser.add_argument('--device', choices=['cpu', 'gpu', 'mps'], default="mps", help="Select which device to train model, defaults to gpu.")

    args = parser.parse_args()

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    save_dir = args.save_dir
    device = torch.device(args.device)
    num_epochs = args.num_epochs
    seed = args.seed

    print('num_epochs', num_epochs)
    print('batch_size', batch_size)

    torch.manual_seed(seed)
    random.seed(seed)

    train_dataset = TranslationDataset(lang1, lang2, 'train', max_lines, device)
    val_dataset = TranslationDataset(lang1, lang2, 'val', None, device)
    lang1_vocab = vocabs[lang1]
    lang2_vocab = vocabs[lang2]
    lang1_vocab_size = len(lang1_vocab)
    lang2_vocab_size = len(lang2_vocab)
    

    def collate_fn(batch):
        source = [item[0] for item in batch] 
        #pad them using pad_sequence method from pytorch. 
        source = pad_sequence(source, batch_first=True, padding_value=lang1_vocab['<pad>']) 
        
        #get all target indexed sentences of the batch
        target = [item[1] for item in batch] 
        #pad them using pad_sequence method from pytorch. 
        target = pad_sequence(target, batch_first=True, padding_value=lang2_vocab['<pad>'])
        return source.to(device), target.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print('length of train_loader', len(train_loader))
    print('length of val_loader', len(val_loader))

    d_model = 256  # Embedding Size
    d_ff = 1024 # FeedForward dimension
    d_k = d_v = 32  # dimension of K(=Q), V
    n_layers = 3  # number of Encoder of Decoder Layer
    n_heads = 4  # number of heads in Multi-Head Attention

    model = Transformer(d_model, n_layers, lang1_vocab_size, lang2_vocab_size, d_k, d_v, n_heads, d_ff, device).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=lang2_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), betas = (0.9, 0.98), eps = learning_rate)

    count_parameters(model)

    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, lang2_vocab)
        valid_loss = evaluate(model, val_loader, criterion, lang2_vocab)

        if(epoch == num_epochs - 1):
            torch.save(model.state_dict(), "{}/transformer-final.pt".format(save_dir))
            
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "{}/transformer-best.pt".format(save_dir))

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')




 