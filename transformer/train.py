import argparse
import torch
from torch.utils.data import DataLoader
from transformer import Transformer
from dataset_local import vocab_transform, data_ites, collate_fn, SRC_LANGUAGE, TGT_LANGUAGE, PAD_IDX
# from dataset_multi30k import vocab_transform, data_ites, collate_fn, SRC_LANGUAGE, TGT_LANGUAGE, PAD_IDX
from scheduler import Scheduler
from loss import TranslationLoss
from utils import epoch_time, count_parameters
import time
import random

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
seed = 0
batch_size=32
d_model = 256  # Embedding Size
d_ff = 512 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 3  # number of Encoder of Decoder Layer
n_heads = 4  # number of heads in Multi-Head Attention

lr = 0.0001
warmup_steps = 10000


def train(model, iterator, optimizer, criterion, scheduler, device):
    model.train()
    epoch_loss = 0
    for src, tgt in iterator:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]

        optimizer.zero_grad()
         # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(src, tgt_input)

        tgt_out = tgt[:, 1:]
        loss = criterion(outputs, tgt_out)
        loss.backward()
        optimizer.step()

        # learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()
    return epoch_loss / len(list(iterator))

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in iterator:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]

            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(src, tgt_input)

            tgt_out = tgt[:, 1:]
            loss = criterion(outputs, tgt_out)
            epoch_loss += loss.item()
    return epoch_loss / len(list(iterator))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default='./models/', type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--num_epochs", default=20, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--device', choices=['cpu', 'gpu', 'mps'], default="mps", help="Select which device to train model, defaults to gpu.")
    args = parser.parse_args()

    save_dir = args.save_dir
    device = torch.device(args.device)
    num_epochs = args.num_epochs

    print('num_epochs', num_epochs)
    print('batch_size', batch_size)

    torch.manual_seed(seed)
    random.seed(seed)

    train_iter = data_ites['train']
    train_loader = DataLoader(train_iter, batch_size=batch_size,shuffle=True, collate_fn=collate_fn)
    val_iter = data_ites['val']
    val_loader = DataLoader(val_iter, batch_size=batch_size,shuffle=False, collate_fn=collate_fn)
    print('length of train_loader', len(list(train_loader)))
    print('length of val_loader', len(list(val_loader)))


    model = Transformer(d_model, n_layers, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, d_k, d_v, n_heads, d_ff, device).to(device)
    criterion = TranslationLoss(PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    scheduler = Scheduler(optimizer, d_model, warmup_steps)

    count_parameters(model)

    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, scheduler, device)
        valid_loss = evaluate(model, val_loader, criterion, device)

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




 