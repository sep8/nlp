from dataset import TranslationDataset
from transformer import Transformer
import torch
from torch.utils.data import DataLoader

lang1 = 'en'
lang2 = 'de'
max_lines = None
num_test = 10
device = torch.device('mps')

if __name__ == "__main__":
    lang1_vocab = torch.load('data/vocab_%s.pth' % lang1)
    lang2_vocab = torch.load('data/vocab_%s.pth' % lang2)
    dataset = TranslationDataset(lang1, lang2, max_lines, device, val=True)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    d_model = 256  # Embedding Size
    d_ff = 1024 # FeedForward dimension
    d_k = d_v = 32  # dimension of K(=Q), V
    n_layers = 3  # number of Encoder of Decoder Layer
    n_heads = 4  # number of heads in Multi-Head Attention

    model = Transformer(d_model, n_layers, len(lang1_vocab), len(lang2_vocab), d_k, d_v, n_heads, d_ff, device).to(device)
    
    for i in range(num_test):
        enc_inputs, _ = next(iter(test_loader))
        enc_input = enc_inputs[0].view(1, -1)
        predict = model.interface(enc_input, start_symbol=lang2_vocab['sos'], tgt_eos=lang2_vocab['eos'])
        print(enc_input[0], '->', [lang2_vocab.get_itos()[n.item()] for n in predict.squeeze()])
    
