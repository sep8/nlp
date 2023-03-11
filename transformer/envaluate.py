import torch
from dataset import TranslationDataset, vocabs
from transformer import Transformer
from torch.utils.data import DataLoader

lang1 = 'cn'
lang2 = 'en'
max_lines = None
num_test = 10
device = torch.device('mps')

if __name__ == "__main__":
    test_dataset = TranslationDataset(lang1, lang2, 'test', None, device)
    lang1_vocab = vocabs[lang1]
    lang2_vocab = vocabs[lang2]
    lang1_vocab_size = len(lang1_vocab)
    lang2_vocab_size = len(lang2_vocab)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    d_model = 256  # Embedding Size
    d_ff = 1024 # FeedForward dimension
    d_k = d_v = 32  # dimension of K(=Q), V
    n_layers = 3  # number of Encoder of Decoder Layer
    n_heads = 4  # number of heads in Multi-Head Attention

    model = Transformer(d_model, n_layers, len(lang1_vocab), len(lang2_vocab), d_k, d_v, n_heads, d_ff, device).to(device)
    model.load_state_dict(torch.load('models/transformer-best.pt'))

    for i, (enc_inputs, target) in enumerate(test_loader):
        if i >= num_test:
            break
        enc_input = enc_inputs[0].view(1, -1)
        sos_token = torch.tensor([[lang1_vocab['<sos>']]], device=device)
        enc_input = torch.cat((sos_token, enc_input), dim=1)
        predict = model.interface(enc_input, start_symbol=lang2_vocab['<sos>'], tgt_eos=lang2_vocab['<eos>'])
        print(''.join([lang1_vocab.get_itos()[n.item()] for n in enc_input.squeeze()]))
        print('>', ' '.join([lang2_vocab.get_itos()[n.item()] for n in predict.squeeze()]))
        print('=', ' '.join([lang2_vocab.get_itos()[n.item()] for n in target[0]]))

    
