import torch
from transformer import Transformer
# from dataset_multi30k import vocab_transform, data_ites, collate_fn, SRC_LANGUAGE, TGT_LANGUAGE, PAD_IDX, BOS_IDX, EOS_IDX
from dataset_local import data_ites, vocab_transform, text_transform, SRC_LANGUAGE, TGT_LANGUAGE, PAD_IDX, BOS_IDX, EOS_IDX


SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
seed = 0
batch_size=32
d_model = 256  # Embedding Size
d_ff = 512 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 3  # number of Encoder of Decoder Layer
n_heads = 4  # number of heads in Multi-Head Attention
warmup_steps = 10000
device = torch.device('mps')

def envaluate(model, input):
    input_tensor = text_transform[SRC_LANGUAGE](input.rstrip("\n")).view(1, -1)
    input_tensor = input_tensor.to(device)
    predict = model.interface(input_tensor, start_symbol=BOS_IDX, tgt_eos=EOS_IDX)
    translation = (' ').join(vocab_transform[TGT_LANGUAGE].lookup_tokens([n.item() for n in predict.squeeze()]))
    return translation
    
        
num_test = 10
if __name__ == "__main__":
    model = Transformer(d_model, n_layers, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, d_k, d_v, n_heads, d_ff, device).to(device)
    model.load_state_dict(torch.load('models/transformer-best.pt'))
    val_dataset = data_ites['train']
    for i, (src, tgt) in enumerate(val_dataset):
        if i >= num_test:
            break
        translation = envaluate(model, src)
        print(src)
        print('>',translation)
        print('=', tgt)
    
