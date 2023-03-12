from typing import List
import torch
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

SRC_LANGUAGE = 'cn'
TGT_LANGUAGE = 'en'

token_transform = {}
vocab_transform = {}
data_ites = {}

spacy_cn_tokenizer = get_tokenizer('spacy', language='zh_core_web_md')
spacy_en_tokenizer = get_tokenizer('spacy', language='en_core_web_md')
def cn_tokenizer(sentence):
    return [token for token in spacy_cn_tokenizer(sentence)]
def en_tokenizer(sentence):
    return [token.lower() for token in spacy_en_tokenizer(sentence)]

token_transform[SRC_LANGUAGE] = cn_tokenizer
token_transform[TGT_LANGUAGE] = en_tokenizer

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
cn_vocab = torch.load('vocab/cn_vocab.pt')
en_vocab = torch.load('vocab/en_vocab.pt')
vocab_transform = {'cn': cn_vocab, 'en': en_vocab}

class TranslationDataset(Dataset):
    def __init__(self, lang1, lang2, type, max_lines):
        super(TranslationDataset, self).__init__()
        self.max_lines = max_lines
        self.lang1 = lang1
        self.lang2 = lang2
        self.data_path = 'data/%s-%s.%s.txt' % (lang1, lang2, type)

        self.lang1_vocab = vocab_transform[lang1]
        self.lang2_vocab = vocab_transform[lang2]
        self.lang1_vocab_size = len(self.lang1_vocab)
        self.lang2_vocab_size = len(self.lang2_vocab)

        data = self.build_data()
        self.data = data
        self.len = len(self.data)

    def build_data(self):
        # Read the file and split into lines
        lines = open(self.data_path, encoding='utf-8').read().strip().split('\n')
        data = []
        num_lines = 0
        for l in lines:
            l1, l2 = l.split('\t')
            if (self.max_lines is None or self.max_lines > num_lines):
                data.append((l1, l2))
                num_lines += 1
                
        return data 
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
    
data_ites['train'] = TranslationDataset(SRC_LANGUAGE, TGT_LANGUAGE, 'train', None)
data_ites['val'] = TranslationDataset(SRC_LANGUAGE, TGT_LANGUAGE, 'val', None)

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True,  padding_value=PAD_IDX)
    return src_batch, tgt_batch
