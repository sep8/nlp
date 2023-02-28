from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torch.utils.data import DataLoader, Dataset


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s, is_cn=False):
    if (is_cn):
        s = s.strip()
        s = re.sub(r"([。！？])", r" \1", s)
    else:
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# def cn_tokenizer(sentence):
#     return [normalizeString(word, True) for word in list(sentence)]

cn_tokenizer = get_tokenizer('spacy', language='zh_core_web_sm')
en_tokenizer_ = get_tokenizer('spacy', language='en_core_web_sm')

def en_tokenizer(sentence):
    return en_tokenizer_(sentence)

def build_vocab():
    counter1 = Counter()
    counter2 = Counter()
    # Read the file and split into lines
    lines = open('data/en-cn.txt', encoding='utf-8').read().strip().split('\n')
    for l in lines:
        l1, l2 = l.split('\t')
        counter1.update(en_tokenizer(l1))
        counter2.update(cn_tokenizer(l2))

    vocab1 = vocab(counter1, specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    vocab2 = vocab(counter2, specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    vocab1.set_default_index(vocab1["<unk>"])
    vocab2.set_default_index(vocab2["<unk>"])
    return [vocab1, vocab2]


en_vocab, cn_vocab = build_vocab()

cn_vocab_size = len(cn_vocab)
en_vocab_size = len(en_vocab)



eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_sentence(l1, l2, max_length):
    l1_words = en_tokenizer(l1)
    l2_words = cn_tokenizer(l2)
    return len(l1_words) < max_length and len(l2_words) < max_length and normalizeString(l1, False).startswith(eng_prefixes)


class TranslationDataset(Dataset):
    def __init__(self, max_length, device=torch.device("cpu")):
        self.device = device
        self.data = []
        self.pairs = []
        lines = open('data/en-cn.txt',
                     encoding='utf-8').read().strip().split('\n')
        for l in lines:
            l1, l2 = l.split('\t')
            if (max_length is None or filter_sentence(l1, l2, max_length)):
                cn_tensor_ = self.tokenize_sentence(l2, True)
                en_tensor_ = self.tokenize_sentence(l1, False)
                self.data.append((cn_tensor_, en_tensor_))
                self.pairs.append((l2, l1))

        self.len = len(self.data)

    def tokenize_sentence(self, sentence, is_cn):
        vocab = cn_vocab if is_cn else en_vocab
        tokenizer = cn_tokenizer if is_cn else en_tokenizer
        indexes = [vocab[token] for token in tokenizer(sentence)]
        indexes = [vocab['<sos>']] + indexes + [vocab['<eos>']]
        return torch.tensor(indexes, dtype=torch.long, device=self.device)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def collate_fn(batch):
    source = [item[0] for item in batch] 
    #pad them using pad_sequence method from pytorch. 
    source = pad_sequence(source, batch_first=False, padding_value=cn_vocab['<pad>']) 
    
    #get all target indexed sentences of the batch
    target = [item[1] for item in batch] 
    #pad them using pad_sequence method from pytorch. 
    target = pad_sequence(target, batch_first=False, padding_value=en_vocab['<pad>'])
    return source, target


def to_sentence(indexes, is_cn):
    vacab = cn_vocab if (is_cn) else en_vocab
    return[vacab.get_itos()[index[0]] for index in indexes]

def get_cn_en_dataloader(seq_len, batch_size, device):
    train_dataset = TranslationDataset(seq_len, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return train_loader, train_dataset

def collate_padded_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    source = [item[0] for item in batch] 
    source_lengths = torch.tensor([len(item) for item in source])
    #pad them using pad_sequence method from pytorch. 
    source = pad_sequence(source, batch_first=False, padding_value=cn_vocab['<pad>'])
    
    
    #get all target indexed sentences of the batch
    target = [item[1] for item in batch] 
    #pad them using pad_sequence method from pytorch. 
    target = pad_sequence(target, batch_first=False, padding_value=en_vocab['<pad>'])
    return (source, source_lengths), target


def get_cn_en_padded_loader(max_length, batch_size, device):
    train_dataset = TranslationDataset(max_length, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_padded_fn)
    return train_loader, train_dataset

def get_cn_en_dataset(max_length, device):
    dataset = TranslationDataset(max_length, device)
    return dataset
    
