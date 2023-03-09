import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torch.utils.data import Dataset
from torchtext.vocab import vocab


spacy_en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
spacy_de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
def en_tokenizer(sentence):
    return [token.lower() for token in spacy_en_tokenizer(sentence)]
def de_tokenizer(sentence):
    return [token for token in spacy_de_tokenizer(sentence)]
lang_tokenizers = {'en': en_tokenizer, 'de': de_tokenizer }

class TranslationDataset(Dataset):
    def __init__(self, lang1, lang2, max_lines, device, val=False):
        super(TranslationDataset, self).__init__()
        self.device = device
        self.max_lines = max_lines
        self.lang1 = lang1
        self.lang2 = lang2
        self.file_path = 'data/%s-%s.txt' % (lang1, lang2) if not val else 'data/%s-%s.val.txt' % (lang1, lang2)

        lang1_vocab, lang2_vocab = self.build_vocab_data(lang1, lang2)
        self.lang1_vocab = lang1_vocab
        self.lang2_vocab = lang2_vocab
        self.lang1_vocab_size = len(lang1_vocab)
        self.lang2_vocab_size = len(lang2_vocab)

        data, untokenized_data = self.build_data(lang1, lang2)
        self.data = data
        self.untokenized_data = untokenized_data
        self.len = len(self.data)

    def tokenize_sentence(self, sentence, is_lang1):
        vocab = self.lang1_vocab if is_lang1 else self.lang2_vocab
        tokenizer = lang_tokenizers[self.lang1] if is_lang1 else lang_tokenizers[self.lang2]
        indexes = [vocab[token] for token in tokenizer(sentence)]
        # indexes = [vocab['<sos>']] + indexes + [vocab['<eos>']]
        return torch.tensor(indexes, dtype=torch.long, device=self.device)

    def build_data(self, lang1, lang2):
        # Read the file and split into lines
        lines = open(self.file_path, encoding='utf-8').read().strip().split('\n')
        data = []
        untokenized_data = []
        num_lines = 0
        for l in lines:
            l1, l2 = l.split('\t')
            if (self.max_lines is None or self.max_lines > num_lines):
                l1_tokens = self.tokenize_sentence(l1, True)
                l2_tokens = self.tokenize_sentence(l2, False)
                data.append((l1_tokens, l2_tokens))
                untokenized_data.append((l1, l2))
                num_lines += 1
                
        return data, untokenized_data 
    
    def build_vocab_data(self, lang1, lang2):
        counter1 = Counter()
        counter2 = Counter()
        lang1_tokenizer = lang_tokenizers[lang1]
        lang2_tokenizer = lang_tokenizers[lang2]
        # Read the file and split into lines
        lines = open(self.file_path, encoding='utf-8').read().strip().split('\n')
        for l in lines:
            l1, l2 = l.split('\t')
            counter1.update(lang1_tokenizer(l1))
            counter2.update(lang2_tokenizer(l2))

        vocab1 = vocab(counter1, min_freq=2, specials=['<unk>', '<pad>', '<sos>', '<eos>'])
        vocab2 = vocab(counter2, min_freq=2, specials=['<unk>', '<pad>', '<sos>', '<eos>'])
        vocab1.set_default_index(vocab1["<unk>"])
        vocab2.set_default_index(vocab2["<unk>"])

        return vocab1, vocab2

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
