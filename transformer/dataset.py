import torch
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset


spacy_cn_tokenizer = get_tokenizer('spacy', language='zh_core_web_md')
spacy_en_tokenizer = get_tokenizer('spacy', language='en_core_web_md')
def cn_tokenizer(sentence):
    return [token for token in spacy_cn_tokenizer(sentence)]
def en_tokenizer(sentence):
    return [token.lower() for token in spacy_en_tokenizer(sentence)]
tokenizers = {'cn': cn_tokenizer, 'en': en_tokenizer }
cn_vocab = torch.load('vocab/cn_vocab.pt')
en_vocab = torch.load('vocab/en_vocab.pt')
vocabs = {'cn': cn_vocab, 'en': en_vocab}

class TranslationDataset(Dataset):
    def __init__(self, lang1, lang2, type, max_lines, device):
        super(TranslationDataset, self).__init__()
        self.device = device
        self.max_lines = max_lines
        self.lang1 = lang1
        self.lang2 = lang2
        self.data_path = 'data/%s-%s.%s.txt' % (lang1, lang2, type)

        self.lang1_vocab = vocabs[lang1]
        self.lang2_vocab = vocabs[lang2]
        self.lang1_vocab_size = len(self.lang1_vocab)
        self.lang2_vocab_size = len(self.lang2_vocab)

        data, untokenized_data = self.build_data()
        self.data = data
        self.untokenized_data = untokenized_data
        self.len = len(self.data)

    def tokenize_sentence(self, sentence, is_lang1):
        vocab = self.lang1_vocab if is_lang1 else self.lang2_vocab
        tokenizer = tokenizers[self.lang1] if is_lang1 else tokenizers[self.lang2]
        indexes = [vocab[token] for token in tokenizer(sentence)]
        # indexes = [vocab['<sos>']] + indexes + [vocab['<eos>']]
        return torch.tensor(indexes, dtype=torch.long, device=self.device)

    def build_data(self):
        # Read the file and split into lines
        lines = open(self.data_path, encoding='utf-8').read().strip().split('\n')
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
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
