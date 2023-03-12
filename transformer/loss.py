import torch.nn as nn

class TranslationLoss(nn.Module):
    def __init__(self, pad_index, label_smoothing=0.0):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(ignore_index = pad_index, label_smoothing=label_smoothing)

    def forward(self, logits, labels):
        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1)
        return self.loss_func(logits, labels)