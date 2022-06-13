import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizerFast


def get_labels_to_ids(data_dir):
    tags = []
    file_path = os.path.join(data_dir, 'tags.txt')
    with open(file_path, 'r') as file:
        for tag in file:
            tags.append(tag.strip())
    labels_to_ids = {k: v for v, k in enumerate(tags)}
    ids_to_labels = {v: k for v, k in enumerate(tags)}
    return labels_to_ids, ids_to_labels


def get_data_path(data_dir='./data', type='train'):
    if type in ['train', 'val', 'test']:
        sentences_path = os.path.join(data_dir, type, 'sentences.txt')
        tags_path = os.path.join(data_dir, type, 'tags.txt')
        return sentences_path, tags_path
    else:
        raise ValueError("data type not in ['train', 'val', 'test']")


class DataLoader(object):
    def __init__(self, data_dir, model_name, params):
        self.data_dir = data_dir
        self.max_len = params['max_len']
        self.max_lines = params['max_lines']
        self.padded_token = params['padded_token']
        labels_to_ids, ids_to_labels = get_labels_to_ids(data_dir)
        self.tag2idx = labels_to_ids
        self.idx2tag = ids_to_labels
        self.tag_pad_idx = len(self.idx2tag)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def load_data(self, type='train'):
        """Loads sentences and tags from their corresponding files. 
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        sentences_file, tags_file = get_data_path(self.data_dir, type)
        data = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "tags": [],
            "size": 0
        }

        with open(sentences_file, 'r') as file:
            lines = 0
            for line in file:
                if self.max_lines and lines >= self.max_lines:
                    break
                padiding = 'max_length' if self.padded_token else True
                # replace each token by its index
                tokens = self.tokenizer(text=line.strip(
                ), max_length=self.max_len, truncation=True, padding=padiding, add_special_tokens=True)
                data['input_ids'].append(tokens['input_ids'])
                data['token_type_ids'].append(tokens['token_type_ids'])
                data['attention_mask'].append(tokens['attention_mask'])
                lines += 1

        with open(tags_file, 'r') as file:
            lines = 0
            for line in file:
                if self.max_lines and lines >= self.max_lines:
                    break
                # replace each tag by its index
                target_tags = [self.tag2idx.get(tag)
                               for tag in line.strip().split(' ')]
                # Add token for tag [CLS] and [SEP]
                target_tags = target_tags[:self.max_len - 2]
                target_tags = [self.tag_pad_idx] + \
                    target_tags + [self.tag_pad_idx]
                # Add padded TAG tokens
                padding_len = self.max_len - len(target_tags)
                target_tags = target_tags + ([self.tag_pad_idx] * padding_len)
                data['tags'].append(target_tags)
                lines += 1

        # checks to ensure there is a tag for each token
        # assert len(data['input_ids']) == len(data['tags'])
        # for i in range(len(data['input_ids'])):
        #     assert len(data['tags'][i]) == len(data['input_ids'][i])
        data["input_ids"] = tf.constant(data["input_ids"])
        data["token_type_ids"] = tf.constant(data["token_type_ids"])
        data["attention_mask"] = tf.constant(data["attention_mask"])
        data["tags"] = tf.constant(data["tags"])
        data['size'] = len(data['input_ids'])
        return data

    def get_text_tokens(self, text):
        token = self.tokenizer.encode(text, add_special_tokens=True)
        original_len = len(token)
        tokens = self.tokenizer(text=text, max_length=self.max_len,
                                truncation=True, padding='max_length',
                                add_special_tokens=True)
        x = [tf.constant([tokens['input_ids']]), tf.constant([tokens['token_type_ids']]), tf.constant([tokens['attention_mask']])]
        return x, original_len
