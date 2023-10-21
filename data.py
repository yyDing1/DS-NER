import logging
import os
import random
import pandas as pd
import csv
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, words, label_spans, bio_labels, real_label_spans):
        self.guid = guid
        self.words = words
        self.label_spans = label_spans
        self.conf_mat = None
        self.bio_labels = bio_labels
        self.length = len(self.words)
        self.real_label_spans = real_label_spans
        
    def __str__(self):
        return "\n".join("%s: %s" % item for item in self.__dict__.items())
    
    def reset_conf_mat(self, conf_mat):
        self.conf_mat = conf_mat


class Span(object):
    def __init__(self, l, r, label, conf=None):
        self.l = l
        self.r = r
        self.label = label
        self.conf = conf
        
    def __str__(self):
        return " ".join("%s: %s" % item for item in self.__dict__.items())


def read_examples_from_file(file_path):
    if not os.path.exists(file_path):
        logger.info("[Warning] file {} not exists".format(file_path))
        return []

    data_df = pd.read_csv(file_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, keep_default_na=False, na_values=[''])
    examples = []
    guid, sentence, labels, real_labels = 0, [], [], []
    max_seq_length = 84
    for row in data_df.values:
        if not pd.isna(row[1]):
            sentence.append(row[0])
            labels.append(row[1])
            if len(row) == 3:
                real_labels.append(row[2])
        elif sentence != []:
            sentence = sentence[:max_seq_length]
            labels = labels[:max_seq_length]
            if len(real_labels) != []:
                real_labels = real_labels[:max_seq_length]
            span_begin, span_end, span_label, span_tuple = None, None, None, []
            for idx, label in enumerate(labels):
                if label[0] != 'I' and span_label is not None:
                    assert span_begin <= span_end
                    span_tuple.append((span_begin, span_end, span_label))
                    span_begin, span_end, span_label = None, None, None
                if label[0] == 'B':
                    span_label = label[2:]
                    span_begin = span_end = idx
                elif label[0] == 'I':
                    span_end = idx
            if span_label is not None:
                span_tuple.append((span_begin, span_end, span_label))
                span_begin, span_end, span_label = None, None, None
            
            # real labels
            span_begin, span_end, span_label, read_span_tuple = None, None, None, []
            for idx, label in enumerate(real_labels):
                if label[0] != 'I' and span_label is not None:
                    assert span_begin <= span_end
                    read_span_tuple.append((span_begin, span_end, span_label))
                    span_begin, span_end, span_label = None, None, None
                if label[0] == 'B':
                    span_label = label[2:]
                    span_begin = span_end = idx
                elif label[0] == 'I':
                    span_end = idx
            if span_label is not None:
                read_span_tuple.append((span_begin, span_end, span_label))
                span_begin, span_end, span_label = None, None, None
                
            examples.append(InputExample(guid=guid, words=sentence, label_spans=span_tuple, bio_labels=labels, real_label_spans=read_span_tuple))
            guid += 1
            sentence, labels, real_labels = [], [], []
    
    return examples


def load_examples(args, mode):
    class ExampleDataset(Dataset):
        def __init__(self, examples):
            super().__init__()
            self.examples = examples
        def __getitem__(self, index):
            return self.examples[index]
        def __len__(self):
            return len(self.examples)
        def merge(self, merge_dataset):
            self.examples.extend(merge_dataset.examples)
    
    file_path = os.path.join(args.data_dir, "%s.txt" % mode)
    if not os.path.exists(file_path):
        logger.info("%s dataset is empty" % mode)
        return None
    
    examples = read_examples_from_file(file_path)
    example_dataset = ExampleDataset(examples)
    
    return example_dataset


def bio_tagging(label_spans, length):
    tags = ["O"] * length
    for span in label_spans:
        for index in range(span.l, span.r + 1):
            if index == span.l:
                tags[index] = "B-" + span.label
            else:
                tags[index] = "I-" + span.label
    return tags


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def conflict_judge(span_list, cur_span):
    for span in span_list:
        if span.l > cur_span.r or span.r < cur_span.l:
            continue
        else:
            return True
    return False
