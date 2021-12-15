import sys
import os
from numpy import int64
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import tensorflow as tf
#import torch.utils.data as data
import random
import math
import os
import logging 
from utils import config
import pickle
from tqdm import tqdm
import pprint
pp = pprint.PrettyPrinter(indent=1)
import re
import ast
## from utils.nlp import normalize
import time
from model.common_layer_wz import write_config  # @wz
from utils.data_reader import load_dataset

tf.random.set_seed(1)

# class Dataset():
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab, batch_size, shuffle=True):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.emo_map = {
        'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
        'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13, 'anxious': 14, 'disappointed': 15,
        'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23,
        'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}
        random.seed(0)
        random.shuffle(data) 

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data["target"]) // self.batch_size

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]

        item["context"], item["context_mask"] = self.preprocess(item["context_text"])

        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"], self.emo_map)

        return item

    def preprocess(self, arr, anw=False):
        """Converts words to ids."""
        if(anw):
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in arr] + [config.EOS_idx]
            return tf.Variable(sequence, dtype=tf.int64)
        else:
            X_dial = [config.CLS_idx]
            X_mask = [config.CLS_idx]
            for i, sentence in enumerate(arr):
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in sentence]
                spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                X_mask += [spk for _ in range(len(sentence))] 
            assert len(X_dial) == len(X_mask)

            return tf.Variable(X_dial, dtype=tf.int64), tf.Variable(X_mask, dtype=tf.int64)

    def preprocess_emo(self, emotion, emo_map):
        program = [0]*len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]

def process_fn(data):
  """Returns one data pair (source and target)."""
  emo_map = {
        'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
        'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13, 'anxious': 14, 'disappointed': 15,
        'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23,
        'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}
  item = {}
  item["context_text"] = data["context"]
  item["target_text"] = data["target"]
  item["emotion_text"] = data["emotion"]

  item["context"], item["context_mask"] = preprocess(item["context_text"])

  item["target"] = preprocess(item["target_text"], anw=True)
  item["emotion"], item["emotion_label"] = preprocess_emo(item["emotion_text"], emo_map)

  return item

def preprocess(vocab, arr, anw=False):
  """Converts words to ids."""
  if(anw):
      sequence = [vocab.word2index[word] if word in vocab.word2index else config.UNK_idx for word in arr] + [config.EOS_idx]
      return tf.convert_to_tensor(sequence)
  else:
      X_dial = [config.CLS_idx]
      X_mask = [config.CLS_idx]
      for i, sentence in enumerate(arr):
          X_dial += [vocab.word2index[word] if word in vocab.word2index else config.UNK_idx for word in sentence]
          spk = vocab.word2index["USR"] if i % 2 == 0 else vocab.word2index["SYS"]
          X_mask += [spk for _ in range(len(sentence))] 
      assert len(X_dial) == len(X_mask)

      return tf.convert_to_tensor(X_dial), tf.convert_to_tensor(X_mask)

def preprocess_emo(emotion, emo_map):
    program = [0]*len(emo_map)
    program[emo_map[emotion]] = 1

    return program, emo_map[emotion]



## to get the dataset
import pandas as pd
# records needs to be linked to the original dataset
dataset = tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(records).to_dict(orient="list")).map(process_fn)

# {'a': [0, 1], 'b': [2, 3]} -> [{'a': 0, 'b': [2,3]} -> [{"a": 0, 'b': 2}, {'a':1, 'b':3}]
# ref: https://stackoverflow.com/questions/68567630/converting-a-list-of-dictionaries-to-a-tf-dataset

######################
# def collate_fn(data):
#     def merge(sequences):
#         lengths = [len(seq) for seq in sequences]
#         padded_seqs = tf.ones([len(sequences), max(lengths)], dtype=tf.int64) ## padding index 1
#         for i, seq in enumerate(sequences):
#             end = lengths[i]
#             padded_seqs[i, :end] = seq[:end]
#         return padded_seqs, lengths 


#     data.sort(key=lambda x: len(x["context"]), reverse=True) ## sort by source seq
#     item_info = {}
#     for key in data[0].keys():
#         item_info[key] = [d[key] for d in data]

#     ## input
#     input_batch, input_lengths     = merge(item_info['context'])
#     mask_input, mask_input_lengths = merge(item_info['context_mask'])

#     ## Target
#     target_batch, target_lengths   = merge(item_info['target'])


#     if config.USE_CUDA:
#         input_batch = input_batch.cuda()
#         mask_input = mask_input.cuda()
#         target_batch = target_batch.cuda()
 
#     d = {}
#     d["input_batch"] = input_batch
#     d["input_lengths"] = tf.Variable(input_lengths, dtype=int64)
#     d["mask_input"] = mask_input
#     d["target_batch"] = target_batch
#     d["target_lengths"] = tf.Variable(target_lengths, dtype=int64)
#     ##program
#     d["target_program"] = item_info['emotion']
#     d["program_label"] = item_info['emotion_label']

#     ##text
#     d["input_txt"] = item_info['context_text']
#     d["target_txt"] = item_info['target_text']
#     d["program_txt"] = item_info['emotion_text']
#     return d 


# def prepare_data_seq(batch_size=32):  

    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=collate_fn)

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=collate_fn)
    #print('val len:',len(dataset_valid))
    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test,
                                                 batch_size=1,
                                                 shuffle=False, collate_fn=collate_fn)
    write_config()
    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map)