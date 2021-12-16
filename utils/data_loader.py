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

pairs_tra, pairs_val, pairs_tst, vocab = load_dataset() # list of dict

# def process_fn(data):
#   """Returns one data pair (source and target)."""
#   emo_map = {
#         'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
#         'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13, 'anxious': 14, 'disappointed': 15,
#         'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23,
#         'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}
#   item = {}
#   item["context_text"] = data["context"]
#   item["target_text"] = data["target"]
#   item["emotion_text"] = data["emotion"]

#   item["context"], item["context_mask"] = preprocess(item["context_text"])

#   item["target"] = preprocess(item["target_text"], anw=True)
#   item["emotion"], item["emotion_label"] = preprocess_emo(item["emotion_text"], emo_map)

#   return item

def preprocess(arr, anw=False):
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

def make_ds(data):
  dataset = []
  # print(type(data))  # <class 'dict'>
  # print(data.keys())  # dict_keys(['context', 'target', 'emotion', 'situation'])

  #ld_to_dl = {k: [dic[k] for dic in data] for k in data}  # convert "list of dict" to "dict of list"
  #print(ld_to_dl)
  #for k, v in ld_to_dl.items():
  for k, v in data.items():
    print(v)
    if k in ['seq_of_seqs_feature','sequence_feature']:  # add in all uncertain length target
      print(k)
      dataset.append(tf.data.Dataset.from_tensor_slices(tf.ragged.constant(v)))
    else:
      dataset.append(tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(v)))
  return tf.data.Dataset.zip(tuple(dataset))

# ds = make_ds(data).map(
#     lambda context, target_text, emo : preprocess(context), preprocess(target_text), preprocess_emo(emo)
# ).batch(batch_size = 16)


def prepare_data_seq(batch_size=32):  

    
    # data_loader_tra = make_ds(process_fn(pairs_tra))
    # data_loader_val = make_ds(process_fn(pairs_val))
    # data_loader_tst = make_ds(process_fn(pairs_tst))
    logging.info("Vocab  {} ".format(vocab.n_words))
    #return data_loader_tra, data_loader_val, data_loader_tst, vocab  #, len(dataset_train.emo_map)
    return make_ds(pairs_tra)

