from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle as pkl
from config import config
from transformers import BertTokenizer
config=config()
from utils import keyword_pos2sta_vec
import sys
import os
from utils import WORD_START_IDX, QMASK_IDX, tokenizer
sys.path.insert(0,config.dict_path)
# from dict_use import *
from nltk.tokenize import word_tokenize

class dataset_str:
	def __init__(self, filename):
		self.data = []
		self.keys = []
		self.token_ids = []
		self.sta_vec_list=[]
		self.tokens = []
		with open(filename, 'r', encoding='utf-8') as f:
			for line in f:
				sent = line.strip().lower()
				if '|' in sent:
					splits = sent.split('|')
					sent = splits[0]
					keys = splits[1]
				else:
					sent = sent
					keys = sent
				keywords = []
				for k in keys.strip().split(' '):
					if len(k) > 0 and k in tokenizer.vocab:
						keywords.append(k)
				max_key = min(config.max_key, len(keywords))
				self.keys.append(keywords[:max_key])
				token_id = tokenizer.encode(sent.strip(), add_special_tokens=False)
				bert_tokenized = tokenizer.decode(token_id)
				self.tokens.append(word_tokenize(bert_tokenized))
				if config.mode[0] == 'q':
					if token_id[-1] < WORD_START_IDX:
						token_id = token_id[:-1]
					token_id.append(QMASK_IDX)
				self.token_ids.append(np.array(token_id))
		self.length = len(self.token_ids)