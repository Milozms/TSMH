from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
from config import config
from nltk.tokenize import word_tokenize
from copy import copy, deepcopy
import sys
import json
config=config()
sys.path.insert(0,config.skipthoughts_path)
sys.path.insert(0,config.emb_path)
# sys.path.insert(0,'../utils/dict_emb')
from dict_use import dict_use
sys.path.insert(0,'../bert')
from bert_scorer import BERT_Scorer, GPT2_Scorer
bert_scorer = BERT_Scorer(config.bert_path)
gpt2_scorer = GPT2_Scorer(config.gpt2_path)
tokenizer = bert_scorer.tokenizer
PAD_IDX = tokenizer._convert_token_to_id('[PAD]')  # 0
MASK_IDX = tokenizer._convert_token_to_id('[MASK]')  # 103
QMASK_IDX = tokenizer._convert_token_to_id('?')
WORD_START_IDX = 1996

if config.mode[0] == 's':
	import fasttext
	# model_dir = './yelp_polarity_model'
	model_dir = '../sentiment/yelp_polarity_model'
	model = fasttext.load_model(model_dir)

# os.environ['CORENLP_HOME'] = '/home/zms/stanford-corenlp-full-2018-10-05/'
# os.environ['CORENLP_HOME'] = '/Users/zms/stanford-corenlp-full'
# export CORENLP_HOME=/Users/zms/stanford-corenlp-full
# export CORENLP_HOME=/home/zms/stanford-corenlp-full-2018-10-05/
# export CORENLP_HOME=/home/maosen/stanford-corenlp-full-2018-10-05/
# client = corenlp.CoreNLPClient(annotators="parse".split())

import pickle as pkl
if config.sim=='word_max' or config.sim=='combine':
	emb_word,emb_id=pkl.load(open(config.emb_path, 'rb'), encoding='latin1')


dict_use=dict_use(config.dict_path)
sen2id=dict_use.sen2id
id2sen=dict_use.id2sen

aux_verbs = ['do', 'does', 'did', 'be', 'am', 'are', 'is', 'was', 'were',
			 'shall', 'will', 'should', 'would', 'can', 'could', 'may', 'might', 'must']
question_words = ['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how']
start_words = aux_verbs + question_words
start_word_ids = tokenizer.convert_tokens_to_ids(start_words)
question_word_ids = tokenizer.convert_tokens_to_ids(question_words)
aux_verb_ids = tokenizer.convert_tokens_to_ids(aux_verbs)


if config.sim=='skipthoughts' or config.sim=='combine':
	import skipthoughts
	skip_model = skipthoughts.load_model()
	skip_encoder = skipthoughts.Encoder(skip_model)
if config.sim=='word_max' or config.sim=='combine':
	#id2freq=pkl.load(open('./data/id2freq.pkl'))
	pass
def normalize(x, e=0.05):
	tem = copy(x)
	if max(tem)==0:
		tem+=e
	return tem/tem.sum()

class ConstraintSearch:
	def __init__(self, keywords):
		# sentence template
		self.keyword_ids = tokenizer.convert_tokens_to_ids(keywords)
		self.temp2set = {
			'[OTHER]': set(list(range(len(tokenizer.vocab))))
		}
		# task specific
		if config.mode[0] == 'i':
			self.temp2set['[PLZ]'] = set(tokenizer.convert_tokens_to_ids(['please', 'always', 'never']))
			with open(config.verb_path) as fverb:
				verb_set = set(json.load(fverb))
				verb_set -= self.temp2set['[PLZ]']
			self.keywords = []
			for kw, kwid in zip(keywords, self.keyword_ids):
				if kwid in verb_set:
					kw = '[VER]-' + kw
					verb_set.discard(kwid)
				self.temp2set[kw] = set([kwid])
				self.keywords.append(kw)
			self.temp2set['[VER]'] = verb_set
		else:
			self.keywords = keywords
			if config.mode[0] == 'q':
				# question generation
				self.temp2set['[AUX]'] = set(aux_verb_ids)
				self.temp2set['[QWH]'] = set(question_word_ids)
				self.temp2set['[?]'] = set(tokenizer.convert_tokens_to_ids(['?']))
			elif config.mode[0] == 'p':
				with open(config.participle_path) as fpart:
					part_set = set(json.load(fpart))
					self.temp2set['[VBN]'] = set(part_set)
					self.temp2set['[VBE]'] = \
						set(tokenizer.convert_tokens_to_ids(['be', 'am', 'are', 'is', 'was', 'were']))
			for kw, kwid in zip(keywords, self.keyword_ids):
				self.temp2set[kw] = set([kwid])
		self.temptags = list(self.temp2set.keys())
		for tag, tset in self.temp2set.items():
			if tag != '[OTHER]':
				self.temp2set['[OTHER]'] -= tset
		self.results = []
		self.all_act_set = self.enumerate_all_act_set()

	@staticmethod
	def is_verb(tag):
		return len(tag) >= 5 and tag[:5] == '[VER]'

	def count_verb(self, temp):
		result = 0
		for tag in self.temp2set.keys():
			if self.is_verb(tag):
				result += temp.count(tag)
		return result

	def count_unsafisfied_constraint(self, temp):
		result = 0
		for kw in self.keywords:
			result += abs(temp.count(kw) - 1)
		if config.mode[0] == 'q':
			result += 7 - (int(temp[0] == '[QWH]')
				     + int('[AUX]' in temp)
				     + int(temp[-1] == '[?]')
				     + int(temp.count('[?]') == 1)
				     + int(temp.count('[AUX]') == 1)
				     + int(temp[0] == '[QWH]' and temp.count('[QWH]') == 1)
				     + int(('[AUX]' in temp) and (temp.index('[AUX]') in [1, 2])))
		elif config.mode[0] == 'i':
			result += 1 - int((len(temp)>=1 and self.is_verb(temp[0])) or
			                  (len(temp)>=2 and temp[0] == '[PLZ]' and self.is_verb(temp[1]))) # + abs(self.count_verb(temp) - 1)
		elif config.mode[0] == 'p':
			result += abs(temp.count('[VBN]') - 1) \
			          + abs(temp.count('[VBE]') - 1) \
			          + 1 - int(temp.count('[VBN]')>0
			                    and temp.count('[VBE]')>0
			                    and temp.index('[VBN]') - temp.index('[VBE]')==1)
		return result


	def sent_id2temp_tag(self, tid):
		'''
		convert bert id to template tag
		'''
		for tag, tset in self.temp2set.items():
			if tid in tset:
				return tag
		return '[OTHER]'

	def sent2tag(self, sent_ids):
		temp = []
		for tid in sent_ids:
			temp.append(self.sent_id2temp_tag(tid))
		return temp

	@staticmethod
	def enumerate_all_act_set(num_pos=4, num_step=config.step_size):
		all_act_set = []
		queue = [[]]
		while len(queue) > 0:
			top = queue.pop(0)
			for pos in range(num_pos):
				cur = deepcopy(top)
				cur.append(pos)
				if len(cur) < num_step:
					queue.append(cur)
				else:
					all_act_set.append(cur)
		invalid = [3]*num_step
		if invalid in all_act_set:
			all_act_set.remove(invalid)
		return all_act_set


	def dfs_template(self, temp, depth, pos_set, action_set):
		if depth >= len(pos_set) or depth >= len(action_set):
			if len(temp) > 0:
				cnt_constr = self.count_unsafisfied_constraint(temp)
				self.results.append((temp, cnt_constr, action_set))
			return
		pos = pos_set[depth]
		act = action_set[depth]
		if act in [0, 1]:
			if pos < 0 or pos >= len(temp) or temp[pos] != '[MASK]':
				# print('Invalid action combination.')
				return
			assert temp[pos] == '[MASK]'
			for cand in self.temptags:
				temp = deepcopy(temp)
				temp[pos] = cand
				self.dfs_template(temp, depth+1, pos_set, action_set)
		else:
			self.dfs_template(temp, depth + 1, pos_set, action_set)


	def search_template(self, sent_ids, pos_set, action_set=None, prune=True):
		temp_0 = self.sent2tag(sent_ids)
		self.results = []
		if action_set is not None:
			# first sample action, then search
			temp, adjusted_pos_set = mask_template(temp_0, pos_set, action_set)
			self.dfs_template(temp, 0, adjusted_pos_set, action_set)
		else:
			# first search, then get action
			for action_set in self.all_act_set:
				temp, adjusted_pos_set = mask_template(temp_0, pos_set, action_set)
				if temp is not None and adjusted_pos_set is not None:
					self.dfs_template(temp, 0, adjusted_pos_set, action_set)
		min_constr_num = min([temp[1] for temp in self.results]) if config.pen_num_constr else 0
		if prune:
			self.results = [(temp[0], temp[2]) for temp in self.results if temp[1] == min_constr_num]
		return self.results, min_constr_num

	def generate_candidate_input_with_mask(self, input_ids, ind, prob, search_size, temp_tag=None):
		# get top-k candidates
		prob_tmp = np.array(prob)
		if temp_tag is not None:
			if temp_tag in self.temp2set:
				temp_tag_set = self.temp2set[temp_tag]
				for i in range(len(prob_tmp)):
					if i not in temp_tag_set:
						prob_tmp[i] = -np.inf
				search_size = min(search_size, len(temp_tag_set))
			elif temp_tag != '[OTHER]':
				print('Warning: %s not in template tags.' % temp_tag)

		input_candidate = np.array([input_ids] * search_size)
		tok_candidate = np.argsort(prob_tmp)[-search_size:]

		for i in range(search_size):
			input_candidate[i][ind] = tok_candidate[i]

		prob_candidate = np.array([prob[tok] for tok in tok_candidate])
		return input_candidate, prob_candidate


def mask_template(temp_, pos_set, action_set):
	temp = deepcopy(temp_)
	adjusted_pos_set = np.array(pos_set)
	for idx, (pos, act) in enumerate(zip(pos_set, action_set)):
		if act == 0:  # replace
			if pos >= len(temp):
				return None, None
			temp[pos] = '[MASK]'
			adjusted_pos_set[:idx] += 0
		elif act == 1:  # insert
			if pos > len(temp):
				return None, None
			temp = temp[:pos] + ['[MASK]'] + temp[pos:]
			adjusted_pos_set[:idx] += 1
		elif act == 2:  # delete
			if pos >= len(temp):
				return None, None
			temp = temp[:pos] + temp[pos + 1:]
			adjusted_pos_set[:idx] -= 1
	return temp, adjusted_pos_set

def mask_sentence(input, pos_set, action_set):
	sent = np.array(input)   # copy
	adjusted_pos_set = np.array(pos_set)
	for idx, (pos, act) in enumerate(zip(pos_set, action_set)):
		if act == 0:  # replace
			sent[pos] = MASK_IDX
			adjusted_pos_set[:idx] += 0
		elif act == 1:  # insert
			sent = np.concatenate([sent[:pos], [MASK_IDX], sent[pos:]])
			adjusted_pos_set[:idx] += 1
		elif act == 2:  # delete
			sent = np.concatenate([sent[:pos], sent[pos+1:]])
			adjusted_pos_set[:idx] -= 1
	return sent, adjusted_pos_set

def intersection(listA, listB):
	return list(set(listA).intersection(set(listB)))

def get_sent_root_tag(sent, client):
	try:
		ann = client.annotate(sent)
		tag = ann.sentence[0].parseTree.child[0].value
		return tag
	except:
		return 'S'

def get_sample_positions(seq_len, prev_inds, step_size):
	candidates = []
	if step_size >= seq_len:
		print('Warning: too short sequence length', file=sys.stderr)
		if len(prev_inds) >= seq_len:
			candidates = list(range(seq_len))
		else:
			for ind in range(seq_len):
				if ind not in prev_inds:
					candidates.append(ind)
		np.random.shuffle(candidates)
		return [candidates[0]]
	if step_size + len(prev_inds) >= seq_len:
		candidates = list(range(seq_len))
	else:
		for ind in range(seq_len):
			if ind not in prev_inds:
				candidates.append(ind)
	np.random.shuffle(candidates)
	pos_set = candidates[:step_size]
	pos_set = sorted(pos_set, reverse=True)   # descending order, for avoiding conflicts
	return pos_set

def get_reverse_action(act):
	assert act <= 3 and act >= 0
	reverses = [0, 2, 1, 3]
	return reverses[act]

def get_reverse_action_set(action_set):
	return [get_reverse_action(act) for act in action_set]


def generate_candidate_input_with_mask(input_ids, sequence_length, ind, prob, search_size, old_tok=-1,
									   mode=0, keep_non=False):
	# difference:
	# 1. index is already adjusted
	# 2. input is already masked
	# 3. for inserting: mask is already added, only consider replace
	# 4. when ind == 0, constraint on start word
	# 5. return new normalized prob
	# sequence_length_new=np.array([sequence_length]*search_size)
	# get top-k candidates
	prob_tmp = np.array(prob)
	# if ind == 0:
	# 	for i in range(len(prob_tmp)):
	# 		if i not in start_word_ids:
	# 			prob_tmp[i] = -np.inf
	# 	search_size = len(start_word_ids)

	input_candidate=np.array([input_ids]*search_size)
	tok_candidate = np.argsort(prob_tmp)[-search_size:]

	for i in range(search_size):
		input_candidate[i][ind] = tok_candidate[i]

	# dealing with reverse proposal
	reverse_candidate_idx = -1
	if mode == 0:
		if old_tok in tok_candidate:
			for reverse_candidate_idx in range(len(tok_candidate)):
				if old_tok == tok_candidate[reverse_candidate_idx]:
					break
		if reverse_candidate_idx < 0 or reverse_candidate_idx >= len(input_candidate):
			# print('Warning: reverse candidate not sampled.', file=sys.stderr)
			reverse_candidate_idx = len(input_candidate)
			reverse_candidate = np.array(input_ids)
			reverse_candidate[ind] = old_tok
			input_candidate = np.concatenate([input_candidate, [reverse_candidate]], axis=0)
			tok_candidate = np.concatenate([tok_candidate, [old_tok]])

	prob_candidate = np.array([prob[tok] for tok in tok_candidate])

	non_idx = -1
	if mode == 1 and keep_non:
		non_cand = np.concatenate([input_ids[:ind], input_ids[ind+1:], [PAD_IDX]])
		prob_base = bert_scorer.sent_score(input_ids, ignore_idx=ind)  # P(x1, x2, ..., x_m-1, [MASK], x_m, x_m+1, ..., x_N)
		prob_non = bert_scorer.sent_score(non_cand)
		non_idx = len(input_candidate)
		input_candidate = np.concatenate([input_candidate, [non_cand]])
		prob_candidate = np.concatenate([prob_candidate * prob_base, [prob_non]])

	return input_candidate, prob_candidate, reverse_candidate_idx, non_idx


'''
def sample_from_candidate(prob_candidate):
	return np.argmax(prob_candidate)
''' 
def sample_from_candidate(prob_candidate):
	return choose_action(normalize(prob_candidate))

def choose_action(c):
	r=np.random.random()
	c=np.array(c)
	for i in range(1, len(c)):
		c[i]=c[i]+c[i-1]
	for i in range(len(c)):
		if c[i]>=r:
			return i

def bert_ids_to_dict_ids(s):
	sent = tokenizer.decode(s)
	sent_tokens = word_tokenize(sent)
	return sen2id(sent_tokens)

def sentence_embedding(s):
	emb_sum=0
	cou=0
	for item in s:
		if item<config.dict_size:
			emb_sum+=emb[item]
			cou+=1
	return emb_sum/(cou+0.0001)
'''
def similarity(s1, s2):
	e1=sentence_embedding(s1)
	e2=sentence_embedding(s2)
	cos=(e1*e2).sum()/((e1**2).sum()*(e2**2).sum())**0.5
	return cos**config.sim_hardness
'''
if config.sim=='skipthoughts' or config.sim=='combine':
	def sigma_skipthoughts(x):
		return (np.abs(1-((x-1)*2)**2)+(1-((x-1)*2)**2))/2.0
		#return 1
	def similarity_skipthoughts(s1, s2):
		#s2 is reference_sentence
		s1=' '.join(id2sen(s1))
		s2=' '.join(id2sen(s2))
		#print(s1,s2)
		e=skip_encoder.encode([s1,s2])
		e1=e[0]
		e2=e[-1]
		cos=(e1*e2).sum()/((e1**2).sum()*(e2**2).sum())**0.5
		return sigma_skipthoughts(cos)
	def similarity_batch_skipthoughts(s1, s2):
		#s2 is reference_sentence
		s1=[' '.join(id2sen(x)) for x in s1]
		s2=' '.join(id2sen(s2))
		s1.append(s2)
		e=skip_encoder.encode(s1)
		e1=e[:-1]
		e2=e[-1]
		cos=(e1*e2).sum(axis=1)/((e1**2).sum(axis=1)*(e2**2).sum())**0.5
		return sigma_skipthoughts(cos)

if config.sim=='word_max' or config.sim=='combine':
	def sigma_word(x):
		return x
		# if x > 0.8:
		# 	return x
		# elif x > 0.6:
		# 	return 4 * (x - 0.6)
		# else:
		# 	return 0.0
		# if x>0.7:
		# 	return x
		# elif x>0.65:
		# 	return (x-0.65)*14
		# else:
		# 	return 0
		#return max(0, 1-((x-1))**2)
		#return (((np.abs(x)+x)*0.5-0.6)/0.4)**2
	def sen2mat(s):
		mat=[]
		for item in s:
			if item==config.dict_size+2:
				continue
			if item==config.dict_size+1:
				break
			word=id2sen([item])[0]
			if  word in emb_word:
				mat.append(np.array(emb_word[word]))
			else:
				mat.append(np.random.random([config.hidden_size]))
		return np.array(mat)
	def similarity_word(s1,s2, sta_vec):
		'''
		similarity AND constraint
		'''
		return 1.0
	def similarity_word_key(s1,s2, sta_vec):
		s1 = bert_ids_to_dict_ids(s1)
		if type(s2[0]) == str:
			s2 = sen2id(s2)
		else:
			s2 = bert_ids_to_dict_ids(s2)
		e=1e-5
		emb1=sen2mat(s1)
		#wei2=normalize( np.array([-np.log(id2freq[x]) for x in s2 if x<=config.dict_size]))
		emb2=sen2mat(s2)
		wei2=np.array(sta_vec[:len(emb2)]).astype(np.float32)    # keyword  len(s2)
		#wei2=normalize(wei2)

		emb_mat=np.dot(emb2,emb1.T)                            # (len(s2), len(s1))
		norm1=np.diag(1/(np.linalg.norm(emb1,2,axis=1)+e))     # (len(s1), len(s1))
		norm2=np.diag(1/(np.linalg.norm(emb2,2,axis=1)+e))     # (len(s2), len(s2))
		sim_mat=np.dot(norm2,emb_mat).dot(norm1)               # (len(s2), len(s1))
		sim_vec=sim_mat.max(axis=1)                            #  len(s2)  # for each word in s2, the closest word in s1
		#sim=(sim_vec*wei2).sum()
		sim=min([x for x in list(sim_vec*wei2) if x>0]+[1])    # for each keyword in s2, the closest word in s1
		#sim=(sim_vec).mean()
		return sigma_word(sim)

	def similarity_batch_word(s1, s2, sta_vec=None):
		return np.array([ similarity_word(x,s2,sta_vec) for x in s1 ])

if config.sim=='skipthoughts':
	similarity=similarity_skipthoughts
	similarity_batch=similarity_batch_skipthoughts
elif config.sim=='word_max':
	similarity=similarity_word
	similarity_batch=similarity_batch_word
elif config.sim=='combine':
	def similarity(s1,s2):
		return (similarity_skipthoughts(s1, s2)+similarity_word(s1, s2))/2.0
	def similarity_batch(s1,s2):
		return (similarity_batch_skipthoughts(s1, s2)+similarity_batch_word(s1, s2))/2.0

if config.sim is None:
	def similarity(s1,s2):
		return 1.0
	def similarity_batch(s1,s2):
		return 1.0

def keyword_pos2sta_vec(keyword, pos):
	key_ind=[]
	# pos=pos[:config.num_steps-1]
	for i in range(len(pos)):
		if pos[i]=='NNP':
			key_ind.append(i)
		elif pos[i] in ['NN', 'NNS'] and keyword[i]==1:
			key_ind.append(i)
		elif pos[i] in ['VBZ'] and keyword[i]==1:
			key_ind.append(i)
		elif keyword[i]==1:
			key_ind.append(i)
		elif pos[i] in ['NN', 'NNS','VBZ']:
			key_ind.append(i)
	key_ind=key_ind[:max(int(config.max_key_rate*len(pos)), config.max_key)]
	sta_vec=[]
	for i in range(len(keyword)):
		if i in key_ind:
			sta_vec.append(1)
		else:
			sta_vec.append(0)
	return sta_vec



def just_acc():
	r=np.random.random()
	if r<config.just_acc_rate:
		return 0
	else:
		return 1

def write_log(str, path):
	with open(path, 'a') as g:
		g.write(str+'\n')


def penalty_constraint(pen_constr):
	if not config.pen_num_constr:
		if pen_constr > 0:
			return 0.0
	if config.pen_num_constr:
		return math.pow(config.pen_num_constr_base, pen_constr)



def get_sentiment_score(line, mode='neg'):
	if type(line) != str:
		line = tokenizer.decode(line)
	assert mode[0] in ['p', 'n']
	mode2label = {'p': '__label__2', 'n': '__label__1'}
	labels, probs = model.predict(line)
	if mode2label[mode[0]] == labels[0]:
		return probs[0]
	else:
		return 1.0 - probs[0]


def get_batch_sentiment_scores(batch, mode='neg'):
	return np.array([get_sentiment_score(x, mode) for x in batch])

reverse_label = {'positive': 'neg', 'negative': 'pos'}