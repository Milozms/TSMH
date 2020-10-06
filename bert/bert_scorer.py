import torch
import numpy as np
import math
from transformers import BertTokenizer, BertForPreTraining, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

maxlen = 60

class BERT_Scorer:
	def __init__(self, pretrained='bert-base-uncased'):
		self.tokenizer = BertTokenizer.from_pretrained(pretrained)
		self.bert_model = BertForMaskedLM.from_pretrained(pretrained).to(device)
		self.mask_id = self.tokenizer._convert_token_to_id('[MASK]')
		self.sep_id = self.tokenizer._convert_token_to_id('[SEP]')
		self.cls_id = self.tokenizer._convert_token_to_id('[CLS]')

	def score(self, line, maxlen=None, log_prob=False):
		""" Deprecated """
		if type(line) == str:
			id_line = self.tokenizer.encode(line.strip(), add_special_tokens=False)
		else:
			id_line = line
		sent_len = len(id_line)
		if maxlen is not None:
			sent_len = min(sent_len, maxlen)
		input_tensor = torch.tensor(id_line[:sent_len]).unsqueeze(0).to(device)
		outputs = self.bert_model(input_tensor)
		prediction_scores = outputs[0]
		if log_prob:
			pred_log_probs = torch.log_softmax(prediction_scores, dim=2)
			pred_log_probs = pred_log_probs.detach().cpu().numpy()
			sent_log_prob = 0.0
			for tok_id in range(sent_len):
				sent_log_prob += pred_log_probs[0][tok_id][id_line[tok_id]]
			return sent_log_prob, pred_log_probs
		else:
			pred_probs = torch.softmax(prediction_scores, dim=2)
			pred_probs = pred_probs.detach().cpu().numpy()
			sent_prob = 1.0
			for tok_id in range(sent_len):
				sent_prob *= pred_probs[0][tok_id][id_line[tok_id]]
			return sent_prob, pred_probs

	def mask_score(self, sent_ids, mask_idx, mode=0, log_prob=False, maxlen=maxlen):
		if maxlen:
			if mask_idx > maxlen:
				dist = int(mask_idx - maxlen/2)
				sent_ids = sent_ids[dist:dist+maxlen]
				mask_idx -= dist
		sent_ids = np.concatenate([[self.cls_id], sent_ids, [self.sep_id]])
		mask_idx += 1
		if mode==0:
			masked_sent_ids = np.array(sent_ids)
			masked_sent_ids[mask_idx] = self.mask_id
		else:
			masked_sent_ids = np.concatenate([sent_ids[:mask_idx], [self. mask_id], sent_ids[mask_idx:]])

		sent_len = len(masked_sent_ids)
		if maxlen is not None:
			sent_len = min(sent_len, maxlen+2)
		input_tensor = torch.tensor(masked_sent_ids[:sent_len]).unsqueeze(0).to(device)
		outputs = self.bert_model(input_tensor)
		prediction_scores = outputs[0]
		if log_prob:
			log_pred_probs = torch.log_softmax(prediction_scores, dim=2)
			return log_pred_probs[0][mask_idx].detach().cpu().numpy()
		else:
			pred_probs = torch.softmax(prediction_scores, dim=2)
			return pred_probs[0][mask_idx].detach().cpu().numpy()

	def sent_score(self, line, maxlen=maxlen, log_prob=False, ignore_idx=-1):
		if type(line) == str:
			sent_ids = self.tokenizer.encode(line.strip(), add_special_tokens=False)
		else:
			sent_ids = line
		if len(sent_ids) == 0:
			if log_prob:
				return -math.inf
			else:
				return 0.0
		sent_ids = np.concatenate([[self.cls_id], sent_ids, [self.sep_id]])
		sent_len = len(sent_ids)
		if maxlen is not None:
			sent_len = min(sent_len, maxlen+2)
		input_tensor = torch.tensor((sent_len-2)*[sent_ids[:sent_len]]).to(device)
		for idx in range(sent_len - 2):
			input_tensor[idx][idx+1] = self.mask_id
		outputs = self.bert_model(input_tensor)
		prediction_scores = outputs[0]
		log_pred_probs = torch.log_softmax(prediction_scores, dim=2)
		sent_log_prob = 0.0
		for idx in range(sent_len - 2):
			tok_ind = idx + 1
			tok = sent_ids[tok_ind]
			if tok_ind != ignore_idx:
				sent_log_prob += log_pred_probs[idx][tok_ind][tok].item()
		if log_prob:
			return sent_log_prob
		else:
			return math.exp(sent_log_prob)


	def multi_mask_score(self, sent_ids, mask_idx_set, mode=0, log_prob=False, maxlen=None, output_ind=None):
		if output_ind is None:
			output_ind = min(mask_idx_set)
		if maxlen:
			assert maxlen >= max(mask_idx_set)
		if mode==0:
			masked_sent_ids = np.array(sent_ids)
			for mask_idx in mask_idx_set:
				masked_sent_ids[mask_idx] = self.mask_id
		else:
			raise NotImplementedError

		sent_len = len(masked_sent_ids)
		if maxlen is not None:
			sent_len = min(sent_len, maxlen)
		input_tensor = torch.tensor(masked_sent_ids[:sent_len]).unsqueeze(0).to(device)
		outputs = self.bert_model(input_tensor)
		prediction_scores = outputs[0]
		if log_prob:
			log_pred_probs = torch.log_softmax(prediction_scores, dim=2)
			return log_pred_probs[0][output_ind].detach().cpu().numpy()
		else:
			pred_probs = torch.softmax(prediction_scores, dim=2)
			return pred_probs[0][output_ind].detach().cpu().numpy()

	def id2sent(self, ids):
		return self.tokenizer.decode(ids)


	def close(self):
		pass



class GPT2_Scorer:
	def __init__(self, pretrained='gpt2'):
		self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained)
		self.gpt2_model = GPT2LMHeadModel.from_pretrained(pretrained).to(device)
		self.pad_id = self.tokenizer._convert_token_to_id('<|endoftext|>')


	def sent_score(self, line, maxlen=None, log_prob=False, ppl=False):
		id_line = self.tokenizer.encode(line.strip(), add_special_tokens=False)
		id_line = [self.pad_id] + id_line + [self.pad_id]
		sent_len = len(id_line)
		if maxlen is not None:
			sent_len = min(sent_len, maxlen+2)
		input_tensor = torch.tensor(id_line[:sent_len]).unsqueeze(0).to(device)
		try:
			outputs = self.gpt2_model(input_tensor, labels=input_tensor)
		except RuntimeError:
			print('RuntimeError in GPT2_Scorer.sent_score, input line:', line)
			return 0.0
		loss, prediction_scores = outputs[:2]
		# log_pred_probs = torch.log_softmax(prediction_scores[0], dim=-1)
		# sent_log_prob = 0.0
		# for idx in range(sent_len - 1):
		#     tok = id_line[idx+1]
		#     sent_log_prob += log_pred_probs[idx][tok].item()
		sent_log_prob = -loss.item()*(sent_len-1)
		if ppl:
			ppl_val = math.pow(math.exp(sent_log_prob), -1/(sent_len-1))
			return ppl_val
		elif log_prob:
			return sent_log_prob, sent_len-1
		else:
			return math.exp(sent_log_prob)

	def sent_score_batch(self, lines, maxlen=None, log_prob=False, batch_size=100):
		if not torch.cuda.is_available():
			batch_size=16
		cnt = len(lines)
		sent_ids = [[self.pad_id]+self.tokenizer.encode(line.strip(), add_special_tokens=False)+[self.pad_id] for line in lines]
		batches = [sent_ids[i:i+batch_size] if i+batch_size<=cnt else sent_ids[i:] for i in range(0, cnt, batch_size)]
		results = []
		for batch_sent_ids in batches:
			bsize = len(batch_sent_ids)
			sent_max_len = max([len(s) for s in batch_sent_ids])
			if maxlen is not None:
				sent_max_len = min(sent_max_len, maxlen+2)
			input_tensor = torch.zeros(bsize, sent_max_len, dtype=torch.long).fill_(self.pad_id)
			for idx, sent_id in enumerate(batch_sent_ids):
				sent_len = min(len(sent_id), sent_max_len)
				input_tensor[idx][:sent_len] = torch.tensor(sent_id[:sent_len])
			input_tensor = input_tensor.to(device)
			outputs = self.gpt2_model(input_tensor, labels=input_tensor)
			loss, prediction_scores = outputs[:2]
			for idx, sent_id in enumerate(batch_sent_ids):
				sent_len = min(len(sent_id), sent_max_len)
				pred_log_probs = torch.log_softmax(prediction_scores, dim=2)
				sent_log_prob = 0.0
				for tok_id in range(sent_len - 1):
					sent_log_prob += pred_log_probs[idx][tok_id][sent_id[tok_id + 1]].item()
				if log_prob:
					results.append(sent_log_prob)
				else:
					results.append(math.exp(sent_log_prob))
		return results