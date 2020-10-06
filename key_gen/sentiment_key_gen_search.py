from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import locale
os.environ["PYTHONIOENCODING"] = "utf-8"
# myLocale=locale.setlocale(category=locale.LC_ALL, locale="en_GB.UTF-8")
import sys
import time
import torch
print('Checking CUDA status:')
print(torch.cuda.is_available())
if torch.cuda.is_available():
	print(torch.cuda.current_device())
	print(torch.cuda.device(0))
	print(torch.cuda.device_count())
	print(torch.cuda.get_device_name(0))

import numpy as np
from reader import dataset_str
from config import config
from scipy.special import perm, comb
import csv
config=config()
import os 
# os.environ['CUDA_VISIBLE_DEVICES']=config.GPU
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)

from utils import choose_action, similarity, similarity_batch, normalize, sample_from_candidate, just_acc, \
	get_sample_positions, mask_sentence, MASK_IDX, PAD_IDX, bert_scorer, gpt2_scorer, \
	tokenizer, ConstraintSearch, get_reverse_action_set, penalty_constraint, get_sentiment_score, \
	get_batch_sentiment_scores, reverse_label

def_sent_scorer = bert_scorer.sent_score
sentiment = 'positive'
# proposal_sentiment_score = config.proposal_sentiment_score
proposal_sentiment_score = False   # suggested

def eval_template(searcher, input_original, cand_template, masked_sent, adjusted_pos_set, action_set, sim=None, verbose=False):
	proposal_prob = 1.0  # Q(x'|x)
	input_ids_tmp = np.array(masked_sent)  # copy

	for step_i in range(len(adjusted_pos_set)):
		ind = adjusted_pos_set[step_i]
		# ind_old = pos_set[step_i]
		action = action_set[step_i]
		if action in [0, 1]:
			temp_tag = cand_template[ind]
		else:
			temp_tag = None

		# word replacement (action: 0)
		if action == 0:
			prob_mask = bert_scorer.mask_score(input_ids_tmp, ind, mode=0)
			input_candidate, prob_candidate = \
				searcher.generate_candidate_input_with_mask(input_ids_tmp, ind, prob_mask,
				                                            config.search_size, temp_tag=temp_tag)
			if sim is not None:
				similarity_candidate = similarity_batch(input_candidate, input_original)
				prob_candidate = prob_candidate * similarity_candidate
			if config.mode == 'sentiment' and proposal_sentiment_score:
				prob_candidate *= get_batch_sentiment_scores(input_candidate, sentiment)
			prob_candidate_norm = prob_candidate  # no normalize here
			prob_candidate_ind = sample_from_candidate(prob_candidate_norm)
			input_ids_tmp = input_candidate[prob_candidate_ind]  # changed
			proposal_prob *= prob_candidate_norm[prob_candidate_ind]  # Q(x'|x)
			if verbose:
				print('action:0, pos:', ind, prob_candidate_norm[prob_candidate_ind])

		# word insertion(action:1)
		if action == 1:
			prob_mask = bert_scorer.mask_score(input_ids_tmp, ind, mode=0)

			input_candidate, prob_candidate = \
				searcher.generate_candidate_input_with_mask(input_ids_tmp, ind, prob_mask,
				                                            config.search_size, temp_tag=temp_tag)
			if sim is not None:
				similarity_candidate = similarity_batch(input_candidate, input_original)
				prob_candidate = prob_candidate * similarity_candidate
			if config.mode == 'sentiment' and proposal_sentiment_score:
				prob_candidate *= get_batch_sentiment_scores(input_candidate, sentiment)
			prob_candidate_norm = prob_candidate  # no normalize here
			prob_candidate_ind = sample_from_candidate(prob_candidate_norm)
			input_ids_tmp = input_candidate[prob_candidate_ind]

			proposal_prob *= prob_candidate_norm[prob_candidate_ind]  # Q(x'|x)
			if verbose:
				print('action:1, pos:', ind, prob_candidate_norm[prob_candidate_ind])

		# word deletion(action: 2)
		if action == 2:
			input_ids_tmp = input_ids_tmp  # already deleted

			proposal_prob *= 1.0  # Q(x'|x)
			if verbose:
				print('action:2, pos:', ind, 1.0)

	if verbose:
		print(cand_template)
		print(tokenizer.decode(input_ids_tmp).encode('utf8', errors='ignore'))
	return proposal_prob, input_ids_tmp

# warning: here the sentiment scores are not normalized, which is asymmetrical with previous function!!!
def eval_reverse_proposal(input_original, masked_sent, input_ids_old, pos_set, reverse_action_set, sim=None):
	proposal_prob_reverse = 1.0  # Q(x|x')
	input_ids_tmp = np.array(masked_sent)
	for step_i in range(len(pos_set)):
		ind = pos_set[step_i]  # note: here the positions are exchanged
		action = reverse_action_set[step_i]
		old_tok = input_ids_old[ind]
		# word replacement (action: 0)
		if action == 0:
			prob_mask = bert_scorer.mask_score(input_ids_tmp, ind, mode=0)
			input_ids_tmp[ind] = old_tok
			proposal_prob_reverse *= prob_mask[old_tok]  # Q(x|x')
			if sim is not None:
				proposal_prob_reverse *= similarity(input_ids_tmp, input_original)
			if config.mode == 'sentiment' and proposal_sentiment_score:
				proposal_prob_reverse *= get_sentiment_score(input_ids_tmp, sentiment)

		# word insertion(action:1)
		if action == 1:
			prob_mask = bert_scorer.mask_score(input_ids_tmp, ind, mode=0)
			input_ids_tmp[ind] = old_tok
			proposal_prob_reverse *= prob_mask[old_tok]  # Q(x|x')
			if sim is not None:
				proposal_prob_reverse *= similarity(input_ids_tmp, input_original)
			if config.mode == 'sentiment' and proposal_sentiment_score:
				proposal_prob_reverse *= get_sentiment_score(input_ids_tmp, sentiment)

		# word deletion(action: 2)
		if action == 2:
			input_ids_tmp = input_ids_tmp  # already deleted
			proposal_prob_reverse *= 1.0  # Q(x|x')
	return proposal_prob_reverse, input_ids_tmp


def main():
	if os.path.exists(config.use_output_path):
		os.system('rm ' + config.use_output_path)
	with open(config.use_output_path, 'a') as g:
		g.write(str(config) + '\n\n')
	sim=config.sim
	# sta_vec=list(np.zeros([config.num_steps-1]))
	config.shuffle=False
	#original sentence input
	use_data = dataset_str(config.use_data_path)
	config.batch_size=1
	step_size = config.step_size

	start_time = time.time()
	proposal_cnt = 0
	accept_cnt = 0
	all_samples = []
	all_acc_samples = []
	all_chosen_samples = []
	for sen_id in range(use_data.length):
		sent_ids = use_data.token_ids[sen_id]
		keys = use_data.keys[sen_id]
		searcher = ConstraintSearch(keys)
		sequence_length = len(sent_ids)
		#generate for each sentence
		sta_vec = np.zeros(sequence_length)
		input_ids = np.array(sent_ids)
		input_original = use_data.tokens[sen_id]
		prev_inds = []
		old_prob = def_sent_scorer(tokenizer.decode(input_ids))
		old_prob_pen = penalty_constraint(searcher.count_unsafisfied_constraint(searcher.sent2tag(input_ids)))
		if config.mode == 'sentiment':
			old_prob *= get_sentiment_score(input_ids, sentiment)
		if sim != None:
			old_prob *= similarity(input_ids, input_original, sta_vec)

		outputs = []
		output_p = []
		for iter in range(config.sample_time):
			pos_set = np.array(get_sample_positions(sequence_length, prev_inds, step_size))
			prev_inds = pos_set
			proposal_cnt += 1

			search_cands, constr_num = searcher.search_template(input_ids, pos_set)
			group_prob = 1.0
			new_prob_pen = penalty_constraint(constr_num)
			original_temp = searcher.sent2tag(input_ids)
			original_constr_num = searcher.count_unsafisfied_constraint(original_temp)
			input_ids_old = np.array(input_ids)
			if len(search_cands) == 0:
				print('No candidate satisfies constraints. Continue.', pos_set)
			else:
				candidates = []
				candidate_probs = []
				for cand_template, action_set in search_cands:
					masked_sent, adjusted_pos_set = mask_sentence(input_ids, pos_set, action_set)
					proposal_prob, input_ids_tmp = eval_template(searcher, input_original, cand_template, masked_sent,
					                                             adjusted_pos_set, action_set, sim=None)
					input_text_tmp = tokenizer.decode(input_ids_tmp)
					new_prob = def_sent_scorer(input_text_tmp)
					if sim != None:
						sim_constr = similarity(input_ids_tmp, input_original, sta_vec)
						new_prob *= sim_constr
					if config.mode == 'sentiment':
						new_prob *= get_sentiment_score(input_text_tmp, sentiment)
					candidates.append((input_ids_tmp, proposal_prob, cand_template, action_set, adjusted_pos_set))
					candidate_probs.append(new_prob)

				candidate_probs_norm = normalize(np.array(candidate_probs))
				cand_idx = sample_from_candidate(np.array(candidate_probs_norm))
				input_ids_tmp, proposal_prob, cand_template, action_set, adjusted_pos_set = candidates[cand_idx]
				new_prob = candidate_probs[cand_idx]
				input_ids_new = np.array(input_ids_tmp)
				new_pos_set = np.array(adjusted_pos_set)
				print(cand_template)
				print(tokenizer.decode(input_ids_new).encode('utf8', errors='ignore'))

				# evaluate reverse proposal
				reverse_action_set = get_reverse_action_set(action_set)
				reverse_search_cands, reverse_min_constr_num, = searcher.search_template(input_ids_new, new_pos_set,
				                                                    prune=False)
				reverse_group_prob = penalty_constraint(original_constr_num - reverse_min_constr_num)
				reverse_search_cands_pruned = [(x[0], x[2]) for x in reverse_search_cands if x[1] == original_constr_num]

				# check reverse search cand
				reverse_search_cand_str = [','.join(x[0]) for x in reverse_search_cands]
				original_temp_str = ','.join(original_temp)
				if original_temp_str not in reverse_search_cand_str:
					print('Warning', original_temp, cand_template, pos_set, action_set, new_pos_set)
				if len(reverse_search_cands_pruned) == 0:
					print('Warning')
					reverse_search_cands_pruned = [original_temp]


				# evaluate reverse_candidate_probs_norm
				reverse_cand_idx = -1
				reverse_candidate_probs = []
				for c_idx, (reverse_cand_template, r_action_set) in enumerate(reverse_search_cands_pruned):
					if ','.join(reverse_cand_template) == original_temp_str:
						reverse_candidate_probs.append(old_prob)
						reverse_cand_idx = c_idx
					else:
						masked_sent, new_adjusted_pos_set = mask_sentence(input_ids_new, new_pos_set, r_action_set)
						_, r_input_ids_tmp = eval_template(searcher, input_original, reverse_cand_template, masked_sent,
						                                             new_adjusted_pos_set, r_action_set, sim=None)
						r_input_text_tmp = tokenizer.decode(r_input_ids_tmp)
						r_new_prob = def_sent_scorer(r_input_text_tmp)
						if sim != None:
							sim_constr = similarity(r_input_ids_tmp, input_original, sta_vec)
							r_new_prob *= sim_constr
						if config.mode == 'sentiment':
							r_new_prob *= get_sentiment_score(r_input_text_tmp, sentiment)
						# candidates.append((input_ids_tmp, proposal_prob))
						reverse_candidate_probs.append(r_new_prob)
				reverse_candidate_probs_norm = normalize(np.array(reverse_candidate_probs))

				# evaluate proposal_prob_reverse
				r_masked_sent, pos_set_ = mask_sentence(input_ids_new, new_pos_set, reverse_action_set)
				assert (pos_set == pos_set_).all()
				proposal_prob_reverse, input_ids_tmp_0 = \
					eval_reverse_proposal(input_original, r_masked_sent, input_ids_old, pos_set, reverse_action_set, sim=None)

				if (input_ids_tmp_0 != input_ids_old).any():
					print('Warning, ', input_ids_old, input_ids_new, input_ids_tmp_0)
				assert (input_ids_tmp_0 == input_ids_old).all()

				# decide acceptance
				sequence_length_new = len(input_ids_new)
				input_text_new = tokenizer.decode(input_ids_new)
				if proposal_prob == 0.0 or old_prob == 0.0:
					alpha_star = 1.0
				else:
					alpha_star = (comb(sequence_length_new, 3) * proposal_prob_reverse * reverse_group_prob *
					              reverse_candidate_probs_norm[reverse_cand_idx] * new_prob * new_prob_pen) / \
					             (comb(sequence_length, 3) * proposal_prob  * group_prob *
					              candidate_probs_norm[cand_idx] * old_prob * old_prob_pen)
				alpha = min(1, alpha_star)


				all_samples.append([input_text_new,
				                    new_prob*new_prob_pen,
				                    new_prob,
				                    constr_num,
					                bert_scorer.sent_score(input_ids_new, log_prob=True),
					                gpt2_scorer.sent_score(input_text_new, ppl=True)])
				if tokenizer.decode(input_ids_new) not in output_p:
					outputs.append(all_samples[-1])
				if outputs != []:
					output_p.append(outputs[-1][0])
				print(alpha, old_prob, proposal_prob, new_prob, new_prob*new_prob_pen, proposal_prob_reverse)
				if choose_action([alpha, 1 - alpha]) == 0 and (
						new_prob > old_prob * config.threshold or just_acc() == 0):
					if tokenizer.decode(input_ids_new) != tokenizer.decode(input_ids):
						accept_cnt += 1
						print('Accept')
						all_acc_samples.append(all_samples[-1])
					input_ids = input_ids_new
					sequence_length = sequence_length_new
					assert sequence_length == len(input_ids)
					old_prob = new_prob
				print('')


		# choose output from samples
		for num in range(config.min_length, 0, -1):
			outputss = [x for x in outputs if len(x[0].split()) >= num]
			print(num, outputss)
			if outputss != []:
				break
		if outputss == []:
			outputss.append([tokenizer.decode(input_ids), 0])
		outputss = sorted(outputss, key=lambda x: x[1])[::-1]
		with open(config.use_output_path, 'a') as g:
			g.write(outputss[0][0] + '\t' + str(outputss[0][1]) + '\n')
		all_chosen_samples.append(outputss[0])

		print('Sentence %d, used time %.2f\n' % (sen_id, time.time()-start_time))
	print(proposal_cnt, accept_cnt, float(accept_cnt/proposal_cnt))

	print("All samples:")
	all_samples_ = list(zip(*all_samples))
	for metric in all_samples_[1:]:
		print(np.mean(np.array(metric)))

	print("All accepted samples:")
	all_samples_ = list(zip(*all_acc_samples))
	for metric in all_samples_[1:]:
		print(np.mean(np.array(metric)))

	print("All chosen samples:")
	all_samples_ = list(zip(*all_chosen_samples))
	for metric in all_samples_[1:]:
		print(np.mean(np.array(metric)))

	with open(config.use_output_path + '-result.csv', 'w', newline='') as f:
		csv_writer = csv.writer(f, delimiter='\t')
		csv_writer.writerow(['Sentence', 'Prob_sim', 'Constraint_num', 'Log_prob', 'PPL'])
		csv_writer.writerows(all_samples)


if __name__ == "__main__":
	main()
