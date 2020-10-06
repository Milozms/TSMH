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
import csv
config=config()
import os 
# os.environ['CUDA_VISIBLE_DEVICES']=config.GPU
np.random.seed(config.seed)

from utils import choose_action, similarity, similarity_batch, normalize, sample_from_candidate, just_acc, \
	get_sample_positions, mask_sentence, generate_candidate_input_with_mask, MASK_IDX, PAD_IDX, bert_scorer, \
	gpt2_scorer, tokenizer, ConstraintSearch, penalty_constraint

def_sent_scorer = bert_scorer.sent_score

def main():
	if os.path.exists(config.use_output_path):
		os.system('rm ' + config.use_output_path)
	with open(config.use_output_path, 'a') as g:
		g.write(str(config) + '\n\n')
	# for item in config.record_time:
	# 	if os.path.exists(config.use_output_path + str(item)):
	# 		os.system('rm ' + config.use_output_path + str(item))
	#CGMH sampling for paraphrase
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
		old_prob *= penalty_constraint(searcher.count_unsafisfied_constraint(searcher.sent2tag(input_ids)))
		if sim != None:
			old_prob *= similarity(input_ids, input_original, sta_vec)

		outputs = []
		output_p = []
		for iter in range(config.sample_time):
			# if iter in config.record_time:
			# 	with open(config.use_output_path, 'a', encoding='utf-8') as g:
			# 		g.write(bert_scorer.tokenizer.decode(input_ids)+'\n')
			# print(bert_scorer.tokenizer.decode(input_ids).encode('utf8', errors='ignore'))
			pos_set = get_sample_positions(sequence_length, prev_inds, step_size)
			action_set = [choose_action(config.action_prob) for i in range(len(pos_set))]
			# if not check_constraint(input_ids):
			# 	if 0 not in pos_set:
			# 		pos_set[-1] = 0
			keep_non = config.keep_non
			masked_sent, adjusted_pos_set = mask_sentence(input_ids, pos_set, action_set)
			prev_inds = pos_set

			proposal_prob = 1.0  # Q(x'|x)
			proposal_prob_reverse = 1.0  # Q(x|x')
			input_ids_tmp = np.array(masked_sent)  # copy
			sequence_length_tmp = sequence_length

			for step_i in range(len(pos_set)):

				ind = adjusted_pos_set[step_i]
				ind_old = pos_set[step_i]
				action = action_set[step_i]
				if config.restrict_constr:
					if step_i == len(pos_set) - 1:
						use_constr = True
					else:
						use_constr = False
				else:
					use_constr = True
				#word replacement (action: 0)
				if action==0:
					prob_mask = bert_scorer.mask_score(input_ids_tmp, ind, mode=0)
					input_candidate, prob_candidate, reverse_candidate_idx, _ = \
						generate_candidate_input_with_mask(input_ids_tmp, sequence_length_tmp, ind, prob_mask, config.search_size,
						                                   old_tok=input_ids[ind_old], mode=action)
					if sim is not None and use_constr:
						similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec)
						prob_candidate=prob_candidate*similarity_candidate
					prob_candidate_norm=normalize(prob_candidate)
					prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
					input_ids_tmp = input_candidate[prob_candidate_ind]  # changed
					proposal_prob *= prob_candidate_norm[prob_candidate_ind]                      # Q(x'|x)
					proposal_prob_reverse *= prob_candidate_norm[reverse_candidate_idx]           # Q(x|x')
					sequence_length_tmp += 0
					print('action:0', prob_candidate_norm[prob_candidate_ind], prob_candidate_norm[reverse_candidate_idx])

				#word insertion(action:1)
				if action==1:
					prob_mask = bert_scorer.mask_score(input_ids_tmp, ind, mode=0)

					input_candidate, prob_candidate, reverse_candidate_idx, non_idx = \
						generate_candidate_input_with_mask(input_ids_tmp, sequence_length_tmp, ind, prob_mask, config.search_size,
						                                   mode=action, old_tok=input_ids[ind_old], keep_non=keep_non)

					if sim is not None and use_constr:
						similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec)
						prob_candidate=prob_candidate*similarity_candidate
					prob_candidate_norm=normalize(prob_candidate)
					prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
					input_ids_tmp = input_candidate[prob_candidate_ind]
					if prob_candidate_ind == non_idx:
						if input_ids_tmp[-1] == PAD_IDX:
							input_ids_tmp = input_ids_tmp[:-1]
						print('action:1 insert non', 1.0, 1.0)
					else:
						proposal_prob *= prob_candidate_norm[prob_candidate_ind]  # Q(x'|x)
						proposal_prob_reverse *= 1.0                              # Q(x|x'), reverse action is deleting
						sequence_length_tmp += 1
						print('action:1', prob_candidate_norm[prob_candidate_ind], 1.0)

				#word deletion(action: 2)
				if action==2:
					input_ids_for_del = np.concatenate([input_ids_tmp[:ind], [MASK_IDX], input_ids_tmp[ind:]])
					if keep_non:
						non_cand = np.array(input_ids_for_del)
						non_cand[ind] = input_ids[ind_old]
						input_candidate = np.array([input_ids_tmp, non_cand])
						prob_candidate = np.array([bert_scorer.sent_score(x) for x in input_candidate])
						non_idx = 1
						if sim is not None and use_constr:
							similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec)
							prob_candidate=prob_candidate*similarity_candidate
						prob_candidate_norm=normalize(prob_candidate)
						prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
						input_ids_tmp = input_candidate[prob_candidate_ind]
					else:
						non_idx = -1
						prob_candidate_ind = 0
						input_ids_tmp = input_ids_tmp  # already deleted

					if prob_candidate_ind == non_idx:
						print('action:2 delete non', 1.0, 1.0)
					else:
						# add mask, for evaluating reverse probability
						prob_mask = bert_scorer.mask_score(input_ids_for_del, ind, mode=0)
						input_candidate, prob_candidate, reverse_candidate_idx, _ = \
							generate_candidate_input_with_mask(input_ids_for_del, sequence_length_tmp, ind, prob_mask,
							                                   config.search_size, mode=0, old_tok=input_ids[ind_old])

						if sim!=None:
							similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec)
							prob_candidate=prob_candidate*similarity_candidate
						prob_candidate_norm = normalize(prob_candidate)

						proposal_prob *= 1.0                    # Q(x'|x)
						proposal_prob_reverse *= prob_candidate_norm[reverse_candidate_idx]   # Q(x|x'), reverse action is inserting
						sequence_length_tmp -= 1

						print('action:2', 1.0, prob_candidate_norm[reverse_candidate_idx])

			new_prob = def_sent_scorer(tokenizer.decode(input_ids_tmp))
			new_prob *= penalty_constraint(searcher.count_unsafisfied_constraint(searcher.sent2tag(input_ids_tmp)))
			if sim != None:
				sim_constr = similarity(input_ids_tmp, input_original, sta_vec)
				new_prob *= sim_constr
			input_text_tmp = tokenizer.decode(input_ids_tmp)
			all_samples.append([input_text_tmp,
			                    new_prob,
			                    searcher.count_unsafisfied_constraint(searcher.sent2tag(input_ids_tmp)),
			                    bert_scorer.sent_score(input_ids_tmp, log_prob=True),
			                    gpt2_scorer.sent_score(input_text_tmp, ppl=True)])
			if tokenizer.decode(input_ids_tmp) not in output_p:
				outputs.append(all_samples[-1])
			if outputs != []:
				output_p.append(outputs[-1][0])
			if proposal_prob == 0.0 or old_prob == 0.0:
				alpha_star = 1.0
			else:
				alpha_star = (proposal_prob_reverse * new_prob) / (proposal_prob * old_prob)
			alpha = min(1, alpha_star)
			print(tokenizer.decode(input_ids_tmp).encode('utf8', errors='ignore'))
			print(alpha, old_prob, proposal_prob, new_prob, proposal_prob_reverse)
			proposal_cnt += 1
			if choose_action([alpha, 1 - alpha]) == 0 and (
					new_prob > old_prob * config.threshold or just_acc() == 0):
				if tokenizer.decode(input_ids_tmp) != tokenizer.decode(input_ids):
					accept_cnt += 1
					print('Accept')
					all_acc_samples.append(all_samples[-1])
				input_ids = input_ids_tmp
				sequence_length = sequence_length_tmp
				old_prob = new_prob


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
