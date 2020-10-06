import numpy as np
import sys
import os
import math
from utils import bert_scorer, gpt2_scorer, tokenizer, config, ConstraintSearch, get_sentiment_score
from reader import dataset_str

# fname = './output/output-yelp'
fname = './output/yelp-20200424/output-yelp-base-full.txt'
datacnt = 18
all_chosen_samples = []
use_data = dataset_str(config.use_data_path)

if os.path.isdir(fname):
	lines = []
	# with open('./output/yelp-20200424/output-yelp.txt', 'r') as f:
	# 	lines = f.readlines()[2:]
	for idx in range(datacnt):
		with open(os.path.join(fname, 'output-%d.txt' % idx), 'r') as f:
			this_lines = f.readlines()[2:]
			lines.extend(this_lines)
else:
	with open(fname, 'r') as f:
		lines = f.readlines()[2:]

for idx, line in enumerate(lines):
	line = line.strip().split('\t')
	keys = use_data.keys[idx]
	searcher = ConstraintSearch(keys)
	sent = line[0]
	new_prob = float(line[1])
	if new_prob == 1:
		new_prob = 0.0
	input_ids_tmp = tokenizer.encode(sent, add_special_tokens=False)
	input_text_tmp = sent
	num_constr = searcher.count_unsafisfied_constraint(searcher.sent2tag(input_ids_tmp))
	print(num_constr)
	log_prob, sent_len = gpt2_scorer.sent_score(input_text_tmp, log_prob=True)
	all_chosen_samples.append([input_text_tmp,
	                    new_prob,
	                    num_constr,
	                           log_prob,
	                           log_prob / sent_len,
	                           math.exp(log_prob),
	                           get_sentiment_score(input_text_tmp, 'positive')
	                    # bert_scorer.sent_score(input_ids_tmp, log_prob=True),
	                    # gpt2_scorer.sent_score(input_text_tmp, ppl=True)
	])

print("All chosen samples:")
all_samples_ = list(zip(*all_chosen_samples))
for metric in all_samples_[1:]:
	print(np.mean(np.array(metric)))

all_num_constr = all_samples_[2]
zero_cnt = all_num_constr.count(0)
one_cnt = all_num_constr.count(1)
print(zero_cnt)
print(zero_cnt/len(all_num_constr))
# print(one_cnt+zero_cnt)
# print((one_cnt+zero_cnt)/len(all_num_constr))


# senti_scores = np.array(all_samples_[-1])
# print(np.mean(senti_scores[:150]))
# print(np.mean(senti_scores[150:]))
for samples in [all_chosen_samples[:150], all_chosen_samples[150:]]:
	all_samples_ = list(zip(*samples))
	for metric in all_samples_[1:]:
		print(np.mean(np.array(metric)))

	all_num_constr = all_samples_[2]
	zero_cnt = all_num_constr.count(0)
	one_cnt = all_num_constr.count(1)
	print(zero_cnt)
	print(zero_cnt/len(all_num_constr))