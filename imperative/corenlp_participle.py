from nltk.corpus import wordnet as wn
import corenlp
import os
from utils import tokenizer
from tqdm import tqdm
import json

os.environ['CORENLP_HOME'] = '~/stanford-corenlp-full'

def check_word_pos(word, pos):
	return len(wn.synsets(word, pos=pos)) > 0


def find_all_pos_words(pos):
	results = []
	verbs = []
	with corenlp.CoreNLPClient(annotators="tokenize ssplit pos".split()) as client:
		for idx in tqdm(range(tokenizer.vocab_size)):
			word = tokenizer._convert_id_to_token(idx)
			if check_word_pos(word, pos):
				ann = client.annotate(word)
				pos_all = [[token.pos for token in sent.token] for sent in ann.sentence]
				pos_list = []
				for pos_sent in pos_all:
					pos_list.extend(pos_sent)
				if len(pos_list) == 1 and pos_list[0] in ['VBN', 'VVN', 'VHN']:
					results.append(idx)
					verbs.append(word)

	# print(results)
	print(len(results))
	return results, verbs

if __name__ == '__main__':
	results, verbs = find_all_pos_words(wn.VERB)
	with open('../bert/models/past_ids.txt', 'w') as fout:
		json.dump(results, fout)
	with open('./past.txt', 'w') as fout:
		for v in verbs:
			fout.write(v+'\n')