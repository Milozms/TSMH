import argparse

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def config():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use_data_path', type=str, default='./input/input-yelp-keys.txt')
	# parser.add_argument('--use_data_path', type=str, default='./input/input-imp.txt')
	parser.add_argument('--dict_path', type=str, default='../data/quora/dict.pkl')
	parser.add_argument('--emb_path', type=str, default='../data/quora/emb.pkl')
	parser.add_argument('--skipthoughts_path', type=str, default='../skip_thought')
	parser.add_argument('--use_output_path', type=str, default='./output/output.txt')
	parser.add_argument('--dict_size', type=int, default=30000)
	parser.add_argument('--vocab_size', type=str, default=30003)
	parser.add_argument('--num_steps', type=int, default=100)
	parser.add_argument('--min_length', type=int, default=7)
	parser.add_argument('--max_key', type=int, default=3)
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('--sample_time', type=int, default=101)
	parser.add_argument('--step_size', type=int, default=3)
	parser.add_argument('--search_size', type=int, default=100)
	parser.add_argument('--action_prob', nargs='+', type=float, default=[0.3,0.25,0.25,0.2])
	parser.add_argument('--threshold', type=float, default=0.1)
	parser.add_argument('--no_aux_weight', type=float, default=0.0)
	parser.add_argument('--aux_start_weight', type=float, default=0.0)
	parser.add_argument('--just_acc_rate', type=float, default=0.0)
	parser.add_argument('--pen_num_constr', type=str2bool, default=True)
	parser.add_argument('--pen_num_constr_base', type=float, default=1e-10)
	parser.add_argument('--keep_non', type=str2bool, default=False)
	parser.add_argument('--sim', type=str, default=None)
	parser.add_argument('--parser_tag', type=str2bool, default=True)
	parser.add_argument('--sq_weight', type=float, default=1e-10)
	parser.add_argument('--restrict_constr', type=str2bool, default=True)
	parser.add_argument('--mode', type=str, default='sentiment')
	parser.add_argument('--proposal_sentiment_score', type=str2bool, default=False)  # warning: suggest using false here (for search)
	parser.add_argument('--bert_path', type=str, default='../bert/models/bert')
	parser.add_argument('--gpt2_path', type=str, default='../bert/models/gpt2')
	parser.add_argument('--verb_path', type=str, default='../bert/models/verbs_ids.txt')
	parser.add_argument('--participle_path', type=str, default='../bert/models/participle_ids.txt')
	args = parser.parse_args()
	return args

'''
class config(object):
	def __init__(self):
		self.use_data_path = './input/input.txt'
		self.dict_path='../data/quora/dict.pkl'                                                  #dictionary path
		# self.pos_path='../POS/english-models'                                    #path for pos tagger
		self.emb_path='../data/quora/emb.pkl'                                    #word embedding path, used when config.sim=='word_max' or config.sim=='combine'
		self.skipthoughts_path='../skip_thought'                                  #path of skipthoughts, used when config.sim=='skipthoughts' or config.sim=='combine'
		self.dict_size=30000
		self.vocab_size=30003
		# self.use_fp16=False
		# self.shuffle=False
		self.keep_non = False

		self.num_steps=100

		self.min_length = 7
		# self.GPU=''
		self.sample_time=101
		# self.record_time=[5,10,15,20,25,30,35,40,45,50]
		# for i in range(len(self.record_time)):
		#     self.record_time[i]*=2
		# self.record_time = list(range(0,self.sample_time))
		self.step_size = 1
		self.search_size=100
		self.use_output_path='./output/output.txt'         #output path


		self.action_prob=[0.3,0.25,0.25,0.2]                                              #the prior of 4 actions
		self.threshold=0.1
		self.sim='word_max'
		# self.keyword_pos=True
		# self.rare_since=30000
		self.just_acc_rate=0.0
		# self.max_key=3
		# self.max_key_rate=0.5

'''