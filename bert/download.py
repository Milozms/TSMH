from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel

bert_name = 'bert-base-uncased'
gpt2_name = 'gpt2'
bert_dir = './models/bert'
gpt2_dir = './models/gpt2'

bert_model = BertForMaskedLM.from_pretrained(bert_name)
bert_tokenizer = BertTokenizer.from_pretrained(bert_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name)

bert_model.save_pretrained(bert_dir)
bert_tokenizer.save_pretrained(bert_dir)
gpt2_model.save_pretrained(gpt2_dir)
gpt2_tokenizer.save_pretrained(gpt2_dir)