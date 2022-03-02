import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, Vocab, GloVe
from utils import load_vocab
from pathlib import Path

class EntailmentDataset(data.Dataset):
	def __init__(self, path, premise_field, hypothesis_field, label_field, **kwargs):
		if hypothesis_field == None:
			fields = [("premise", premise_field),("label", label_field)]
		else:
			fields = [("premise", premise_field), ("hypothesis", hypothesis_field),("label", label_field)]
		examples = []
		with open(path) as rf:
			for line in rf:
				try:
					line_split = line.strip().split("\t") #space_sep_sequence \t label \t distance
					assert len(line_split) > 2
				except AssertionError:
					line_split == line.strip().split(" ") #space_sep_sequence \t label \t distance
					print(line_split)
					assert len(line_split) > 2
				label = "entailed" if line_split[2] == "1" else "not entailed"
				if hypothesis_field == None:
					p = line_split[0] + " SEP " + line_split[1]
					examples.append(data.Example.fromlist([p, label], fields))
				else:
					p = line_split[0]
					h = line_split[1]
					examples.append(data.Example.fromlist([p,h, label], fields))
		super().__init__(examples, fields, **kwargs)

	@staticmethod
	def sort_key(ex):
		try:
			return len(ex.premise) + len(ex.hypothesis)
		except AttributeError:
			return len(ex.premise)

	@classmethod
	def splits(self,premise_field,hypothesis_field,  label_field, path,
			   train, val, test, **kwargs):
		return super().splits(path,
			premise_field=premise_field, hypothesis_field=hypothesis_field,
			label_field=label_field,
			train=train, validation=val, test=test, **kwargs)


class MyLanguageModelingDataset(data.Dataset):
	def __init__(self, path, text_field, target_field, example_idx_field, **kwargs):
		fields = [("text", text_field), ("target",target_field), ("example_idx", example_idx_field)]
		examples = []
		with open(path) as rf:
			for line_idx, line in enumerate(rf):
				text = line.strip()
				target = " ".join(text.split(" ")[1:]) #Breaks if tokenization changes
				examples.append(data.Example.fromlist([text, target, line_idx], fields))
		super().__init__(examples, fields, **kwargs)

	@staticmethod
	def sort_key(ex): return len(ex.text)

	@classmethod
	def splits(self,text_field,target_field,example_idx_field,path,
			   train, val, test, **kwargs):
		return super().splits(path,
			text_field=text_field, target_field=target_field, example_idx_field=example_idx_field,
			train=train, validation=val, test=test, **kwargs)



def load_dataset(test_sen=None, args = {}):
	if args.get("finetuning") ==True and args.get("dir",None) == args.get("finetuning_dir", True) and args.get("ph_as_one", False ==True):
		max_seq_len = 2 * args.get("max_seq_len", 60)
	else:
		max_seq_len = args.get("max_seq_len", 60)
	print("max seq len", max_seq_len)
	torch.manual_seed(args["seed"])
	special_tokens = ["PAD", "EOS", "&", "|", "&2", "|2", ">", "(",")", "~", "~2", "<grounding>", "True", "False", "UNK", "SEP"]
	tokenize = lambda x: x.split()
	TEXT = data.Field(sequential=True, tokenize=tokenize, pad_token="PAD", eos_token="EOS",
					  include_lengths=True, batch_first=True, fix_length=max_seq_len)
	# LABEL = data.LabelField(dtype=torch.FloatTensor)
	LABEL = data.LabelField(dtype=torch.float32)
	Path("dummy.txt").touch()
	if args.get("pretraining", False) == True:
		dataset = MyLanguageModelingDataset(path="dummy.txt", text_field=TEXT,target_field=TEXT, example_idx_field=LABEL)
		train_data, valid_data = dataset.splits(path=args["dir"],text_field=TEXT, target_field=TEXT, example_idx_field=LABEL, train="train.txt", val="validate.txt",test=None)
		if args.get("probe", False) == True or args.get("minimalpairs", False) == True:
			LABEL.build_vocab(valid_data)
		else:
			LABEL.build_vocab(train_data)
	elif args.get("snli", False) == True:
		train_data, valid_data, test_data = datasets.SNLI.splits(TEXT, LABEL)
		TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
		args["predefined_vocab"] = None
	elif args.get("mnli", False) == True:
		train_data, valid_data, test_data = datasets.MultiNLI.splits(TEXT, LABEL)
		TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
		args["predefined_vocab"] = None
	elif args.get("sst", False) == True:
		train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL)
		args["predefined_vocab"] = None
		args["ph_as_one"] = True
		new_ex = []
		for i in range(len(train_data.examples)):
			if train_data.examples[i].label != "neutral":
				new_ex.append(train_data.examples[i])
		train_data.examples = new_ex
		new_ex = []
		for i in range(len(valid_data.examples)):
			if valid_data.examples[i].label != "neutral":
				new_ex.append(valid_data.examples[i])
		valid_data.examples = new_ex
	else:
		if args.get("ph_as_one", False) == False:
			hyp = TEXT
		else:
			hyp = None
		dataset = EntailmentDataset(path="dummy.txt", premise_field=TEXT, hypothesis_field=TEXT,label_field=LABEL)
		train_data, valid_data = dataset.splits(path=args["dir"],premise_field=TEXT, hypothesis_field=hyp,
		label_field=LABEL, train="train.txt", val="validate.txt", test=None)
	#args["predefined_vocab"] =
	word_embeddings = None
	if args.get("snli", False) == True or args.get("mnli", False) == True or args.get("sst", False) == True:
		TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
		LABEL.build_vocab(train_data)
		word_embeddings = TEXT.vocab.vectors
	elif args.get("predefined_vocab", None) == None:
		#min_vocab_frequency = args.get("min_vocab_frequency", 1)
		#print("setting min vocab freq to {}".format(min_vocab_frequency))
		TEXT.build_vocab(train_data)#, min_freq = min_vocab_frequency)
		LABEL.build_vocab(train_data)#, min_freq =  min_vocab_frequency)
		#train_data.filter_examples(["text", "target"])
		#valid_data.filter_examples(["text", "target"])
	else:
		imported_vocab = {}
		itos = {}
		#for ct, word in enumerate(special_tokens):
		#	imported_vocab[word] = ct
		try:
			with open(args["predefined_vocab"]) as rf:
				for ct, line in enumerate(rf):
					idx, word = line.strip().split("\t")
					imported_vocab[word] = int(idx)
					itos[int(idx)] = word
		except ValueError:
			with open(args["predefined_vocab"]) as rf:
				for ct, line in enumerate(rf):
					word, idx = line.strip().split("\t")
					if word == "and":
						print(word, idx)
					imported_vocab[word] = int(idx)
					itos[int(idx)] = word
		print("Todo: you changed vocab import behavior. Might mess up on symbols")
		#HACK! TO ENSURE VOCAB ORDERING.
		#The vocab thingie like... works by freq. So we need to sort things in reverse freq
		#for word in imported_vocab.keys():
			#if word not in special_tokens:
			#imported_vocab[word] = ct+100 - imported_vocab[word]
		#imported_vocab["<unk>"] = 1
		#vocab = Vocab(imported_vocab, specials=["<unk>", "PAD", "EOS"])#special_tokens)
		imported_vocab, itos_list = load_vocab(args["predefined_vocab"])

		vocab = Vocab(imported_vocab, specials=["<unk>", "PAD", "EOS"])
		vocab.stoi = imported_vocab
		vocab.itos = itos_list
		TEXT.vocab = vocab
		print(TEXT.vocab["and"])

	test_data = valid_data
	if args.get("pretraining", False) == False and args.get("sst", False) == False:
		LABEL.build_vocab(train_data)
		print ("Label Length: " + str(len(LABEL.vocab)))
		if args.get("ph_as_one", False) == False:
			sort_key = lambda x: len(x.premise) + len(x.hypothesis)
		else:
			sort_key = lambda x: len(x.premise)
	else:
		sort_key = lambda x: len(x.text)
	print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
	if "PAD" in TEXT.vocab.stoi.keys():
		pad_idx = TEXT.vocab.stoi["PAD"]
	elif "<pad>" in TEXT.vocab.stoi.keys():
		pad_idx = TEXT.vocab.stoi["<pad>"]
	else:
		print("no padding token?")
		sys.exit(1)
	if pad_idx != 0:
		print("WARNING: Padding idx is {}, not 0.".format(pad_idx))

	#print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
	train_iter, val_iter, test_iter = data.BucketIterator.splits(
		(train_data, valid_data, test_data), batch_size=args["batch_size"],
		sort_key=sort_key, repeat=False, shuffle=True)

	vocab_size = len(TEXT.vocab)

	return TEXT, vocab_size, train_iter, val_iter, test_iter, word_embeddings, len(train_data), pad_idx
