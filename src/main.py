import os
import time
import load_data
import torch
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy

from collections import defaultdict

from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import sys
from models import LSTMClassifier, EntailmentModule, TransformerModel, CBOWClassifier
import argparse
from utils import create_config
from toma import toma

from load_data import EntailmentDataset, MyLanguageModelingDataset
from torchtext import data, vocab

import matplotlib.pyplot as plt

INIT_BATCHSIZE=512

"""
Contains every function for training and testing the langauge models.
"""

def cossim(a, b):
	num = np.dot(a, b)
	denom = np.linalg.norm(a) * np.linalg.norm(b)
	return num * 1.0 / max(1e-10, denom)

def probe_pretraining_model(model, val_iter, args={}, write=None, vocab=None, max_iter = 30):
	"""
	Function for sampling from the language model. Batch size always equals one.

	Will sample from LM until EOS token is reached or until it generates max_iter tokens.
	"""
	if torch.cuda.is_available():
		model.cuda()
	stoi = args["TEXT"].vocab.stoi
	itos = args["TEXT"].vocab.itos

	if args["sequence_model_name"] == "lstm":
		weights = model.word_embeddings.weight.cpu().data.numpy()
	else: #model is transformer
		weights = model.encoder.weight.cpu().data.numpy()

	batchsize = args["batchsize"]
	loss_fn = args["loss_fn"]
	total_epoch_loss = 0
	total_epoch_ppl = 0
	model.eval()
	output = {}
	sequence_pairs = defaultdict(list)
	with torch.no_grad():
		for q in range(0, 1): #so we sample multiple times
			for idx, batch in enumerate(val_iter):
				#Batch: [1, MAX_SEQ_LEN]
				n_iter = 0
				most_recently_generated = ""
				p = batch.text[0]
				length = batch.text[1]

				#REMOVE EOS TOKEN!
				p = p[:, :length - 1] #removes the last token of the string
				length -= 1 #now our sequence is one shorter
				assert p[0,-1] != stoi["EOS"] #make sure the new last token isn't EOS
				assert p[0,-1] != stoi["PAD"] #make sure the new last token isn't PAD

				orig_sentence = deepcopy(p) #for comparison

				while n_iter < max_iter and most_recently_generated != "EOS":
					if torch.cuda.is_available():
						p = p.cuda()
					n_iter += 1

					prediction = model(p, length) #shape = [1, len(p)]
					#now get the best prediction
					last_state_predictions = prediction[0, -1, :] #shape = [1, vocab_size]
					#if there were padding tokens, which there aren't, pack_padded_seq
					#propagates the last output forward thru paddings
					probs = F.softmax(last_state_predictions).cpu().squeeze(0).data.numpy() #shape = [1, vocab_size]
					new_label = np.random.choice(len(probs),p=probs) #shape = 1
					most_recently_generated = itos[new_label]

					seq = p.cpu().data.numpy()[0] #shape = [len(p)], dtype = nparr
					seq = seq.tolist()[:length] + [new_label] #len = len(p) + 1, dtype = list
					seq = torch.LongTensor(seq) #shape = [len(p)+1]
					p = seq.unsqueeze(0) #shape = [1, len(p) + 1]

					length = length + 1
				o_s = "orig:\t{}".format(" ".join([itos[int(x)] for x in orig_sentence.cpu().data.numpy()[0]]))
				g_s = "generated:\t{}\n".format(" ".join([itos[int(x)] for x in p.cpu().data.numpy()[0]]))
				sequence_pairs[o_s].append(g_s)
				#TODO: Save output.
		with open(write, "w+") as wf:
			for key in sequence_pairs.keys():
				for val in sequence_pairs[key]:
					wf.write("{}\t{}\n".format(key.split("\t")[-1], val.split("\t")[-1]))

def clip_gradient(model, clip_value):
	params = list(filter(lambda p: p.grad is not None, model.parameters()))
	for p in params:
		p.grad.data.clamp_(-clip_value, clip_value)

@toma.batch(initial_batchsize=INIT_BATCHSIZE)
def test(batchsize, args, predefined_vocab=None, model=None):

	args["batchsize"] = batchsize
	special_tokens = ["PAD", "EOS", "&", "|", "&2", "|2", ">", "(",")", "~", "<grounding>", "True", "False", "UNK", "SEP"]
	tokenize = lambda x: x.split()
	if args["sequence_model_name"] == "transformer":
		max_seq_len = 120
	elif args.get("ph_as_one", False ==True):
		max_seq_len = 140#2 * args.get("max_seq_len", 60)
	else:
		max_seq_len = 70#args.get("max_seq_len", 60)
	TEXT = data.Field(sequential=True, tokenize=tokenize, pad_token="PAD", eos_token="EOS",
					  include_lengths=True, batch_first=True, fix_length=max_seq_len)
	LABEL = data.LabelField(dtype=torch.float32)
	if args.get("ph_as_one", False) == False:
		hyp = TEXT
		sort_key=lambda x: len(x.premise) + len(x.hypothesis)
	else:
		hyp = None
		sort_key=lambda x: len(x.premise)
	if predefined_vocab is None:
		imported_vocab = {}
		for ct, word in enumerate(special_tokens):
			imported_vocab[word] = ct
		with open(args["predefined_vocab"]) as rf:
			try:
				#for ct, word in enumerate(special_tokens):
				#	imported_vocab[word] = ct
				for ct, line in enumerate(rf):
					idx,word = line.strip().split("\t")
					imported_vocab[word] = int(idx)
			except ValueError:
				for ct, line in enumerate(rf):
					word,idx = line.strip().split("\t")
					imported_vocab[word] = int(idx)

		if "PAD" in imported_vocab.keys():
			pad_idx = imported_vocab["PAD"]
		elif "<pad>" in imported_vocab.keys():
			pad_idx = imported_vocab["<pad>"]
		else:
			print("no padding token?")
			sys.exit(1)
		args["pad_idx"] = pad_idx
		predefined_vocab = vocab.Vocab(imported_vocab, specials=special_tokens)
	TEXT.vocab = predefined_vocab
	args["vocab_size"] = len(predefined_vocab)


	print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
	if model == None:
		model, args = load_model(args)
	if torch.cuda.is_available():
		model.cuda()
	scores = []
	test_path = os.path.join("test_results", args["experiment_label"], str(args["seed"]))
	if not os.path.exists(test_path):
		os.makedirs(test_path)
	print(test_path)
	output_path = os.path.join(test_path,"predictions")
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	for f in os.listdir(args["test_directory"]):
		if ".txt" in f and "test" in f:
			print(f)
			if args.get("finetuning", False) == False:
				#pretraining, this is probably kludged
				dataset = MyLanguageModelingDataset(path=os.path.join(args["test_directory"], f), text_field=TEXT, target_field=TEXT, example_idx_field=LABEL)
				sort_key = lambda x: len(x.text)
			else:
				dataset = EntailmentDataset(path=os.path.join(args["test_directory"], f), premise_field=TEXT, hypothesis_field=hyp,label_field=LABEL)
			LABEL.build_vocab(dataset)
			test_iter = data.BucketIterator(
				dataset, batch_size=args["batchsize"],
				sort_key=sort_key, repeat=False, shuffle=True)
			if args.get("finetuning", False) == True:
				test_loss, test_score = eval_model(model, test_iter, args, write=os.path.join(output_path, f), vocab=TEXT.vocab)
			else:
				test_loss, test_score = eval_model_pretraining(model, test_iter, args, write=os.path.join(output_path, f), vocab=TEXT.vocab)
			print(test_loss, test_score)
			scores.append([f, test_loss, test_score])

	with open(os.path.join(test_path, "scores.txt"), "w+") as wf:
		for name, l, a in scores:
			wf.write("{}\t{}\t{}\n".format(name, l, a))
	wf.close()

def train_model_pretraining(model, optim, train_iter, epoch, args={}):

	batchsize = args["batchsize"]
	loss_fn = args["loss_fn"]
	total_epoch_loss = 0
	total_epoch_ppl = 0
	if torch.cuda.is_available():
		model.cuda()

	steps = 0
	model.train()
	s = nn.Softmax(dim=0)
	for idx, batch in enumerate(train_iter):
		p = batch.text[0]
		lengths = batch.text[1]
		target = batch.target[0]
		target = torch.autograd.Variable(target).long()
		if torch.cuda.is_available():
			p = p.cuda()
			target = target.cuda()
		optim.zero_grad()
		prediction = model(p, lengths)
		prediction = prediction.reshape(prediction.shape[0]* prediction.shape[1],prediction.shape[2])
		target = target.reshape(target.shape[0] * target.shape[1])
		loss = loss_fn(prediction, target)
		perplexity = torch.exp(loss)
		loss.backward()
		clip_gradient(model, 1e-1)
		optim.step()
		steps += 1

		if steps % 100 == 0:
			print('Pretraining epoch: {}, Idx: {}, Training Loss:\
			 {}, Training Perplexity: {}'.format(epoch+1, idx+1, loss.item(), perplexity.item()))

		total_epoch_loss += loss.item()
		total_epoch_ppl += perplexity.item()

	return total_epoch_loss/len(train_iter), total_epoch_ppl/len(train_iter)



def train_model(model, optim, train_iter, epoch, args={}):

	batchsize = args["batchsize"]
	loss_fn = args["loss_fn"]
	total_epoch_loss = 0
	total_epoch_acc = 0
	if torch.cuda.is_available():
		model.cuda()

	steps = 0
	model.train()
	for idx, batch in enumerate(train_iter):
		if args.get("load_sample_dataset", None) == "sst":
			p = batch.text[0]
			pl = batch.text[1]
			h = None
			hl = None
		else:
			p = batch.premise[0]
			pl = batch.premise[1]
			if args.get("ph_as_one", False) == False:
				h = batch.hypothesis[0]
				hl = batch.hypothesis[1]
			else:
				h = None
				hl = None
		target = batch.label
		target = torch.autograd.Variable(target).long()
		if torch.cuda.is_available():
			p = p.cuda()
			if args.get("ph_as_one", False) == False:
				h = h.cuda()
			target = target.cuda()
		optim.zero_grad()
		prediction = model(p, pl,h, hl)
		loss = loss_fn(prediction, target)
		num_corrects = (torch.max(
			prediction, 1)[1].view(target.size()).data == target.data).float().sum()
		if args.get("load_sample_dataset", None) == "sst":
			acc = 100.0 * num_corrects/(len(batch) - sum(target == 2))
		else:
			acc = 100.0 * num_corrects/len(batch)
		loss.backward()
		clip_gradient(model, 1e-1)
		optim.step()
		steps += 1

		if steps % 100 == 0:
			print('Epoch: {}, Idx: {}, Training Loss:\
			 {}, Training Accuracy: {}'.format(epoch+1, idx+1, loss.item(), acc.item()))

		total_epoch_loss += loss.item()
		total_epoch_acc += acc.item()

	return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


def eval_model_pretraining(model, val_iter, args={}, write=None, vocab=None):
	batchsize = args["batchsize"]
	loss_fn = args["loss_fn"]
	total_epoch_loss = 0
	total_epoch_ppl = 0
	model.eval()
	output = {}
	with torch.no_grad():
		for idx, batch in enumerate(val_iter):
			p = batch.text[0]
			lengths = batch.text[1]
			target = batch.target[0]
			target = torch.autograd.Variable(target).long()
			if torch.cuda.is_available():
				p = p.cuda()
				target = target.cuda()
			prediction = model(p, lengths)
			prediction2 = prediction.reshape(prediction.shape[0]* prediction.shape[1],prediction.shape[2])
			target2 = target.reshape(target.shape[0] * target.shape[1])
			loss = loss_fn(prediction2, target2)
			perplexity = torch.exp(loss)
			total_epoch_loss += loss.item()
			total_epoch_ppl += perplexity.item()
			if write != None:
				for ct, b in enumerate(batch.text[0]):
					write_line = {}
					write_line["example_idx"] = int(batch.example_idx[ct])
					write_line["ppl"] = torch.exp(loss_fn(prediction[ct], target[ct])).item() #todo dbl check
					output[b] = write_line
	if write != None:
		with open(write, "w+") as wf:
			for key in output.keys():
				p = " ".join([vocab.itos[i.item()] for i in key if i.item() > 1])
				wf.write("{}\t{}\t{}\n".format(p,output[key]["example_idx"], output[key]["ppl"]))


	return total_epoch_loss/len(val_iter), total_epoch_ppl/len(val_iter)


def eval_model(model, val_iter, args={}, write=None, vocab=None):
	batchsize = args["batchsize"]
	loss_fn = args["loss_fn"]
	total_epoch_loss = 0
	total_epoch_acc = 0
	model.eval()

	output = {}

	with torch.no_grad():
		for idx, batch in enumerate(val_iter):
			if args.get("load_sample_dataset", None) == "sst":
				p = batch.text[0]
				pl = batch.text[1]
				h = None
				hl = None
			else:
				p = batch.premise[0]
				pl = batch.premise[1]
				if args.get("ph_as_one", False) == False:
					h = batch.hypothesis[0]
					hl = batch.hypothesis[1]
				else:
					h = None
					hl = None
			target = batch.label
			target = torch.autograd.Variable(target).long()
			if torch.cuda.is_available():
				p = p.cuda()
				if args.get("ph_as_one", False) == False:
					h = h.cuda()
				target = target.cuda()
			prediction = model(p,pl,h,hl)
			loss = loss_fn(prediction, target)
			num_corrects = (torch.max(
				prediction, 1)[1].view(target.size()).data == target.data).sum()
			if args.get("load_sample_dataset", None) == "sst":
				acc = 100.0 * num_corrects/(len(batch) - sum(target == 2))
			else:
				acc = 100.0 * num_corrects/len(batch)
			total_epoch_loss += loss.item()
			total_epoch_acc += acc.item()
			if write != None:
				smx = nn.Softmax(dim=1)
				prediction = smx(prediction)
				for ct, b in enumerate(batch.premise[0]):
					if args.get("ph_as_one", False) == True:
						output[b] = [prediction[ct][0].item(), target[ct]]
					else:
						output[b] = [batch.hypothesis[0][ct], prediction[ct][0].item(), target[ct]]

	if write != None:
		with open(write, "w+") as wf:
			for key in output.keys():
				p = " ".join([vocab.itos[i.item()] for i in key if i.item() > 1])
				if args.get("ph_as_one", False) == False:
					h = " ".join([vocab.itos[i.item()] for i in output[key][0] if i.item() > 1])
					wf.write("{}\t{}\t{}\t{}\n".format(p, h, output[key][1], output[key][2]))
				else:
					p, h = p.split(" SEP ")
					wf.write("{}\t{}\t{}\t{}\n".format(p, h,output[key][0], output[key][1]))

	return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)

def tsave(model, experiment_out_dir, TEXT, val_score, args={}):
	model_path = os.path.join(experiment_out_dir, 'best_model2_{}.torch'.format(model.name))
	print("model saved to {}".format(experiment_out_dir))
	with open(os.path.join(experiment_out_dir, "vocab.txt"), "w+") as wf:
		for vocab_index in sorted(TEXT.vocab.stoi.keys()):
			wf.write("{}\t{}\n".format(vocab_index, TEXT.vocab.stoi[vocab_index]))
		wf.close()
	with open(os.path.join(experiment_out_dir, "scores.txt"), "w+") as scoref:
		scoref.write("{}\t{}\n".format("filename", args["use_for_patience"]))
		scoref.write("{}\t{}\n".format("validate.txt", val_score))
		scoref.close()
	if args.get("pretrained_model_path", "") == "":
		#We're going to reuse the saved config for testing, so we'll overwrite some stuff
		args["pretrained_model_path"] = model_path
		args["pretrained_model_config"] = os.path.join(experiment_out_dir,"config.txt")
	else:
		with open(os.path.join(experiment_out_dir, "config.txt"), "w+") as wf:
			for key in args.keys():
				if args[key] != "":
					wf.write("{}\t{}\n".format(key, args[key]))
			wf.close()
	model.train(mode=True)


	torch.save(model, os.path.join(experiment_out_dir, 'best_model_{}.torch'.format(model.name)))
	torch.save(model.state_dict(), model_path) #above
	if args.get("pretraining", False) == False:
		model.sequence_model.train(mode=True)
		torch.save(model.sequence_model, os.path.join(experiment_out_dir, 'lm_best_model_{}.torch'.format(model.name)))
		torch.save(model.sequence_model.state_dict(), os.path.join(experiment_out_dir, 'lm_best_model2_{}.torch'.format(model.name)))

def load_model(args, sequence_model=None, word_embeddings = None):
	if word_embeddings is not None:
		args["char_embedding_dim"] = 300
		args["transformer_nhead"] = 10
	vocab_size = args["vocab_size"]
	learning_rate = args["learning_rate"]#2e-5
	batch_size = args["batchsize"]
	if args.get("load_sample_dataset", None) in ["snli", "mnli"]:
		output_size = 3
	elif args.get("load_sample_dataset", None) == "sst":
		output_size = 2
	else:
		output_size = 2
	currently_pretraining = args["pretraining"]
	if currently_pretraining == True:
		args["use_for_patience"] = "ppl"
	else:
		args["use_for_patience"] = "acc"

	load_sequence_model = args.get("pretrained_model_path", "") != ""
	#print("pretrained model path", args.get("pretrained_model_path", ""))
	if sequence_model==None:
		if load_sequence_model == False:
			print("Sequence model from scratch")
			mlp_head_from_scratch = True
			hidden_dim = args["hidden_dim"]
			char_embedding_dim = args["char_embedding_dim"]
			sequence_model_name = args["sequence_model_name"]
			dropout = args["dropout"]
			num_stacked_lstm = args.get("num_stacked_lstm", 1)
			transformer_nhead = args.get("transformer_nhead", 1)
			transformer_stackedlayers = args.get("transformer_stackedlayers", 1)
		else:
			#print("LOADING SEQUENCE MODEL!!!!!")
			config_pretrained_model = create_config(args["pretrained_model_config"])
			hidden_dim = config_pretrained_model["hidden_dim"]
			dropout = config_pretrained_model["dropout"]
			char_embedding_dim = config_pretrained_model["char_embedding_dim"]
			sequence_model_name = config_pretrained_model["sequence_model_name"]
			num_stacked_lstm = config_pretrained_model.get("num_stacked_lstm", 1)
			transformer_nhead = config_pretrained_model.get("transformer_nhead", 1)
			transformer_stackedlayers = config_pretrained_model.get("transformer_stackedlayers", 1)
			if args.get("load_finetuning", False) == True:
				#We are loading a saved entailment model.
				mlp_head_from_scratch = False
			else:
				#We are loading a saved sequence model, and need a new entailment head on top.
				mlp_head_from_scratch = True
		if sequence_model_name == "transformer":
			sequence_model = TransformerModel(vocab_size = vocab_size,
							embedding_dim = char_embedding_dim,
							hidden_dim = hidden_dim,
							nhead = transformer_nhead,
							nlayers = transformer_stackedlayers,
							dropout = dropout, pretraining= currently_pretraining,
							word_embeddings = word_embeddings)
		elif sequence_model_name == "lstm":
			sequence_model = LSTMClassifier(output_size, hidden_dim, vocab_size,
							char_embedding_dim,
							num_stacked_lstm = num_stacked_lstm,
							dropout=dropout,
							pretraining=currently_pretraining,
							word_embeddings=word_embeddings)
		elif sequence_model_name == "cbow":
			sequence_model = CBOWClassifier(output_size, vocab_size,char_embedding_dim)
		if  load_sequence_model == True and args.get("load_finetuning", False) == False:
			#print("actually loaded model")
			sequence_model.load_state_dict(torch.load(args["pretrained_model_path"], map_location=args["device"]))

	else:
		#We passed in a model we were in the process of pretraining
		sequence_model.pretraining = False
		sequence_model_name = sequence_model.name
		mlp_head_from_scratch = True
	if currently_pretraining == False:
		if mlp_head_from_scratch == True:
			#print("mlp head from scratch")
			mlp_hidden_dim = args["mlp_hidden_dim"]
			mlp_num_layers = args["num_layers"]
			mlp_dropout = args["mlp_dropout"]
		else:
			mlp_hidden_dim = config_pretrained_model["mlp_hidden_dim"]
			mlp_num_layers = config_pretrained_model["num_layers"]
			mlp_dropout = config_pretrained_model["mlp_dropout"]
			#print("loading entailment model")
		if sequence_model_name == "transformer":
			sequence_model_output_size = sequence_model.embedding_dim
		else: #lstm
			sequence_model_output_size = sequence_model.hidden_dim
		model = EntailmentModule(sequence_model, input_dim_size=sequence_model_output_size,
					hidden_dim=mlp_hidden_dim,
					dropout=mlp_dropout,
					mlp_num_layers=mlp_num_layers,
					output_size=output_size,
					ph_as_one = args.get("ph_as_one", False))
		if mlp_head_from_scratch == False:
			model.load_state_dict(torch.load(args["pretrained_model_path"], map_location=args["device"]))
		if args.get("load_sample_dataset", None) == "sst":
			loss_fn = nn.CrossEntropyLoss(ignore_index=2)
		else:
			loss_fn = F.cross_entropy
	else:
		loss_fn = nn.CrossEntropyLoss(ignore_index=args["pad_idx"])
		model = sequence_model
	args["loss_fn"] = loss_fn
	return model, args

@toma.batch(initial_batchsize=INIT_BATCHSIZE)
def main(batchsize, args, sequence_model_from_pretraining=None):
	"""
	Main loop handles either pretraining a model, finetuning a model, or pretraining then finetuning.
	"""
	assert args["pretraining"] == True or args["finetuning"] == True

	if args["pretraining"] == True:
		currently = "pretraining"
		args["dir"] = args["pretraining_dir"]
	else:
		currently = "finetuning"
		args["dir"] = args["finetuning_dir"]
	print("Currently {} the model.".format(currently))

	experiment_out_dir = os.path.join(args["write_dir"], "models/{}").format(currently)
	if not os.path.exists(os.path.join(experiment_out_dir, "emb_checkpoints")):
		os.makedirs(os.path.join(experiment_out_dir, "emb_checkpoints"))

	args["batchsize"] = batchsize #for toma
	print("TOMA BATCHSIZE:", args["batchsize"]) #expect this to run a few times if we start with large batch size

	torch.manual_seed(args["seed"])

	print("Experiments will be written to {}".format(experiment_out_dir))
	if not os.path.exists(experiment_out_dir):
		os.makedirs(experiment_out_dir)
	else:
		if args.get("overwrite", False) == False:
			print("Directory already exists! Wipe it or --overwrite")

	if args.get("probe", False) == True:
		args["batchsize"] = 1

	TEXT, vocab_size, train_iter, val_iter, test_iter, word_embeddings, train_size, pad_idx = load_data.load_dataset(args=args)
	args["pad_idx"] = pad_idx
	#TODO: Check right here if vocab size != emb size of pretraining model.
	args["vocab_size"] = vocab_size
	args["TEXT"] = TEXT
	model, args = load_model(args, sequence_model_from_pretraining, word_embeddings)

	if args.get("probe", False) == True:
		#Probe from the sample pretraining model.
		assert args["pretraining"] == True
		assert args["predefined_vocab"] != ""
		assert args["pretrained_model_path"] != ""
		probe_pretraining_model(model, val_iter, args, write=args["probewrite_path"])
		sys.exit(1)

	val_loss_hist = []
	if currently == "finetuning": #Use accuracy as metric
		val_acc_hist = []
		prev_best_val_score = float("-inf")
	else: #If pretraining, use perplexity as metric
		val_ppl_hist = []
		prev_best_val_score = float("inf")

	batchsize = args["batchsize"]

	optim = torch.optim.Adam(filter(lambda p: p.requires_grad,
		model.parameters()), lr=args["learning_rate"], weight_decay=args["weight_decay"])

	"""
	Here is the main training loop.
	"""
	for epoch in range(args.get("num_epochs", 500)):
		if currently == "pretraining":
			if args["sequence_model_name"] == "lstm":
				weights = model.word_embeddings.weight.cpu().data.numpy()
			else:
				weights = model.encoder.weight.cpu().data.numpy()
			checkpoint_path = os.path.join(experiment_out_dir, "emb_checkpoints")
			#Save every embedding, every run. Maybe this is why the grid is so backed up?
			with open(os.path.join(checkpoint_path, "emb_{}.txt".format(epoch)), "w+") as wf:
				for idx, row in enumerate(weights):
					wf.write("{}\t{}\n".format(TEXT.vocab.itos[idx], " ".join([str(x) for x in row])))
			train_loss, train_ppl = train_model_pretraining(model,optim, train_iter, epoch, args)
			val_loss, val_ppl = eval_model_pretraining(model, val_iter, args)

			val_ppl_hist.append(val_ppl)
			if val_ppl < prev_best_val_score:
				prev_best_val_score = val_ppl
				time_since_best = 0
			else:
				time_since_best += 1

			print('Epoch: %s, Train Loss: %s, Train Perplexity: %s, Val. Loss: %s, Val. Perplexity: %s' \
			  	% (str(epoch+1), str(train_loss), str(train_ppl), str(val_loss), str(val_ppl)))

		else: #otherwise we are finetuning
			train_loss, train_acc = train_model(model,optim, train_iter, epoch, args)
			val_loss, val_acc = eval_model(model, val_iter, args)

			val_acc_hist.append(val_acc)
			if val_acc > prev_best_val_score:
				prev_best_val_score = val_acc
				time_since_best = 0
			else:
				time_since_best += 1

			print('Epoch: %s, Train Loss: %s, Train Acc: %s, Val. Loss: %s, Val. Acc: %s' \
			  	% (str(epoch+1), str(train_loss), str(train_acc), str(val_loss), str(val_acc)))

		val_loss_hist.append(val_loss)

		if time_since_best == 0:
			print("New best score! Saving model.....")
			tsave(model, experiment_out_dir, TEXT, prev_best_val_score, args)
		else:
			if time_since_best > args["early_stop"]:
				print("Run ended on epoch {}: overfit (time since best dev loss/acc: {})".format(epoch,time_since_best))
				break

		sys.stdout.flush()
	if currently == "finetuning":
		with open(os.path.join(experiment_out_dir, "acc_hist.txt"), "w+") as wf:
			for acc in val_acc_hist:
				wf.write("{}\n".format(acc))
	else: #currently pretraining
		with open(os.path.join(experiment_out_dir, "ppl_hist.txt"), "w+") as wf:
			for ppl in val_ppl_hist:
				wf.write("{}\n".format(ppl))

	with open(os.path.join(experiment_out_dir, "loss_hist.txt"), "w+") as wf:
		for loss in val_loss_hist:
			wf.write("{}\n".format(loss))

	if args.get("finetuning", False) == True:
		if args.get("pretraining", False) == True:
			#stop pretraining; start finetuning
			args["pretraining"] = False
			main(args, model)
		else:
			#otherwise we are done finetuning, and should test our model
			test(args, predefined_vocab=TEXT.vocab, model=model)

if __name__ == "__main__":
	args = sys.argv
	parser = argparse.ArgumentParser()

	#If you set the config arg, the config will override all below args EXCEPT taskid
	parser.add_argument("--config", type=str, default="")
	parser.add_argument("--sequence_model_name", type=str, default="lstm")
	parser.add_argument("--pretraining_dir", type=str, default="data/compids_generation/pretraining/trial")
	parser.add_argument("--finetuning_dir", type=str, default="data/compids_generation/entailment_100/trial")
	parser.add_argument("--experiment_label", type=str, default="")
	parser.add_argument("--hidden_dim", type=int, default=32)
	parser.add_argument("--mlp_hidden_dim", type=int, default=32)
	parser.add_argument("--mlp_dropout", type=float, default=0)
	parser.add_argument("--num_layers", type=int, default=2)
	parser.add_argument("--char_embedding_dim", type=int, default=8)
	parser.add_argument("--learning_rate", type=float, default=.0001)
	parser.add_argument("--weight_decay", type=float, default=0)
	parser.add_argument("--batchsize", type=int, default=50)
	parser.add_argument("--dropout", type=float, default=.5)
	parser.add_argument("--task_id", type=int, default=0)
	parser.add_argument("--job_id", type=int, default=0)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--seed", type=int, default=2522524)
	parser.add_argument("--num_epochs", type=int, default=1250)
	parser.add_argument("--write_dir", type=str, default="output")
	parser.add_argument("--predefined_vocab", type=str, default=None)
	parser.add_argument("--test", action="store_true")
	parser.add_argument("--test_directory", type=str, default="")
	parser.add_argument("--early_stop", type=int, default=10)
	parser.add_argument("--num_stacked_lstm", type=int, default=1)
	parser.add_argument("--max_seq_len", type=int, default=60)
	parser.add_argument("--save", action="store_true") #save val predictions
	parser.add_argument("--pretrained_model_path", type=str, default="")
	parser.add_argument("--pretrained_model_config", type=str, default="")
	parser.add_argument("--load_sample_dataset", type=str, default=None)
	parser.add_argument("--transformer_nhead", type=int, default=1)
	parser.add_argument("--transformer_stackedlayers", type=int, default=1)
	parser.add_argument("--pretraining", action="store_true")
	parser.add_argument("--finetuning", action="store_true")
	parser.add_argument("--ph_as_one", action="store_true")
	parser.add_argument("--overwrite", action="store_true")
	parser.add_argument("--probe", action="store_true")
	args = vars(parser.parse_args())

	if args["config"] != "":
		c2 = create_config(args["config"])
		task_id = args["task_id"]
		for key in c2.keys():
			args[key] = c2[key]

	if args.get("experiment_label","") == "":
		args["write_dir"] = os.path.join("output", "unknown_exp")
	else:
		args["write_dir"] = os.path.join("output", args["experiment_label"])

	print(args)

	if args["test"] == True:
		test(args)
	else:
		main(args)
