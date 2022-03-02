import argparse
import numpy as np
from scipy.spatial.distance import cdist
import os
import itertools
from collections import defaultdict
import seaborn as sns, matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path += ["../src"]
#Might have to mess with the paths
from utils import load_vocab, create_config
from main import load_model
from torchtext.vocab import Vocab
from torchtext import data

def dir_to_datasets(results_root):
	dataset_to_name = {
					   "syntactic": "Syntactic",
					   "truthfulness":"Truthfulness",
					   "informativity2T2A":"Informativity",
					   "grounding":"Explicit Grounding"
					   }
	mapping = defaultdict(list)
	subdirs = [os.path.join(results_root, o) for o in os.listdir(results_root) if os.path.isdir(os.path.join(results_root,o))]
	for dir in subdirs:
		config_path = os.path.join(dir, "models/pretraining", "config.txt")
		results_path = os.path.join(dir, "models/pretraining/train_pplhist.txt")
		val_path = os.path.join(dir, "models/pretraining/ppl_hist.txt")
		try:
			config = create_config(config_path)
			dataset_path = config["pretraining_dir"]
			seed = config["seed"]
			found = False
			for key in dataset_to_name.keys():
				if key in dataset_path:
					if found == True:
						print("results path matches more than one dir")
						print(results_path)
						sys.exit(1)
					else:
						mapping[dataset_to_name[key]].append(
								{"path_train": results_path,
								 "path_val": val_path,
								 "seed": seed,
								 "learning_rate": config["learning_rate"]}
								 )
						found = True

			if found == False:
				print(dataset_path)
				print("no known dataset found")
		except FileNotFoundError:
			print("dataset didnt run", results_path)
	mapping["Random"] = [0,1,2,3,4]
	return mapping


def load_weights(model_dir):
	config_file = os.path.join(model_dir, "config.txt")
	config = create_config(config_file)
	config["device"] = "cpu"
	vocab_file = os.path.join(model_dir, "vocab.txt")
	#print(vocab_file)
	if config["sequence_model_name"] == "lstm":
		model_path = os.path.join(model_dir, "best_model2_lstm.torch")
	else:
		model_path = os.path.join(model_dir, "best_model2_transformer.torch")
	stoi, itos = load_vocab(vocab_file)
	vocab = Vocab(stoi, specials=["<unk>", "PAD", "EOS"])
	vocab.stoi = stoi
	vocab.itos = itos
	tokenize = lambda x: x.split()
	TEXT = data.Field(sequential=True, tokenize=tokenize, pad_token="PAD", eos_token="EOS",
					  include_lengths=True, batch_first=True, fix_length=config["max_seq_len"])
	TEXT.vocab = vocab
	config["TEXT"] = TEXT
	config["dir"] = config["pretraining_dir"]
	config["pretrained_model_path"] = model_path
	model, model_config = load_model(config)
	#print(model)
	if config["sequence_model_name"] == "lstm":
		weights = model.word_embeddings.weight.cpu().data.numpy()
	else:
		weights = model.encoder.weight.cpu().data.numpy()
	return vocab.itos, weights



def train_classifier(X, y):
	return X #vector for each class label

def predict(X_test, examplars):
	dists = cdist(X_test, examplars, metric='cosine')
	#rank_order = list(np.argsort(dists).squeeze())
	y_pred = np.argmin(dists, axis=1)
	return y_pred

def score(y_true, y_pred):
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	acc = sum(y_true == y_pred) * 1.0 / len(y_true)
	return acc

def create_train_test_split(X, y, train_indices):
	y = np.asarray(y)
	train_indices = np.asarray(list(train_indices))
	#train_indices = []
	test_indices = [i for i in range(len(y)) if i not in train_indices]
	'''for label in [0,1,2]:
		indices = [i for i in range(len(y)) if y[i] == label]
		np.random.shuffle(indices) #This is inplace right
		train_indices.append(indices[0])
		test_indices += indices[1:]'''
	X_train = X[train_indices]
	X_test = X[test_indices]
	y_train = y[train_indices]
	y_test = y[test_indices]
	return X_train, y_train, X_test, y_test

shortcodes = {
				   "Random": "Rand",
				   "Syntactic": "Syn",
				   "Truthfulness": "Tru",
				   "Informativity": "I22",
				   "Explicit Ground": "Grd."
				   }

def load_pplhist(f):
    pplhist = []
    with open(f) as rf:
        for line in rf:
            pplhist.append(float(line.strip()))
    return pplhist

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="")
	parser.add_argument("--runname", type=str, default="all_datasets")
	parser.add_argument("--short", action="store_true")
	parser.add_argument("--seed", type=int, default=716)
	args = parser.parse_args()
	np.random.seed(args.seed)
	config = []
	with open(args.config) as rf:
		for line in rf:
			if len(line) > 0:
				experiment_label, model_name = line.strip().split("\t")
				config.append((experiment_label, model_name))
	results = []#results = defaultdict(list)
	for config_idx, (experiment_label, model_name) in enumerate(config):
		answers = defaultdict(list)
		results_root = 'output/{}/results/'.format(experiment_label)
		mapping = dir_to_datasets(results_root)
		for dataset_name in mapping.keys():
			for run_dict in mapping[dataset_name]:
				if dataset_name == "Random":
					itos = ["&", "&2", "&3","&4","&5","|", "|2", "|3","|4","|5","~", "~2", "~3","~4","~5"]
					weights = np.random.rand(32,15)
				else:
					try:
						train_path = run_dict["path_train"]
						val_path = run_dict["path_val"]
						seed = run_dict["seed"]
						val_ppl = load_pplhist(val_path)
						train_ppl = load_pplhist(train_path)
						print(val_ppl.index(min(val_ppl)), len(train_ppl), model_name)
						ppl = min(val_ppl)#train_ppl[val_ppl.index(min(val_ppl))]
					except FileNotFoundError:
						print("run not yet complete")
						continue

					if args.short == True:
						dn = shortcodes[dataset_name]
					else:
						dn = dataset_name
					if "Rand" in dn:
						results.append({"dataset": dn,
										"model_name": model_name,
										"run": seed,
										"ppl": ppl})
					elif dn not in ["I11", "I12", "I21"]:
						results.append({"dataset": dn,
										"model_name": model_name,
										"run": seed,
										"ppl": ppl,
										"ppl_len": len(train_ppl) - 15})

	subplot_titles = ["Syntactic", "Truthfulness", "Informativity", "Explicit Grounding"]

	df = pd.DataFrame(results)
	fig, axes = plt.subplots(figsize=(5,5))

	sns.stripplot(ax=axes, x="dataset", y="ppl", data=df, hue="model_name", jitter = .15)
	plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)

	print(df.groupby(["model_name", "dataset"]).mean())
	#fig.autofmt_xdate()

	if not os.path.exists("output/nnprobe"):
		os.makedirs("output/nnprobe")
	plt.tight_layout()
	plt.savefig("output/nnprobe/ppl{}.pdf".format(args.runname))


	#print(results)
