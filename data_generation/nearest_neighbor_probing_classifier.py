import argparse
import numpy as np
from scipy.spatial.distance import cdist
import os
import itertools
from collections import defaultdict
import seaborn as sns, matplotlib.pyplot as plt
import pandas as pd

import sys
#Might need to edit your path in here
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
		results_path = os.path.join(dir, "models/pretraining")
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
								{"path": results_path,
								 "seed": seed,
								 "learning_rate": config["learning_rate"]}
								 )
						found = True

			if found == False:
				print("no known dataset found. stopping")
				sys.exit(1)
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
						path = run_dict["path"]
						seed = run_dict["seed"]
						itos, weights = load_weights(path)
					except FileNotFoundError:
						print("run not yet complete")
						continue
				keywords = [x for x in itos if x[0] in ["&", "|", "~"]]
				keywords_idx = [itos.index(x) for x in keywords]
				keywords_weights = weights[keywords_idx]
				ground_truth_labels = []
				operators = {0: "AND", 1: "OR", 2: "NOT"}
				for keyword in keywords:
					if "&" in keyword:
						ground_truth_labels.append(0)
					elif "|" in keyword:
						ground_truth_labels.append(1)
					elif "~" in keyword:
						ground_truth_labels.append(2)
				combos = []
				for label in [0,1,2]:
					indices = [i for i in range(len(ground_truth_labels)) if ground_truth_labels[i] == label]
					combos.append(indices)
				if len(combos[0]) * len(combos[1]) * len(combos[2]) > 1000:
					print ("op combos > 1000")
					all_operator_combinations = []
					for i in range(0, 1000):
						tup = []
						for label in [0,1,2]:
							tup.append(np.random.choice(combos[label]))
						all_operator_combinations.append(tup)
				else:
					all_operator_combinations = itertools.product(*combos)
				for operator_combination in all_operator_combinations:
					X_train, y_train, X_test, y_test = create_train_test_split(keywords_weights, ground_truth_labels, operator_combination)
					examplars = train_classifier(X_train, y_train)
					y_pred = predict(X_test, examplars)
					for label in [0,1,2]:
						indices = [i for i in range(len(y_test)) if y_test[i] == label]
						indices = np.asarray(indices)
						y_test_label = y_test[indices]
						y_pred_label = y_pred[indices]
						[answers[label].append(i) for i in y_pred_label]
						if args.short == True:
							dn = shortcodes[dataset_name]
						else:
							dn = dataset_name
						assert(all([x == label for x in y_test_label]))
						if "Rand" in dn:
							results.append({"dataset": dn,
											"run": seed,
											"learning_rate": 0,
											"operator": operators[label],
											"score": score(y_test_label, y_pred_label)})
						elif dn not in ["I11", "I12", "I21"]:
							results.append({"dataset": dn,
											"run": seed,
											"model": model_name + "\n",
											"learning_rate": run_dict["learning_rate"],
											"operator": operators[label],
											"score": score(y_test_label, y_pred_label)})
						#if dn == "Rand":
						#	print(results[-1])
						#results[operators[label]].append(score(y_test_label, y_pred_label))
	subplot_titles = ["Syntactic", "Truthfulness", "Informativity", "Explicit Grounding"]#["Syntactic", "Truthfulness", "Informativity (1/1)", "Informativity (1/2)", "Informativity (2/1)", "Informativity (2/2)", "Expl. Ground .85", "Expl. Ground .35", "Expl. Ground .55", "Expl. Ground .75", "Expl. Ground 0"]

	df = pd.DataFrame(results)
	fig, ax = plt.subplots()
	sns.barplot(x="model", y="score", hue="dataset", data=df, capsize=.1, hue_order = subplot_titles)
	plt.axhline(.3333, ls='--', color="black")
	ax.set_ylim(0,1.1)
	#fig.autofmt_xdate()

	#fig, axes = plt.subplots(len(config), 1, figsize=(5 * 1, 5 * len(config)))
	'''for config_idx, (experiment_label, model_name) in enumerate(config):
		if args.short == True:
			subplot_titles = [shortcodes[x] for x in subplot_titles]
		if "I12" in subplot_titles:
			subplot_titles.remove("I12")
		if "I11" in subplot_titles:
			subplot_titles.remove("I11")
		if "I21" in subplot_titles:
			subplot_titles.remove("I21")
		print(subplot_titles)
		sns.set(style="whitegrid")
		#group = df[df["operator"] != "NOT"]
		sns.barplot(ax=axes[config_idx], x="dataset", y="score", data=df, capsize=.1, order = subplot_titles)
		#axes[config_idx].axhline(.3333, ls='--', color="black")
		#hue="operator",
		#group = df[df["operator"] == "NOT"]
		#sns.barplot(ax=axes[config_idx,1], x="dataset", y="score", data=group, capsize=.1, color = sns.color_palette()[2], order = subplot_titles)#, ci="sd")
		axes[config_idx].set_title(model_name)
		#axes[config_idx,1].set_title("NOT accuracy (green)")
		axes[config_idx].set_ylim(0,1.1)
		#axes[config_idx,1].set_ylim(0,1.1)

			for label in [0,1,2]:
				maps = []
				for label2 in [0, 1, 2]:
					avg = sum([1 for i in answers[label] if i == label2])
					maps.append(avg * 1.0 / len(answers[label]))

				print(model_name, operators[label], maps[0], maps[1], maps[2])'''

	#sns.stripplot(ax=axes[idx], x="operator", y="score", data=group, color="0", jitter=.4, alpha=.03)
	if not os.path.exists("output/nnprobe"):
		os.makedirs("output/nnprobe")
	plt.savefig("output/nnprobe/{}.pdf".format(args.runname), bbox_inches = 'tight',pad_inches = 0)
	#print(results)
