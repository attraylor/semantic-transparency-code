import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
from collections import defaultdict
import itertools
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import fowlkes_mallows_score, homogeneity_completeness_v_measure
import pandas as pd
from sklearn.decomposition import PCA #Grab PCA functions
from matplotlib import patches
import matplotlib.pylab as pylab


from torchtext.vocab import Vocab
from torchtext import data
import sys
sys.path += ["../src"]
print(sys.path)
#MIGHT HAVE TO FIX YOUR PATH
from utils import load_vocab, create_config
from main import load_model


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)



parser = argparse.ArgumentParser()
parser.add_argument("--nsymbs", type=int, default=5)
parser.add_argument("--model_name", type=str, default="medium-lstm")
args = parser.parse_args()

def cossim(a, b):
	num = np.dot(a, b)
	denom = np.linalg.norm(a) * np.linalg.norm(b)
	return num * 1.0 / max(1e-10, denom)


def load_weights(model_dir):
	config_file = os.path.join(model_dir, "config.txt")
	config = create_config(config_file)
	config["device"] = "cpu"
	vocab_file = os.path.join(model_dir, "vocab.txt")
	print(vocab_file)
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
	print(count_parameters(model))
	if config["sequence_model_name"] == "lstm":
		weights = model.word_embeddings.weight.cpu().data.numpy()
	else:
		weights = model.encoder.weight.cpu().data.numpy()
	return vocab.itos, weights

'''parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True)
args = parser.parse_args()'''
colors = {"&": "orange", "|": "brown", "~": "red", "and/or": "purple", "and/not": "green", "or/not": "blue" }
label_order = ["&", "|", "~", "and/or", "and/not", "or/not"]
dirs_gold = []
dirs_gold.append("Random")
#You (the reviewer) might have to edit these to be the first Syntactic, Truthfulness, and Explicit Grounding condition expts
dirs_gold.append("output/acl-proplogic-ksynNSYN-MNAME/results/11/models/pretraining/")
dirs_gold.append("output/acl-proplogic-ksynNSYN-MNAME/results/1/models/pretraining/")
dirs_gold.append("output/acl-proplogic-ksynNSYN-MNAME/results/5/models/pretraining/")


subplot_titles = ["Random",
				  "Syntactic",
 				  "Truthfulness",
				  #"Informativity",
				  "Explicit Grounding"]

params = {'legend.fontsize': 16,
         'axes.titlesize':24,
		 'axes.labelsize':24}
pylab.rcParams.update(params)

fig_size = 5
settings = [""]
fig = plt.figure(figsize=(fig_size * 2, 2 * fig_size))
fig.tight_layout(pad=1.0)
for setting_idx, setting in enumerate(settings):
	dirs = []
	for dir in dirs_gold:
		complete_model_name = args.model_name + setting
		dirs.append(dir.replace("MNAME", complete_model_name).replace("NSYN", str(args.nsymbs)))
	metr = "cos"

	#models = ["lstm"]#["transformer", "lstm"]
	#for model_idx, model_name in enumerate(models):
	for dir_idx, dir in enumerate(dirs):
		#dir = dir.replace("MNAME", model_name)
		num_rows = len(settings)
		num_cols = len(dirs)
		curr_idx = dir_idx + 1
		plt.subplot(2, 2, curr_idx)
		print(dir, subplot_titles[dir_idx])
		plt.gca().set_title(subplot_titles[dir_idx])
		if dir == "Random":
			itos = []
			for label in ["&", "|", "~"]:
				itos.append(label)
				for i in range(2, args.nsymbs +1):
					itos.append(label + str(i))
			weights = np.random.rand(args.nsymbs * 3,32)
		else:
			#open all files from folder
			itos, weights = load_weights(dir)
		keywords = [x for x in itos if x[0] in ["&", "|", "~"]]
		keywords_idx = [itos.index(x) for x in keywords]
		keywords_weights = weights[keywords_idx]
		itos = keywords
		'''wvs = {}
		all_wvs = []
		itos = []
		ct = 0
		with open(os.path.join(dir, "emb_{}.txt".format(i))) as rf:
			for line in rf:
				word, vec = line.strip().split("\t")
				wv = [float(val) for val in vec.split(" ")]
				wvs[word] = wv
				all_wvs.append(wv)
				itos.append(word)
				if any([x in word for x in ["&", "|", "~"]]):
					keywords.append(word)
				else:
					if ct < args.symbs and "Symbol_" in word:
						ct += 1'''

		#matr = np.asmatrix(all_wvs)#[wvs[word] for word in keywords])


		#assign ground truth clustering labels
		ground_truth_labels = []
		for keyword in keywords:
			if "&" in keyword:
				ground_truth_labels.append(0)
			elif "|" in keyword:
				ground_truth_labels.append(1)
			elif "~" in keyword:
				ground_truth_labels.append(2)


		#cluster_colors = {0: "blue", 1: "red", 2: "orange"}
		cluster_colors = ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]
		markers = ["v", "o", "s"]
		cluster_names = ["AND synonyms", "OR synonyms", "NOT synonyms"]
		pca = PCA(n_components=2)
		result = pca.fit_transform(keywords_weights)
		ground_truth_labels = np.asarray(ground_truth_labels)
		Z = 1.960
		for n, color in enumerate(cluster_colors):
			results = result[ground_truth_labels == n]
			mean_x = np.mean(results[:,0])
			mean_y = np.mean(results[:, 1])
			ellipse_width = 2 * Z * (np.std(results[:, 0]) / np.sqrt(5))
			ellipse_height = 2 * Z * (np.std(results[:, 1]) / np.sqrt(5))
			print(ellipse_width)
			if ellipse_width < .1:
				ellipse_width = .1
			e = patches.Ellipse(xy=(mean_x, mean_y), width = ellipse_width, height = ellipse_height, color=color, linewidth=1, edgecolor="black", alpha=.7, zorder=1)
			plt.scatter(results[:, 0], results[:, 1], marker = markers[n], edgecolor="black", s=100, label=cluster_names[n],zorder=2)
			plt.gca().add_patch(e)

		'''for idx, word in enumerate(itos):
			#if not "~" in word:
			plt.scatter(result[idx, 0], result[idx, 1], color = "grey")#cluster_colors[ground_truth_labels[idx]])
			#plt.annotate(word, xy=(result[idx, 0], result[idx, 1]))
		#plt.xlim(-.5,.5)'''
		'''for idx, word in enumerate(keywords):
			#if not "~" in word:
			if len(word) == 0:
				plt.scatter(result[itos.index(word), 0], result[itos.index(word), 1], label=cluster_names[word[0]], color = cluster_colors[ground_truth_labels[idx]])
			else:
				plt.scatter(result[itos.index(word), 0], result[itos.index(word), 1], color = cluster_colors[ground_truth_labels[idx]])
		'''#plt.ylim(-1,1)
		plt.gca().set_yticklabels([]) #Hide ticks
		plt.gca().set_xticklabels([]) #Hide ticks
		if curr_idx == 2:
			plt.gca().legend(loc = "upper right")#, bbox_to_anchor=(1.8,1))
		plt.gca().set_box_aspect(1)




	#handles, labels = plt.gca().get_legend_handles_labels()
	#fig.legend(handles, labels)
	#ax.legend()
	#plt.xlabel("Epochs")
	#plt.ylabel("Cosine Similarity")
		#ax.legend()
	if not os.path.exists("output/pca"):
		os.makedirs("output/pca")
	plt.savefig("output/pca/{}_{}_{}.pdf".format(args.model_name, args.nsymbs, setting), dpi = 300)
