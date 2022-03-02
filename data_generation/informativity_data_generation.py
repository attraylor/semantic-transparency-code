import argparse
from tqdm import tqdm
import os
import minimal_proplogic as apl
from world_truth_count import all_satisfying_assignments
import numpy as np
import time
from copy import deepcopy

import itertools

def count(s):
	return len(s.replace("(", "").replace(")", "").replace(" ", ""))

def r(x):
	return np.random.random()

def sentence_meets_conditions(sentence, observed_worlds, alternative_worlds, char_map ="abc"):
	"""
	Evaluates whether sentence can be true in observed worlds and cannot be true in alternative worlds.
	input:
		sentence: string
		observed_worlds: list of strs
		alternative_worlds: list of strs
		char_map: string of alphabet
	output:
		True if sentence can be true in observed worlds and not in alternative worlds; False otherwise
	"""
	for alternative_world in alternative_worlds:
		dict_of_world = apl.getdict_char(alternative_world, char_map)
		if apl.eval_assignment_in_world(sentence, dict_of_world) == True:
			return False
	for observed_world in observed_worlds:
		dict_of_world = apl.getdict_char(observed_world, char_map)
		if apl.eval_assignment_in_world(sentence, dict_of_world) == False:
			return False
	return True


def sentence_generator(char_map = "abc"):
	"""
	Generates all possible sentences in order from smallest to largest. Randomizes within length.
	Input:
		char_map: list of alphabet symbols for string
	Yields:
		new_sentence: prop logic sentence to later be evaluated for observed/alternative worlds
		item_count: count how many operators and non-operator symbols are in the sentence
	"""
	sentence_depth = 0
	seen = set()
	while True:
		sent_structures = expand("S", level=sentence_depth)
		sent_structures = list(set(sent_structures) - seen)
		sent_structures = sorted(sent_structures, key= lambda x: (count(x), r(x)))
		for sent in sent_structures:
			item_count = count(sent)
			symbol_count = sent.count("T")
			split_sent = sent.split("T")
			all_char_assignments = list(itertools.product(char_map, repeat=symbol_count))
			np.random.shuffle(all_char_assignments)
			for char_assignment in all_char_assignments:
				#assign symbols
				new_sentence = ""
				for i in range(symbol_count):
					new_sentence += split_sent[i] + char_assignment[i]
				new_sentence += split_sent[-1]
				yield new_sentence.replace(" ",""), item_count
		seen.update(sent_structures)
		sentence_depth += 1

def find_shortest_sentence(observed_worlds, alternative_worlds, char_map = "abc"):
	for x, len_x in sentence_generator(char_map):
		if sentence_meets_conditions(x, observed_worlds, alternative_worlds, char_map=char_map):
			return x, len_x

grammar = {
	'S': ["S", "~ ( S )", "( S & S )", "( S | S )"],
	'T': ["a"]
}

def expand(tpl, level=0):
	ret = []
	if level > 0:
		rewrites = [] # one list for each token in tpl, where list contains all rewrite options
		for t in tpl.split():
			if t in grammar:
				rewrites.append(grammar[t])
			else:
				rewrites.append([t])
		# generate all combinations of rewrite options
		all_rewrites = list(itertools.product(*rewrites))
		for new_tpl in all_rewrites:
			for rw in expand(' '.join(new_tpl), level=level-1):
				ret.append(rw)
	else:
		# bottom out by setting everything to terminal symbols
		ret.append(tpl.replace("S", "T"))
	return ret

def dict_to_world(d, char_map = "abcdefghijklmnopqrstuvwxyz"):
	world = ["x"] * len(char_map)
	for key, val in d.items():
		if val == True:
			bit = "1"
		else:
			bit = "0"
		world[char_map.index(key)] = bit
	return "".join(world)

def main(args):
	if not os.path.exists(os.path.join(args.dir, "data")):
		os.makedirs(os.path.join(args.dir, "data"))
	if not os.path.exists(os.path.join(args.dir, "config")):
		os.makedirs(os.path.join(args.dir, "config"))

	np.random.seed(args.seed)

	if args.tqdm == True:
		pbar = tqdm(total=args.num_examples)
	i = 0
	num_sents_checked = 0
	cmap = "abcde"

	sent_to_worlds = {}
	seen = args.seen #set
	start_time = time.time()
	while i < args.num_examples:
		num_sents_checked += 1
		if num_sents_checked % 10000 == 0:
			print("{} sentences checked".format(num_sents_checked))


		var_from_poisson = np.random.poisson(args.unique_symbols_param) + 1
		min_poss_vars = 1
		while 2 ** min_poss_vars < args.num_observed_worlds + args.num_alternative_worlds:
			min_poss_vars += 1
		var_from_poisson = max(min_poss_vars, var_from_poisson)
		unique_variable_count = min(var_from_poisson, args.max_unique_variables)
		cmap_subset = cmap[:unique_variable_count]
		all_possible_worlds = ["".join(x) for x in itertools.product("01", repeat = len(cmap_subset))]
		#generate observed_worlds
		#make sure we don't generate the same world twice
		worlds = np.random.choice(all_possible_worlds, size = min(2 ** len(cmap_subset), args.num_observed_worlds + args.num_alternative_worlds), replace = False)
		observed_worlds = worlds[:args.num_observed_worlds]
		alternative_worlds = worlds[args.num_observed_worlds:]
		sentence, _ = find_shortest_sentence(observed_worlds, alternative_worlds, char_map = cmap_subset)
		symbol_nums = np.random.choice(list(range(args.vocab_size)), size=len(cmap_subset), replace=False)
		sentence = list(sentence) #split by char
		for idx, char in enumerate(cmap_subset):
			sentence = ["Symbol_{}".format(symbol_nums[idx]) if x == char else x for x in sentence]
		sentence = " ".join(sentence)
		if sentence not in seen:
			unseen = 0
			seen.add(sentence)
			sent_to_worlds[sentence] = (observed_worlds, alternative_worlds)
			if args.tqdm == True:
				pbar.update(1)
			i += 1
		else:
			unseen += 1
			if unseen >= args.unseen_threshold:
				print("Randomly generated {} sentences, all were seen. Returning in {}".format(unseen, str(time.time() - start_time)))
				return seen, sent_to_worlds

	print("{} dataset size {} t / {} f: time to complete {}".format(args.num_examples, args.num_observed_worlds, args.num_alternative_worlds, str(time.time() - start_time)))
	return seen, sent_to_worlds

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_examples", type=int, default=110000)
	parser.add_argument("--val_ratio", type=float, default=1/11)
	parser.add_argument("--vocab_size", type=int, required=True)
	parser.add_argument("--dir", type=str, required=True)
	parser.add_argument("--seed", type=int, default=2522524)
	parser.add_argument("--tqdm", action="store_true")
	parser.add_argument("--unique_symbols_param", type=float, default=1.0) #lambda for poisson
	parser.add_argument("--num_observed_worlds", type=int, default=1)
	parser.add_argument("--num_alternative_worlds", type=int, default=1)
	parser.add_argument("--max_unique_variables", type=int, default = 4)
	parser.add_argument("--unseen_threshold", type=int, default = 500)


	#symbol_distribution can be: topicmodel, uniform

	seen = set()

	args = parser.parse_args()
	args.seen = seen

	seen, sent_to_worlds = main(args)
	sentences = list(seen)

	np.random.shuffle(sentences)
	val_cutoff = int(np.floor(len(seen) * args.val_ratio))
	val_data = sentences[:val_cutoff]
	train_data = sentences[val_cutoff:]
	print("total data: {} ex".format(len(sentences)))
	print("train data: {} ex".format(len(train_data)))
	print("val data: {} ex".format(len(val_data)))

	for dataset, filename in [(train_data,"train.txt"), (val_data, "validate.txt")]:
		with open(os.path.join(args.dir, "data", filename), "w+") as wf_data:
			with open(os.path.join(args.dir, "data", "{}.worlds".format(filename)), "w+") as wf_worlds:
				for sentence in dataset:
					wf_data.write(sentence)
					wf_data.write("\n")
					observed_worlds, alternative_worlds = sent_to_worlds[sentence]
					wf_worlds.write("{}\t{}\t{}\n".format(sentence, observed_worlds, alternative_worlds))
