from nltk.grammar import PCFG
import argparse
import random
from tqdm import tqdm
import os
from world_truth_count import satisfying_assignment_exists, all_satisfying_assignments
from nltk_rules import S0, tree_from_hyperparam, generate_tree
import numpy as np

from copy import deepcopy

def main(args):

	if not "only_syntactic_constraint" in vars(args).keys():
		args.only_syntactic_constraint = False

	"""
	seen_symbols means-- if we haven't seen a symbol in the training set, we shouldn't use it in the validation set
	(Will blow up LM perplexity for no reason)
	"""
	seen = set()
	seen_symbols = set()

	#If validation file, load in seen examples and seen symbols
	if "remove_train_from_val_file" in vars(args).keys() and args.remove_train_from_val_file != None:
		is_val = True
		with open(args.remove_train_from_val_file) as rf:
			for line in rf:
				l = line.strip()
				seen.add(l)
				l = line.split(" ")
				for x in l:
					if "Symbol_" in x:
						seen_symbols.add(x)
	else:
		is_val = False

	if not os.path.exists(os.path.join(args.dir, "data")):
		os.makedirs(os.path.join(args.dir, "data"))
	if not os.path.exists(os.path.join(args.dir, "config")):
		os.makedirs(os.path.join(args.dir, "config"))

	np.random.seed(args.seed)
	random.seed(args.seed)
	grammar = PCFG(S0, tree_from_hyperparam(S0, args.depth_param))
	if args.tqdm == True:
		pbar = tqdm(total=args.num_examples)
	dataset_example_count = 0

	#Some statistics about the dataset
	sum_tas = []
	sentence_lengths = []
	unique_variable_count_hist = []


	with open(os.path.join(args.dir, "data", args.filename), "w+") as wf_data:
		while dataset_example_count < args.num_examples:

			"""
			We generate a random propositional logic sentence structure, based on the depth parameter.
			This tree is populated with filler symbols, which are edited later.
			"""

			tree = generate_tree(grammar)

			#Now get a string representation of the tree
			a = ""
			symbol_locations_in_tree = []
			for idx, leaf in enumerate(tree.leaves()):
				if leaf == "Symbol_FILLER":
					symbol_locations_in_tree.append(idx)
				a += leaf

			split_sentence = a.split(" ")
			symbol_locations_in_str = [j for j in range(len(split_sentence)) if "Symbol_FILLER" == split_sentence[j]]

			#Decide a number of unique variables for the sentence using non-negative Poisson distribution
			unique_variable_count = np.random.poisson(args.unique_symbols_param) + 1 #We want to start at 1, not 0

			#Try each sentence structure a few times with different combinations of unique non-operator symbols.
			for k in range(0, 5):
				modified_sentence = deepcopy(split_sentence)

				#Select the symbol numbers that go into the sentence
				symbol_ints_for_sentence = np.random.choice(list(range(args.vocab_start, args.vocab_stop)), size = unique_variable_count, replace=False)
				symb_strs = ["Symbol_{}".format(n) for n in symbol_ints_for_sentence]

				chosen_symbol_numbers = np.random.choice(symb_strs, size = len(symbol_locations_in_str), replace = True)

				unique_symb_in_sentence = list(set(chosen_symbol_numbers))
				for symbol_idx, str_idx in enumerate(symbol_locations_in_str):
					modified_sentence[str_idx] = chosen_symbol_numbers[symbol_idx]
				final_sentence = " ".join(modified_sentence)
				satisfiable, _ = satisfying_assignment_exists(final_sentence, unique_symb_in_sentence)


				#If validation set: Check to see if a) we haven't seen it, and b) all symbols are in training
				if is_val == True:
					symbs_seen_in_train = [x in seen_symbols for x in unique_symb_in_sentence]
					if not all(symbs_seen_in_train):
						break
					elif final_sentence in seen:
						break

				"""
				If only the syntactic constraint is necessary for creating the dataset,
				allow the sentence-- otherwise only allow it if it is satisfiable
				"""

				if args.only_syntactic_constraint == True or satisfiable == True:

					seen.add(final_sentence)
					tas = all_satisfying_assignments(final_sentence, unique_symb_in_sentence)

					#If we need a grounding, just select a random variable assignment. Then append to sentence
					if args.grounding_map == True:
						chosen_ta = np.random.choice(tas)
						keys_list = [key for key in chosen_ta.keys()]
						np.random.shuffle(keys_list)
						grounding = []
						for key in keys_list:
							grounding.append(key)
							grounding.append(str(chosen_ta[key]))
						final_sentence = " ".join(modified_sentence) + " <grounding> " + " ".join(grounding)
					if args.tqdm == True:
						pbar.update(1)
					dataset_example_count += 1


					#Just keeping track of some statistics
					sum_tas.append(len(tas))
					sentence_lengths.append(len(split_sentence))
					unique_variable_count_hist.append(unique_variable_count)

					wf_data.write(final_sentence)
					wf_data.write("\n")

					break

	with open(os.path.join(args.dir, "config", "stats"), "w+") as wf:
		wf.write("avg possible worlds {}\n".format(sum(sum_tas) * 1.0 / len(sum_tas)))
		wf.write("max sentence length {}\n".format(max(sentence_lengths)))
		wf.write("avg sentence length {}\n".format(sum(sentence_lengths) * 1.0 / len(sentence_lengths)))
		wf.write("avg unique symbols {}\n".format(sum(unique_variable_count_hist) * 1.0 / len(unique_variable_count_hist)))
		wf.write("max unique symbols {}\n".format(max(unique_variable_count_hist)))

	with open(os.path.join(args.dir, "config", "stats"), "r") as rf:
		for line in rf:
			print(line.strip())

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--vocab_start", type=int, default=0, help="Vocab starts on this symbol (e.g. vocab_start=0 means Symbol_0 is the first symbol)")
	parser.add_argument("--vocab_stop", type=int, required=True, help="Vocab ends on this symbol (e.g. vocab_start=500 means Symbol_499 is the last symbol)")
	parser.add_argument("--num_examples", type=int, default=100000, help="Number of training examples")
	parser.add_argument("--dir", type=str, required=True, help="Directory to put dataset generation files in")
	parser.add_argument("--filename", type=str, default="train.txt", help="Name of this file (probably train.txt or validate.txt)")
	parser.add_argument("--seed", type=int, default=2522524, help="Random seed")
	parser.add_argument("--unique_symbols_param", type=float, default=1.0, help="symbol_distribution=poisson: poisson lambda parameter")
	parser.add_argument("--depth_param", type=float, default=.9, help="Gamma parameter for PCFG tree")
	parser.add_argument("--only_syntactic_constraint", action="store_true")
	parser.add_argument("--remove_train_from_val_file", type=str, default=None, help="No examples from this file will be permitted if this arg is not None. Set this to train file path for val sets.")
	parser.add_argument("--grounding_map", action="store_true", help="True if grounding appended to end of sequence")
	parser.add_argument("--tqdm", action="store_true")

	args = parser.parse_args()

	main(args)
	if args.filename == "train.txt": #make the validation set too
		args.remove_train_from_val_file = os.path.join(args.dir, "data", args.filename)
		args.seed = args.seed * 2
		args.filename = "validate.txt"
		args.num_examples = 1000
		print("validate", args.dir)
		main(args)
