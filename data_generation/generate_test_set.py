import argparse
import sys
import os
import numpy as np


"""
This file is for generating 'sentence starters' for the language model.

These sentence starters are how we generate Table 2 in our paper.

"""


def load_vocab(path):
	"""
	load in presaved vocab from file

	input:
		path: path to vocab file, vocab file is index \t word \n
	output:
		imported_vocab: dict of str vocab_item to int index
		itos: dict of int index to str vocab_item (reverse of imported_vocab)
	"""
	imported_vocab = {}
	itos = {}
	try:
		with open(path) as rf:
			for ct, line in enumerate(rf):
				idx, word = line.strip().split("\t")
				imported_vocab[word] = int(idx)
				itos[int(idx)] = word
	except ValueError:
		print("expected vocab in index \t word format. exiting")
		sys.exit(1)
	try:
		itos_list = []
		for i in range(len(imported_vocab.keys())):
			itos_list.append(itos[i])
	except IndexError:
		print("vocab skips index. exiting")
		sys.exit(1)
	except KeyError:
		print("Two or more vocab tokens go to the same index. This is likely because vocab is in the val set but not the training set. This is a bad thing, and those tokens will be left out of the itos list. Don't know what that will cause")
	return imported_vocab, itos_list



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--vocab_file", type=str, default="vocab/ksyn5_5000_vocab.txt")
	parser.add_argument("--writepath", type=str, default="data/test")
	parser.add_argument("--cutoff", type=int, default=None, help="Maximum number of test examples")
	args = parser.parse_args()

	if not os.path.exists(args.writepath):
		os.makedirs(args.writepath)

	with open(os.path.join(args.writepath, "train.txt"), "w+") as wf:
		wf.write("")


	vocab, _ = load_vocab(args.vocab_file)


	symbols = [x for x in vocab.keys() if "Symbol_" in x]
	binary_operators = [y for y in vocab.keys() if any([z in y for z in ["&", "|"]])]
	unary_operators = [y for y in vocab.keys() if "~" in y]
	data = []
	for symbol in sorted(symbols):
		data.append("( {}".format(symbol))
		for unary_operator in unary_operators:
			data.append("{} ( {}".format(unary_operator, symbol))
		for binary_operator in binary_operators:
			data.append("( {} {}".format(symbol, binary_operator))

	if args.cutoff != None:
		np.random.shuffle(data)
		data = data[:args.cutoff]

	with open(os.path.join(args.writepath, "validate.txt"), "w+") as wf:
		for s in data:
			wf.write(s)
			wf.write("\n")
