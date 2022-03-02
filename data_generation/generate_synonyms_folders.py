import os
import argparse
import random

"""
Modifies files written by pcfg_data_generation and informativity_data_generation such that
there are a number of synonyms for each operator (defined by k_syn).

So if a sentence is "~ ( Symbol_1 & Symbol_2 )", reroll & and ~ to be &2, &3, &4, etc...
"""

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--read_dir", type=str, required=True)
	parser.add_argument("--write_dir", type=str, required=True)
	parser.add_argument("--k_syn", type=int, required=True)
	args = parser.parse_args()
	random.seed(2522524)
	if not os.path.exists(args.write_dir):
		os.makedirs(args.write_dir)

	files = os.listdir(args.read_dir)
	symbol_set = ["&", "|", "~"]
	suffixes = [j for j in range(2, args.k_syn+1)]
	modified_operators = [""] + [str(s) for s in suffixes]

	for filename in files:
		if ".txt" in filename:
			with open(os.path.join(args.read_dir, filename)) as rf:
				with open(os.path.join(args.write_dir, filename), "w+") as wf:
					for line in rf:
						sent = line.strip()
						tokens = []
						for i in sent.split(" "):
							if i in symbol_set:
								tokens.append("{}{}".format(i,random.choice(modified_operators)))
							else:
								tokens.append(i)
						wf.write(" ".join(tokens))
						wf.write("\n")
