import argparse

"""
	Generates the vocab file.
	This will be a file at vocab/ksyn5_5000_vocab.txt
"""


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--symbs", type=int, default=5000)
	parser.add_argument("--op_synonyms", type=int, default=5)
	parser.add_argument("--write_path", type=str, default = "vocab/ksyn5_5000_vocab.txt")
	args = parser.parse_args()
	vocab = []
	vocab.append("<unk>")
	vocab.append("PAD")
	vocab.append("EOS")
	vocab.append("(")
	vocab.append(")")
	vocab.append("<grounding>")
	vocab.append("True")
	vocab.append("False")
	for op in ["&", "|", "~"]:
		vocab.append(op)
		for i in range(2, args.op_synonyms + 1):
			vocab.append("{}{}".format(op, i))
	for symb in range(args.symbs):
		vocab.append("Symbol_{}".format(symb))

	with open(args.write_path, "w+") as wf:
		for ct, word in enumerate(vocab):
			wf.write("{}\t{}\n".format(ct, word))
