import os
def isFloat(s):
	return s.replace('.','',1).isdigit()

def isInteger(s):
	return s.isdigit()


def create_config(path, exclude_dir=True):
	config_specification = {}
	with open(os.path.join(path)) as rf:
		for line in rf:
			field, value = line.strip().split("\t")
			if field == "learning_rate":
				 config_specification[field] = float(value)
			if value == "True":
				config_specification[field] = True
			elif value == "False":
				config_specification[field] = False
			elif isInteger(value):
				config_specification[field] = int(value)
			elif isFloat(value):
				config_specification[field] = float(value)
			else:
				config_specification[field] = value

	if exclude_dir == True:
		try:
			config_specification.pop("dir")
		except KeyError:
			pass
	return config_specification

def load_vocab(path):
	imported_vocab = {}
	itos = {}
	#for ct, word in enumerate(special_tokens):
	#	imported_vocab[word] = ct
	try:
		with open(path) as rf:
			for ct, line in enumerate(rf):
				idx, word = line.strip().split("\t")
				imported_vocab[word] = int(idx)
				itos[int(idx)] = word
	except ValueError:
		with open(path) as rf:
			for ct, line in enumerate(rf):
				word, idx = line.strip().split("\t")
				imported_vocab[word] = int(idx)
				itos[int(idx)] = word
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
