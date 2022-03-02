def truth(formula):
	#should be ( V1 OP V2 ), string length 5
	if len(formula) == 3: # if it's ( V1 )
		return formula[1]
	v1 = formula[1]
	op = formula[2]
	v2 = formula[3]
	if op == "|" and (v1 == "1" or v2 == "1"): #or
		return "1"
	elif op == ">" and not (v1 == "1" and v2 == "0"): #entails
		return "1"
	elif op == "&" and (v1 == "1" and v2 == "1"): #and
		return "1"
	else:
		return "0"

def flip(val):
	if val == "1":
		return "0"
	else:
		return "1"


def evaluate_sentence(sentence):
	"""
		Code that evaluates propositional logic sentence in parenthetical tree structure.

		input: sentence (str or list of str)
			precondition: sentence only contains characters from "()&|>~10" or spaces (which will be split)
		output: True if sentence evaluates to True, False otherwise
	"""
	if type(sentence) == str:
		if " " in sentence:
			sentence = sentence.split(" ")
		else:
			sentence = list(sentence)
	while len(sentence) > 1:
		leftmost_rbracket = sentence.index(")")
		leftside = sentence[:leftmost_rbracket+1]
		rightmost_lbracket = len(leftside) - 1 - leftside[::-1].index("(")
		if sentence[rightmost_lbracket - 1] == "~": # ~
			sentence = sentence[0:rightmost_lbracket - 1] + [flip(truth(sentence[rightmost_lbracket:leftmost_rbracket + 1]))] + sentence[leftmost_rbracket+1:]
		else:
			sentence = sentence[0:rightmost_lbracket] + [truth(sentence[rightmost_lbracket:leftmost_rbracket + 1])] + sentence[leftmost_rbracket+1:]
	if sentence[0] == "1":
		return True
	else:
		return False

def getdict_list(bin_vector, symbol_list):
	"""
	input:
		bin_vector: string of "1" and "0"s
		symbol_list: list of strings
	output:
		variable_assignment: dict of {str : bool}

	Takes as input binary vector and list of symbols, and returns a variable assignment.

	For example:
		bin_vector = "001"
		symbol_list = ["Symbol_33", "Symbol_891", "Symbol_2"]
	output:
		variable_assignment = {"Symbol_33": False, "Symbol_891": False, "Symbol_2": True}
	"""

	variable_assignment = {}
	for idx, val in enumerate(bin_vector):
		item = symbol_list[idx]
		if val == "1":
			variable_assignment[item] = True
		elif val == "0":
			variable_assignment[item] = False
		else:
			variable_assignment[item] = val
	return variable_assignment


def getdict_char(bin_vector, charmap = "abcdefghijklmnopqrstuvwxyz"):
	variable_assignment = {}
	for idx, val in enumerate(bin_vector):
		c = charmap[idx]
		if val == "1":
			variable_assignment[c] = True
		elif val == "0":
			variable_assignment[c] = False
		else:
			variable_assignment[c] = val
	return variable_assignment

def eval_assignment_in_world(sentence, variable_dict):
	#precondition: sentence is list, and is split on space
	if type(sentence) is str:
		if " " in sentence:
			sentence = sentence.split(" ")
		elif "Symbol_" not in sentence: #This is for when it's alpha chars e.g. ~(a|b)
			sentence = list(sentence)
		else:
			sentence = [sentence] #Single-symbol sentences e.g. "Symbol_891" need to also be a list
	#key: variable. value: T/F, 0/1
	assert type(variable_dict) is dict
	for variable in variable_dict.keys():
		value = variable_dict[variable]
		if value == True:
			value = 1
		elif value == False:
			value = 0
		sentence = [str(value) if x == variable else x for x in sentence]
	if not all([x in "()&|>~10 " for x in sentence]):
		print("Not all variables in sentence were assigned value.")
		return None
	else:
		return evaluate_sentence(sentence)
