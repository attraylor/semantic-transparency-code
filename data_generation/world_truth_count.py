import sys
import os

import minimal_proplogic as apl
import random
import numpy as np
import sys
import os
import time

import unittest

def generate_random_world(i):
	if type(i) == int:
		return [random.choice(["0","1"]) for j in range(i)]
	if type(i) == list:
		return {j:random.choice([True, False])for j in i}

def w2a(world):
	#world to array of bools
	return [True if x == "1" else False for x in world]

def a2w(world):
	#array of bools to str
	return "".join(["1" if x == True else "0" for x in world])

def arr_to_world(world):
	#Takes array of characters (optionally negated with "~")
	#returns bit vector of 0/1s in corresponding order
	ret_str = ""
	for symbol in world:
		if "~" in symbol:
			ret_str += "0"
		else:
			ret_str += "1"
	return ret_str

def get_relevant_variables(premise, idx=True):
	#return alphabetical pos of each var that is in the premise.
	cmap = "abcdefghijklmnopqrstuvwxyz"
	arr = []
	for i in range(len(cmap)):
		if cmap[i] in premise:
			if idx == True:
				arr.append(i)
			else: #return char itself
				arr.append(cmap[i])
	return arr#[i if (cmap[i] in premise) for i in range(len(cmap))]

def world_in_seen(world, relevant_variables, seen):
	#world: bool array size 26 corresponding to world bits
	#relevant_variables: int array, each int is var included in premise
	#seen: array of int arrays. each int array length of relevant_variables.

	#compares bit arrays, hopefully this is the fastest possible
	#print(world, relevant_variables, seen, True if [world[i] for i in relevant_variables] in seen else False)
	return True if [world[i] for i in relevant_variables] in seen else False

def assign_world(world, vars):
	ret_world = ["0"] * 26
	for i in range(0, len(world)):
		ret_world[vars[i]] = world[i]
	return "".join(ret_world)

def combine_vars(v1, v2):
	v = v1
	for var in v2:
		if var not in v:
			v.append(var)
	return v

def in_both(v1, v2):
	v = []
	for var in v2:
		if var in v1:
			v.append(var)
	return v

def convert_sentence_pair_to_az(s1, s2):
	comb = s1 + "\t" + s2
	comb_replaced, map = convert_sentence_to_az(comb)
	s1_new, s2_new = comb_replaced.split("\t")
	return s1_new, s2_new, map

def convert_sentence_to_az(sentence):
	if not "Symbol_" in sentence:
		return sentence, [] #already in az format
	else:
		split_sentence = sentence.split(" ")
		ind = 0
		symb_to_az_list = [] #should be symb1 symb2 symb3 -> abc
		while ind < len(split_sentence):
			if "Symbol_" in split_sentence[ind]:
				current_symbol = split_sentence[ind]
				symb_to_az_list.append(current_symbol)
				char_mapped = chr(97 + len(symb_to_az_list) - 1)
				split_sentence = list(map(lambda x: x if x != current_symbol else char_mapped, split_sentence))
			elif len(split_sentence[ind]) > 1 and split_sentence[ind][0] in "()&|~>":
				split_sentence[ind] = split_sentence[ind][0]
			ind += 1

	return "".join(split_sentence), symb_to_az_list

class TestConvertSentence(unittest.TestCase):

	def test_pass_forward(self):
		self.assertEqual(convert_sentence_to_az("((a&b)|c)")[0], "((a&b)|c)")
		self.assertEqual(convert_sentence_to_az("((a&b)|c)")[1], [])

	def test_convert(self):
		self.assertEqual(convert_sentence_to_az("( ( Symbol_1 & Symbol_2 ) | Symbol_3 )")[0], "((a&b)|c)")
		self.assertEqual(convert_sentence_to_az("( ( Symbol_1 & Symbol_2 ) | Symbol_1 )")[0], "((a&b)|a)")

def dict_to_world(az_map, symbol_map):
	cmap = "abcdefghijklmnopqrstuvwxyz"
	ret_world = ["0"] * 26
	for symbol in symbol_map.keys():
		if symbol_map[symbol] == True:
			bit = "1"
		else:
			bit = "0"
		ret_world[az_map.index(symbol)] = bit
	return ret_world


def sentence_satisfiable_in_world(sentence, world):
	az_sentence, az_mapping_list = convert_sentence_to_az(sentence)
	relevant_variables = get_relevant_variables(az_sentence)
	if type(world) == list:
		world = assign_world(world, relevant_variables)
	elif type(world) == dict:
		world = dict_to_world(az_mapping_list, world)
	truthval = apl.parse(az_sentence, world)
	return truthval


def satisfying_assignment_exists(sentence, symbols):
	"""
	Determine whether an assignment of symbols exists that satisifes the sentence

	input:
		sentence: space_split string of symbols, parens, and operators
		symbols: list of unique non-operator symbols in `sentence`
	output:
		truthval: True if assignment of symbols exists that satisfies sentence, False otherwise
		assignment: dict of {string : bool} values dictating the symbol:truth value variable assignment that satisfies sentence
	"""

	num_symbols = len(symbols)
	str_num_symbols = str(num_symbols)
	for i in range(2 ** num_symbols):
		world = ('{0:0' + str_num_symbols + 'b}').format(i)
		variable_assignment = apl.getdict_list(world, symbols)
		truthval = apl.eval_assignment_in_world(sentence, variable_assignment)
		if truthval == True:
			return True, variable_assignment
	return False, {}


def all_satisfying_assignments(sentence, symbols):
	"""
	Determine whether an assignment of symbols exists that satisifes the sentence

	input:
		sentence: space_split string of symbols, parens, and operators
		symbols: list of unique non-operator symbols in `sentence`
	output:
		assignments: list of dict of {string : bool}. each dict is values dictating the symbol:truth value variable assignment that satisfies sentence
					 if no assignments satisfy, then return empty list
	"""
	satisfying_assignments = []
	num_symbols = len(symbols)
	str_num_symbols = str(num_symbols)
	for i in range(2 ** num_symbols):
		world = ('{0:0' + str_num_symbols + 'b}').format(i)
		variable_assignment = apl.getdict_list(world, symbols)
		truthval = apl.eval_assignment_in_world(sentence, variable_assignment)
		if truthval == True:
			satisfying_assignments.append(variable_assignment)
	return satisfying_assignments

class TestFindSatisfyingArgument(unittest.TestCase):

	def test_true(self):
		self.assertTrue(satisfying_assignment_exists("( ( Symbol_1 & Symbol_2 ) | Symbol_3 )")[0])
		self.assertEqual(satisfying_assignment_exists("( Symbol_1 & Symbol_2 )")[1], {"Symbol_1": True, "Symbol_2": True})

	def test_false(self):
		self.assertFalse(satisfying_assignment_exists("( Symbol_1 & ~ ( Symbol_1 ) )")[0])

	def test_az(self):
		self.assertTrue(satisfying_assignment_exists("(a&b)"))
		self.assertFalse(satisfying_assignment_exists("(a&~(a))"))



def eval(split, verbose = False):
	prem_vars = get_relevant_variables(split[0])
	hyp_vars = get_relevant_variables(split[1])
	inb = in_both(prem_vars, hyp_vars)
	relevant_variables = combine_vars(prem_vars, hyp_vars)
	l_t = len(relevant_variables)
	s_lt = str(l_t)
	seen = [] #map of int (representing line # for train/test items) to seen worlds
	premise = split[0]
	hypothesis = split[1]

	prem_true = 0
	prem_false = 0
	hyp_true = 0
	hyp_false = 0
	both_true = 0
	#seen_ml = False #have we seen the most likely world yet?
	for i in range(2 ** l_t):
		world = ('{0:0' + s_lt + 'b}').format(i)
		world = assign_world(world, relevant_variables)
		truthval = apl.parse(premise, world)
		hyp_is_true = apl.parse(hypothesis, world)
		if hyp_is_true == True:
			hyp_true += 1
		else:
			hyp_false += 1
		if truthval == True:
			prem_true += 1
		else:
			prem_false += 1
		if hyp_is_true == True and truthval == True:
			both_true += 1
	p_p = prem_true * 1.0 / (prem_true + prem_false)
	p_h = hyp_true * 1.0 / (hyp_true + hyp_false)
	p_b = both_true * 1.0 / (hyp_true + hyp_false)
	try:
		p_cond = p_b / p_p
	except ZeroDivisionError:
		p_cond = 0
	if verbose == True:
		print("Premise true for:\t{}".format(prem_true))
		print("Premise false:\t{}".format(prem_false))
		print("Hypothesis true for:\t{}".format(hyp_true))
		print("Hypothesis false:\t{}".format(hyp_false))
		print("Both true:\t{}".format(both_true))
	return p_p, p_h, p_b, p_cond

def eval2(split, verbose = False):
	premise = split[0]
	hypothesis = split[1]
	'''if type(premise) == list:
		premise = " ".join(premise)
		hypothesis = " ".join(hypothesis)'''
	if "Symbol_" in premise:
		premise, hypothesis,map = convert_sentence_pair_to_az(premise,hypothesis)
	prem_vars = get_relevant_variables(premise)
	hyp_vars = get_relevant_variables(hypothesis)
	inb = in_both(prem_vars, hyp_vars)
	relevant_variables = combine_vars(prem_vars, hyp_vars)
	l_t = len(relevant_variables)
	s_lt = str(l_t)
	seen = [] #map of int (representing line # for train/test items) to seen worlds

	prem_true = 0
	prem_false = 0
	hyp_true = 0
	hyp_false = 0
	both_true = 0
	entailed = True
	equiv = True
	#seen_ml = False #have we seen the most likely world yet?
	for i in range(2 ** l_t):
		world = ('{0:0' + s_lt + 'b}').format(i)
		world = assign_world(world, relevant_variables)
		truthval = apl.parse(premise, world)
		hyp_is_true = apl.parse(hypothesis, world)
		if hyp_is_true == True:
			hyp_true += 1
		else:
			hyp_false += 1
		if truthval == True:
			if hyp_is_true == False:
				entailed = False
			prem_true += 1
		else:
			prem_false += 1
		if hyp_is_true == True and truthval == True:
			both_true += 1
		elif hyp_is_true != truthval:
			equiv = False
	p_p = prem_true * 1.0 / (prem_true + prem_false)
	p_h = hyp_true * 1.0 / (hyp_true + hyp_false)
	p_b = both_true * 1.0 / (hyp_true + hyp_false)
	try:
		p_cond = p_b / p_p
	except ZeroDivisionError:
		p_cond = 0
	if verbose == True:
		print("Premise true for:\t{}".format(prem_true))
		print("Premise false:\t{}".format(prem_false))
		print("Hypothesis true for:\t{}".format(hyp_true))
		print("Hypothesis false:\t{}".format(hyp_false))
		print("Both true:\t{}".format(both_true))
	return p_p, p_h, p_b, p_cond, entailed, equiv


if __name__ == "__main__":
	unittest.main()
