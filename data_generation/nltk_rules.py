from nltk.grammar import PCFG, Nonterminal, ProbabilisticProduction
from nltk.parse import generate
import random
#https://stackoverflow.com/questions/15009656/how-to-use-nltk-to-generate-sentences-from-an-induced-grammar
from nltk.tree import Tree

def tree_from_production(production):
	return Tree(production.lhs(), production.rhs())

def leaf_positions(the_tree):
	return [the_tree.leaf_treeposition(i) for i in range(len(the_tree.leaves()))]

def generate_tree(grammar):
	initial_derivations = grammar._lhs_index[grammar.start()]
	initial_derivation = weighted_choice(initial_derivations) # or weighed_choice if you have that function
	running_tree = tree_from_production(initial_derivation)
	all_terminals = False
	while not all_terminals:
		all_terminals = True
		for position in leaf_positions(running_tree):
			node_label = running_tree[position]
			depth = len(position)
			if node_label in grammar._lhs_index:
				all_terminals = False
				derivations = grammar._lhs_index[node_label]
				derivation = weighted_choice(derivations) #or weighed_choice if you have that function
				running_tree[position] = tree_from_production(derivation)

	return running_tree

def weighted_choice(productions):
	prods_with_probs = [(prod, prod.prob()) for prod in productions]
	total = sum(prob for prod, prob in prods_with_probs)
	r = random.uniform(0, total)
	upto = 0
	for prod, prob in prods_with_probs:
		if upto + prob >= r:
			return prod
		upto += prob
	assert False, "Shouldn't get here"



(S0, S1, S2, N1, N2, Symb) = [Nonterminal(s) for s in 'S0 S1 S2 N1 N2 Symb'.split()]

rules = {}

#TODO: Make this work for either any number of ksyns, or just 1
def tree_from_hyperparam(S0, gamma):
	assert gamma > 0 and gamma < 1
	nonterminals = [S0]
	rules = []
	for i in range(1, 10):
		next_layer = Nonterminal("S{}".format(i))
		previous_layer = nonterminals[-1]
		prob = gamma ** i
		num_connectives = 3
		connective_prob = prob * 1.0 / num_connectives
		symbol_prob = 1 - prob
		rules.append(ProbabilisticProduction(previous_layer,
											["( " , next_layer," & ", next_layer, " )"],
											prob=connective_prob))
		rules.append(ProbabilisticProduction(previous_layer,
											["( " , next_layer," | ", next_layer, " )"],
											prob=connective_prob))
		rules.append(ProbabilisticProduction(previous_layer,
											["~ ( " , next_layer, " )"],
											prob=connective_prob))
		rules.append(ProbabilisticProduction(previous_layer, ["Symbol_FILLER"], prob = symbol_prob))
		nonterminals.append(next_layer)
	rules.append(ProbabilisticProduction(nonterminals[-1], ["Symbol_FILLER"], prob = 1))
	return rules
