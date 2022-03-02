import unittest
from data_generation.minimal_proplogic import *

class Test(unittest.TestCase):

	def test_getdict_list(self):
		self.assertEqual(getdict_list("01", ["Symbol_1", "Symbol_2"]), {"Symbol_1": False, "Symbol_2": True})
		self.assertEqual(getdict_list("10", ["Symbol_10", "Symbol_100"]), {"Symbol_10": True, "Symbol_100": False})

	def test_evaluate(self):
		self.assertTrue(evaluate_sentence("1"))
		self.assertTrue(evaluate_sentence("(1&1)"))
		self.assertTrue(evaluate_sentence("(1|1)"))
		self.assertTrue(evaluate_sentence("(0|1)"))
		self.assertTrue(evaluate_sentence("(1|0)"))
		self.assertTrue(evaluate_sentence("~(0|0)"))
		self.assertTrue(evaluate_sentence("~(0)"))
		self.assertTrue(evaluate_sentence("((1&1)|0)"))
		self.assertTrue(evaluate_sentence("((1&0)|1)"))


		self.assertFalse(evaluate_sentence("0"))
		self.assertFalse(evaluate_sentence("(0&1)"))
		self.assertFalse(evaluate_sentence("(1&0)"))
		self.assertFalse(evaluate_sentence("~(1)"))
		self.assertFalse(evaluate_sentence("~(1&1)"))
		self.assertFalse(evaluate_sentence("~(1|0)"))
		self.assertFalse(evaluate_sentence("~(0|1)"))

	def test_eval_assignment_in_world(self):
		sentence0 = "~ ( Symbol_1 & Symbol_2 )"
		sentence0_true_dict = getdict_list("01", ["Symbol_1", "Symbol_2"])
		sentence0_false_dict = getdict_list("11", ["Symbol_1", "Symbol_2"])

		self.assertTrue(eval_assignment_in_world(sentence0, sentence0_true_dict))
		self.assertFalse(eval_assignment_in_world(sentence0, sentence0_false_dict))


		sentence1 = "~ ( Symbol_3 | Symbol_4 )"
		sentence1_true_dict = getdict_list("00", ["Symbol_4", "Symbol_3"])
		sentence1_false_dict = getdict_list("10", ["Symbol_4", "Symbol_3"])

		self.assertTrue(eval_assignment_in_world(sentence1, sentence1_true_dict))
		self.assertFalse(eval_assignment_in_world(sentence1, sentence1_false_dict))

		sentence2 = "( Symbol_10 & ~ ( Symbol_1010 | Symbol_1010 ) )"
		sentence2_true_dict = getdict_list("01", ["Symbol_1010", "Symbol_10"])
		sentence2_false_dict = getdict_list("10", ["Symbol_1010", "Symbol_10"])

		self.assertTrue(eval_assignment_in_world(sentence2, sentence2_true_dict))
		self.assertFalse(eval_assignment_in_world(sentence2, sentence2_false_dict))


		sentence3 = "( ~ ( Symbol_9 ) | ( Symbol_99 & Symbol_999 ) )"
		sentence3_true_dict = getdict_list("111", ["Symbol_99", "Symbol_999", "Symbol_9"])
		sentence3_false_dict = getdict_list("001", ["Symbol_99", "Symbol_999", "Symbol_9"])

		self.assertTrue(eval_assignment_in_world(sentence3, sentence3_true_dict))
		self.assertFalse(eval_assignment_in_world(sentence3, sentence3_false_dict))


		sentence4 = "( Symbol_3021 )"
		sentence4_true_dict = getdict_list("1", ["Symbol_3021"])
		sentence4_false_dict = getdict_list("0", ["Symbol_3021"])

		self.assertTrue(eval_assignment_in_world(sentence4, sentence4_true_dict))
		self.assertFalse(eval_assignment_in_world(sentence4, sentence4_false_dict))




if __name__ == "__main__":
	unittest.main()
