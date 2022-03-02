import os
import argparse
import numpy as np
import sys
from config_utils import create_config
sys.path.append(".")
import main

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str)
parser.add_argument("--taskid", type=int)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--test_dir", type=str, default="data/pcfg/acl/test")
args = parser.parse_args()

exp_type = "pretraining"
config_path = os.path.join("output", args.experiment_name, "results", format(args.taskid), "models", exp_type, "config.txt")

'''if "experiment_label" in args.keys() and args["experiment_label"] != "":
	test_path = os.path.join("test_results", args["experiment_label"])
else:
	'''

config_specification = create_config(config_path)
config_specification["task_id"] = args.taskid
config_specification["probe"] = True
if args.cpu == True:
	config_specification["device"] = "cpu"
config_specification["probewrite_path"] = os.path.join("output", args.experiment_name, "results", format(args.taskid), "models", exp_type, "generation.txt")
config_specification["pretraining_dir"] = args.test_dir
main.main(config_specification)
