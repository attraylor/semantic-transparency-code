import os
import argparse
import sys
from config_utils import create_config
sys.path.append(".")
from main import main


"""
Simple wrapper for configs made by grid search. Parses the config and calls main
"""

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str)
parser.add_argument("--taskid", type=int)
parser.add_argument("--seed_is_taskid", action="store_true")
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()



config_path = os.path.join("output", args.experiment_name, "config", "{}.txt".format(args.taskid))
results_path = os.path.join("output", args.experiment_name, "results", "{}".format(args.taskid))
completed_path = os.path.join("output", args.experiment_name, "completed_settings", "{}.txt".format(args.taskid))

if os.path.exists(completed_path) and not args.overwrite == True:
	print("Run already completed!")
	sys.exit(1)
else:
	config_specification = create_config(config_path)
	config_specification["write_dir"] = results_path
	config_specification["task_id"] = args.taskid
	experiment_name = args.experiment_name
	print(config_specification)

	#This is true if you want seed to equal config ID. e.g. job 1 = seed 1
	if args.seed_is_taskid == True:
		config_specification["seed"] = args.taskid
	main(config_specification)

	with open(completed_path, "w+") as wf:
		wf.close()
