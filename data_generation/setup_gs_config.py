import os
import argparse
import numpy as np
import itertools

"""
Set up config files for each run in our grid search sweep.

Makes nested folders at output/$EXPERIMENT_NAME with the following directory structure:

- completed_settings
- logs
- config
- - 1.txt
- - 2.txt
- results
- - 1
- - 2

Model files and perplexities will be found in the results directory for that run number.
"""


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str)
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--cutoff", type=int, default=-1)
parser.add_argument("--gs_seed", type=int, default=2522524) #For randomly shuffling results
args = parser.parse_args()
np.random.seed(args.gs_seed)

config_specification = {}
with open(args.config_file) as rf:
	for line in rf:
		field, value = line.strip().split("\t")
		if "," in value:
			value = value.split(",")
		config_specification[field] = value

if not "experiment_folder" in config_specification:
	print("no output specified")
	sys.exit(1)


experiment_dir = os.path.join("output",config_specification["experiment_folder"])
config_path = os.path.join(experiment_dir, "config")

if not os.path.exists(experiment_dir):
	os.makedirs(experiment_dir)
	os.makedirs(config_path)
	os.makedirs(os.path.join(experiment_dir, "completed_settings"))
	os.makedirs(os.path.join(experiment_dir, "results"))
	os.makedirs(os.path.join(experiment_dir, "logs"))
elif args.overwrite == False:
	print("dir already exists! rm -rf or --overwrite to continue")
	sys.exit(1)


config_path = os.path.join(experiment_dir, "config")


values = [config_specification[x] if type(config_specification[x]) == list else [config_specification[x]] for x in config_specification.keys()]

iterator = [list(zip(config_specification.keys(), y)) for y in itertools.product(*values)]
if args.shuffle == True:
	np.random.shuffle(iterator)
for i, config in enumerate(iterator):
	print(config)
	with open(os.path.join(config_path, "{}.txt".format(i+1)), "w+") as wf:
		for field, value in config:
			wf.write("{}\t{}\n".format(field,value))
	if args.cutoff != -1 and args.cutoff < i:
		break

print(i+1, "out of", len(iterator))
