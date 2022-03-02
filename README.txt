Code for "AND does not mean OR: Using Formal Languages to Study Language Models’ Representations"

Requirements:
python 3.7.4

blis==0.7.4
catalogue==1.0.0
certifi==2020.6.20
chardet==3.0.4
click==7.1.2
cycler==0.10.0
cymem==2.0.5
EditorConfig==0.12.3
en-core-web-sm==2.3.1
future==0.18.2
idna==2.10
importlib-metadata==3.4.0
joblib==0.17.0
jsbeautifier==1.13.0
kiwisolver==1.2.0
matplotlib==3.3.2
murmurhash==1.0.5
nltk==3.5
numpy==1.19.2
pandas==1.1.3
Pillow==7.2.0
plac==1.1.3
preshed==3.0.5
psutil==5.7.2
pyparsing==2.4.7
python-dateutil==2.8.1
pytz==2020.1
regex==2020.9.27
requests==2.24.0
scikit-learn==0.23.2
scipy==1.5.2
seaborn==0.11.0
sentencepiece==0.1.92
six==1.15.0
spacy==2.3.5
srsly==1.0.5
thinc==7.4.5
threadpoolctl==2.1.0
toma==1.1.0
torch==1.6.0
torchtext==0.7.0
tqdm==4.50.2
typing-extensions==3.7.4.3
urllib3==1.25.10
wasabi==0.8.0
zipp==3.4.0


To generate datasets:

Syntactic:

python data_generation/pcfg_data_generation.py --dir data/pcfg/syntactic --num_examples 100000 --vocab_stop 5000 --filename train.txt --unique_symbols_param 2.0 --depth_param .85 --only_syntactic_constraint

Truthfulness:

python data_generation/pcfg_data_generation.py --dir data/pcfg/truthfulness --num_examples 100000 --vocab_stop 5000 --filename train.txt --unique_symbols_param 2.0 --depth_param .85 --seed 2

Informativity:

python data_generation/informativity_data_generation.py --dir data/pcfg/informativity2T2A --num_examples 110000 --vocab_size 5000 --unique_symbols_param 2.0 --num_observed_worlds 2 --num_alternative_worlds 2

Explicit Grounding:

python data_generation/pcfg_data_generation.py --dir data/pcfg/grounding --num_examples 100000 --vocab_stop 5000 --filename train.txt --unique_symbols_param 2.0 --depth_param .85 --seed 4 --grounding_map

Make the vocab file:
python data_generation/make_vocab.py --symbs 5000 --op_synonyms 5 --write_path vocab/ksyn5_5000_vocab.txt

Make the test set:
python data_generation/generate_test_set.py --cutoff 1000
(data/test/train.txt will be blank-- this is normal)

Then set number of synonyms to 5:

ksyn=5
python data_generation/generate_synonyms_folders.py --read_dir data/pcfg/syntactic/data --write_dir data/pcfg/ksyn_{$ksyn}_syntactic/data --k_syn $ksyn
python data_generation/generate_synonyms_folders.py --read_dir data/pcfg/truthfulness/data --write_dir data/pcfg/ksyn_{$ksyn}_truthfulness/data --k_syn $ksyn
python data_generation/generate_synonyms_folders.py --read_dir data/pcfg/informativity2T2A/data --write_dir data/pcfg/ksyn_{$ksyn}_informativity2T2A/data --k_syn $ksyn
python data_generation/generate_synonyms_folders.py --read_dir data/pcfg/grounding/data --write_dir data/pcfg/ksyn_{$ksyn}_grounding/data --k_syn $ksyn


Then set up the grid searches:

python data_generation/setup_gs_config.py --config_file config/small_lstm.txt --overwrite
python data_generation/setup_gs_config.py --config_file config/medium_lstm.txt --overwrite
python data_generation/setup_gs_config.py --config_file config/small_transformer.txt --overwrite
python data_generation/setup_gs_config.py --config_file config/medium_transformer.txt --overwrite

Then run the models (using some sort of scheduler to iterate on runs 1-20 for each model)

python src/grid_search_wrapper.py --experiment_name acl-proplogic-ksyn5-small-lstm --taskid $ARRAY_TASK_ID
python src/grid_search_wrapper.py --experiment_name acl-proplogic-ksyn5-medium-lstm --taskid $ARRAY_TASK_ID
python src/grid_search_wrapper.py --experiment_name acl-proplogic-ksyn5-small-transformer --taskid $ARRAY_TASK_ID
python src/grid_search_wrapper.py --experiment_name acl-proplogic-ksyn5-medium-transformer --taskid $ARRAY_TASK_ID


Then do the tests (using some sort of scheduler to iterate on runs 1-20 for each model)

python src/acl_test_wrapper.py --experiment_name acl-proplogic-ksyn5-small-lstm --taskid $ARRAY_TASK_ID
python src/acl_test_wrapper.py --experiment_name acl-proplogic-ksyn5-medium-lstm --taskid $ARRAY_TASK_ID
python src/acl_test_wrapper.py --experiment_name acl-proplogic-ksyn5-small-transformer --taskid $ARRAY_TASK_ID
python src/acl_test_wrapper.py --experiment_name acl-proplogic-ksyn5-medium-transformer --taskid $ARRAY_TASK_ID

To evaluate the tests, do:
python data_generation/ppl.py --config config/graph_config.txt

Then do the nearest neighbor probing classifier:
python data_generation/nearest_neighbor_probing_classifier.py --config config/graph_config.txt


Then do the PCA:
python data_generation/pca.py --config config/graph_config.txt --model_name medium-transformer


good stuff :)
