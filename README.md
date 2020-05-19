# Thesis 2020 Martijn Hooijman

# Installation

Create and enter a virtual environment.
```
python3 -m venv venv
source venv/bin/activate
```

Clone DeepSpeech from a forked repository and install, checkout v0.6.1 and install dependencies (original DeepSpeech repository: https://github.com/mozilla/DeepSpeech). Original docs: https://github.com/mhooijman/DeepSpeech/blob/3df20fee52fda47d08e3726fd0da86dbb414e9d8/doc/TRAINING.rst#training-your-own-model.

Summary of installation (make use of git-lfs).
```
git clone https://github.com/mhooijman/DeepSpeech.git ./DeepSpeech
cd DeepSpeech && git checkout v0.6.1 && pip install -r requirements.txt
pip install $(python ./util/taskcluster.py --decoder)
sudo apt-get install sox

python -m pip install tensorflow-gpu==1.14.0  # optional
```

Download DeepSpeech v0.6.1 model & checkpoints. From the root directory run:
```
mkdir ./model
wget "https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/deepspeech-0.6.1-models.tar.gz" -O ./model/model.tar.gz
tar -xf ./model/model.tar.gz && rm -f ./model/model.tar.gz

wget "https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/deepspeech-0.6.1-checkpoint.tar.gz" -O ./model/checkpoint.tar.gz
tar -xf ./model/checkpoint.tar.gz && rm -f ./model/checkpoint.tar.gz
```

Test DeepSpeech installation. Run the following code from the root folder.
```
python DeepSpeech/DeepSpeech.py --checkpoint_dir ./deepspeech-0.6.1-checkpoint \
--alphabet_config_path ./DeepSpeech/data/alphabet.txt --one_shot_infer ./DeepSpeech/data/smoke_test/LDC93S1.wav \
--lm_binary_path ./DeepSpeech/data/lm/lm.binary --lm_trie_path ./DeepSpeech/data/lm/trie
```

Download Librivox data (python file handles already exist check). From the root folder execute:
```
python utils/import_data.py ./data
rm -rf ./data/LibriSpeech/test-clean
```

## Neuron importance scores
Run the code for neuron importance scores:
```
python neuron_importance_scores.py --checkpoint_dir ./deepspeech-0.6.1-checkpoint --alphabet_config_path ./DeepSpeech/data/alphabet.txt --lm_binary_path ./DeepSpeech/data/lm/lm.binary --lm_trie_path ./DeepSpeech/data/lm/trie
```
Scores for each input are saved to ./results/imp_scores, these results are combined and saved to ./results/final_imp_scores.npy.

## Evaluation with pruning
Run the code for evaluation with pruning (standard at .05, .1, .15, .2, .25, .3 pruning ratio).
```
python evaluate_pruned_model.py --checkpoint_dir ./deepspeech-0.6.1-checkpoint --alphabet_config_path ./DeepSpeech/data/alphabet.txt --lm_binary_path ./DeepSpeech/data/lm/lm.binary --lm_trie_path ./DeepSpeech/data/lm/trie
```

















