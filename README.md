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
sudo apt-get install sox libsox-fmt-all


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


Download Mozilla's Common Voice English dataset
```
wget "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz" -O ./data/en.tar.gz
mkdir CommonVoice
tar -C ./data/CommonVoice -xf ./data/en.tar.gz && rm -f ./data/en.tar.gz
python utils/import_common_voice_en_data.py
rm -rf ./data/CommonVoice/EN/clips # optional to remove unused files


Download Librivox data (python file handles already exist check). From the root folder execute:
```
python utils/import_data.py ./data
rm -rf ./data/LibriSpeech/test-clean
```

## Neuron importance scores
Run the code for obtaining activations of the Common Voice data on the full model:
```
python activations.py --checkpoint_dir ./deepspeech-0.6.1-checkpoint --alphabet_config_path ./DeepSpeech/data/alphabet.txt --lm_binary_path ""--lm_trie_path ""
```

The script fails after obtaining activations for the Common Voice data on the full model as the pruning information is not ready. Run the following to compute the importance scores.
```
python process_activations.py --checkpoint_dir ./deepspeech-0.6.1-checkpoint --alphabet_config_path ./DeepSpeech/data/alphabet.txt --lm_binary_path ""--lm_trie_path ""
```

And re-run the activations code above. It will automatically skip those activations that are already processed.

## Evaluation with pruning
Run the code for evaluation with pruning (standard at .05, .1, .2 pruning percentages).
```
python evaluate_pruning.py --checkpoint_dir ./deepspeech-0.6.1-checkpoint --alphabet_config_path ./DeepSpeech/data/alphabet.txt --lm_binary_path ""--lm_trie_path ""
```

## Encoding experiments
Run the code for conducting the encoding experiments of gender and sentence length.
```
python encoding_experiments.py --checkpoint_dir ./deepspeech-0.6.1-checkpoint --alphabet_config_path ./DeepSpeech/data/alphabet.txt --lm_binary_path ""--lm_trie_path ""
```


## Figures
Run the code for obtaining figures and additional information (figures are saved to ./figures).
```
python plots.py
```
















