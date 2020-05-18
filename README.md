# Thesis 2020 Martijn Hooijman

# Installation

Create a virtual environment
```
python3 -m venv venv
```

Clone DeepSpeech from a forked repository (original DeepSpeech repository: https://github.com/mozilla/DeepSpeech), checkout v0.6.1 and install dependencies.
```
git clone https://github.com/mhooijman/DeepSpeech.git ./DeepSpeech
cd DeepSpeech && git checkout v0.6.1 && pip install -r requirements.txt
pip install $(python ./util/taskcluster.py --decoder)
pip install tensorflow-gpu==1.15
```

Download DeepSpeech v0.6.1 model checkpoints
```
mkdir model
wget "https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/deepspeech-0.6.1-models.tar.gz" -O ./model/model.tar.gz
unzip ./model/model.tar.gz
rm -f ./model/model.tar.gz
```

Test DeepSpeech installation
```
cd DeepSpeech && python DeepSpeech.py --checkpoint_dir ./models/deepspeech-0.6.1-models/checkpoint/ \
--alphabet_config_path=./DeepSpeech/data/alphabet.txt --one_shot_infer ./data/LDC93S1.wav
```

Download Librivox data (python file handles already exist check)
```
python util/import_data.py ./data
```















